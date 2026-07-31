"""Routing, wiring, and end-to-end checks for :class:`AutoMuon`.

The update rules themselves are ``torch.optim.Muon`` and ``torch.optim.AdamW``
and are not retested here; what is tested is that the right parameters reach
the right one and that the composite behaves like a single optimizer.

Run directly::

    python -m dagnabbit.tests.test_optimizers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dagnabbit.dag.graphs import Geometry, sample
from dagnabbit.dag.model import (
    GraphSimulator,
    SimulatorConfig,
    patch_targets,
    sample_patch_indices,
)
from dagnabbit.optimizers import AutoMuon, build_optimizer
from dagnabbit.tasks.logic_gates.evaluate import evaluate_graphs, exhaustive_root_values

GEOMETRY = Geometry(8, 24, 4, 1, (2,))
MODEL_CONFIG = SimulatorConfig(
    embedding_dim=64,
    attention_head_dim=32,
    mlp_expansion_factor=2.0,
    num_simulator_layers=2,
    num_decoder_layers=2,
    num_patches=8,
)


def build_small_model() -> GraphSimulator:
    return GraphSimulator(GEOMETRY, MODEL_CONFIG)


def build_optimizer_for(model: nn.Module, **kwargs) -> AutoMuon:
    kwargs.setdefault("adam_module_names", ("decoder.head",))
    return AutoMuon(model, **kwargs)


def routed_names(optimizer: AutoMuon) -> tuple[set[str], set[str]]:
    return set(optimizer.muon_parameter_names), set(optimizer.adam_parameter_names)


def test_routing_covers_every_parameter_exactly_once():
    model = build_small_model()
    optimizer = build_optimizer_for(model)
    muon_names, adam_names = routed_names(optimizer)

    assert not (muon_names & adam_names), "a parameter was routed to both rules"

    # Tied/shared tensors are deduped by identity, so compare against the
    # deduped set of trainable parameter names.
    expected: set[str] = set()
    seen_ids: set[int] = set()
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in seen_ids:
            seen_ids.add(id(param))
            expected.add(name)
    assert muon_names | adam_names == expected


def test_muon_takes_transformer_matrices_and_nothing_else():
    model = build_small_model()
    optimizer = build_optimizer_for(model)
    muon_names, adam_names = routed_names(optimizer)

    # Attention projections and MLP weights inside every transformer stack.
    assert "simulator.blocks.0.attention.qkv.weight" in muon_names
    assert "simulator.blocks.0.attention.projection.weight" in muon_names
    assert "simulator.blocks.0.mlp.0.weight" in muon_names
    assert "decoder.blocks.1.attention.to_key_value.weight" in muon_names
    # The node embedder's role projections are ordinary hidden-space matrices.
    assert "node_tokens.self_projection.weight" in muon_names
    assert "node_tokens.parent_projections.0.weight" in muon_names

    # Lookup tables, 2-D learned token banks, norms, biases and the named
    # output head all belong to AdamW. The token banks are the cases a plain
    # ``p.ndim == 2`` split would get wrong.
    assert "node_tokens.type_embeddings.weight" in adam_names
    assert "node_tokens.position_embeddings" in adam_names
    assert "node_tokens.null_parent" in adam_names
    assert "decoder.patch_queries" in adam_names
    assert "decoder.head.weight" in adam_names
    assert "decoder.head.bias" in adam_names
    assert "simulator.blocks.0.norm_attention.weight" in adam_names
    assert "simulator.output_norm.bias" in adam_names
    assert "simulator.blocks.0.mlp.0.bias" in adam_names


def test_every_muon_parameter_is_a_2d_weight():
    """torch.optim.Muon rejects non-2-D parameters outright."""
    model = build_small_model()
    optimizer = build_optimizer_for(model)
    for group in optimizer.muon.param_groups:
        for param in group["params"]:
            assert param.ndim == 2, param.shape


def test_adam_module_names_match_paths_prefixes_and_patterns():
    model = build_small_model()
    exact = routed_names(
        AutoMuon(model, adam_module_names=("node_tokens.self_projection.weight",))
    )
    assert "node_tokens.self_projection.weight" in exact[1]
    assert "node_tokens.parent_projections.0.weight" in exact[0]

    subtree = routed_names(
        AutoMuon(model, adam_module_names=("node_tokens.parent_projections",))
    )
    assert "node_tokens.parent_projections.0.weight" in subtree[1]
    assert "node_tokens.self_projection.weight" in subtree[0]

    pattern = routed_names(AutoMuon(model, adam_module_names=("*.qkv.weight",)))
    assert "simulator.blocks.0.attention.qkv.weight" in pattern[1]
    assert "simulator.blocks.0.attention.projection.weight" in pattern[0]


def test_lr_scheduler_drives_both_child_optimizers():
    model = build_small_model()
    optimizer = build_optimizer_for(model, muon_lr=0.02, adam_lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min((step + 1) / 10, 1.0)
    )
    assert sorted(scheduler.base_lrs) == [1e-3, 0.02]

    # The children must see the scheduler's writes, not stale copies.
    assert optimizer.muon.param_groups[0]["lr"] == 0.02 / 10
    assert optimizer.adamw.param_groups[0]["lr"] == 1e-3 / 10
    for _ in range(4):
        scheduler.step()
    assert optimizer.muon.param_groups[0]["lr"] == 0.02 / 2
    assert optimizer.adamw.param_groups[0]["lr"] == 1e-3 / 2


def test_step_updates_both_rules_and_survives_a_state_dict_roundtrip():
    torch.manual_seed(0)
    model = build_small_model()
    optimizer = build_optimizer_for(model, muon_lr=0.01, adam_lr=1e-3)
    graphs = sample(4, GEOMETRY)
    root_values = exhaustive_root_values(GEOMETRY.num_root_nodes)
    packed = evaluate_graphs(graphs, root_values)
    patch_indices = sample_patch_indices(MODEL_CONFIG.num_patches, 4, "cpu")
    targets = patch_targets(packed, patch_indices, model.decoder.rows_per_patch)

    def training_step() -> float:
        logits = model.forward_graphs(graphs, patch_indices)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    muon_probe = model.simulator.blocks[0].mlp[0].weight
    adam_probe = model.node_tokens.type_embeddings.weight
    muon_before = muon_probe.detach().clone()
    adam_before = adam_probe.detach().clone()

    first_loss = training_step()
    assert not torch.allclose(muon_probe, muon_before), "Muon params did not move"
    assert not torch.allclose(adam_probe, adam_before), "AdamW params did not move"
    assert torch.isfinite(muon_probe).all() and torch.isfinite(adam_probe).all()

    for _ in range(8):
        last_loss = training_step()
    assert last_loss < first_loss

    state = optimizer.state_dict()
    restored = build_optimizer_for(model, muon_lr=0.01, adam_lr=1e-3)
    restored.load_state_dict(state)
    assert torch.allclose(
        restored.muon.state[muon_probe]["momentum_buffer"],
        optimizer.muon.state[muon_probe]["momentum_buffer"],
    )
    assert torch.equal(
        restored.adamw.state[adam_probe]["exp_avg"],
        optimizer.adamw.state[adam_probe]["exp_avg"],
    )
    # Loading must not orphan the composite's view of the children's groups.
    assert restored.param_groups == restored.muon.param_groups + (
        restored.adamw.param_groups
    )
    torch.optim.lr_scheduler.LambdaLR(restored, lr_lambda=lambda step: 0.5)
    assert restored.muon.param_groups[0]["lr"] == 0.01 / 2


def test_zero_grad_clears_both_groups():
    model = build_small_model()
    optimizer = build_optimizer_for(model)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.zero_grad()
    assert all(param.grad is None for param in model.parameters())


def test_build_optimizer_dispatches_on_class():
    model = build_small_model()
    assert isinstance(build_optimizer(AutoMuon, model), AutoMuon)
    adam = build_optimizer(torch.optim.Adam, model, lr=1e-4)
    assert isinstance(adam, torch.optim.Adam)
    assert len(adam.param_groups[0]["params"]) == len(list(model.parameters()))


def test_rejects_a_bare_parameter_iterable():
    model = build_small_model()
    try:
        AutoMuon(model.parameters())
    except TypeError as error:
        assert "nn.Module" in str(error)
    else:
        raise AssertionError("expected a TypeError for a parameter iterable")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"{name}: ok")
    print("all optimizer tests passed")
