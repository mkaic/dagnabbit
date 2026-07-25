"""Routing, wiring, and end-to-end checks for :class:`AutoMuon`.

The update rules themselves are ``torch.optim.Muon`` and ``torch.optim.AdamW``
and are not retested here; what is tested is that the right parameters reach
the right one and that the composite behaves like a single optimizer.

Run directly::

    python -m dagnabbit.tests.test_optimizers
"""

import torch
import torch.nn as nn

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.optimizers import AutoMuon, build_optimizer

EMBEDDING_DIM = 64
NUM_ROOTS = 8
NUM_TRUNKS = 24
NUM_OUTPUTS = 4
NUM_TRUNK_TYPES = 2
IN_DEGREES = 2


def build_small_model() -> DagnabbitAutoEncoder:
    return DagnabbitAutoEncoder(
        node_embedding_dim=EMBEDDING_DIM,
        trunk_node_type_in_degrees=IN_DEGREES,
        num_trunk_node_types=NUM_TRUNK_TYPES,
        num_root_nodes=NUM_ROOTS,
        num_trunk_nodes=NUM_TRUNKS,
        num_output_nodes=NUM_OUTPUTS,
        mlp_expansion_factor=2.0,
        encoder_num_layers=1,
        compressor_num_layers=2,
        decoder_num_layers=2,
    )


def build_optimizer_for(model: nn.Module, **kwargs) -> AutoMuon:
    kwargs.setdefault("adam_module_names", ("node_type_predictor",))
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
    assert "compressor.blocks.0.attn.in_proj_weight" in muon_names
    assert "compressor.blocks.0.attn.out_proj.weight" in muon_names
    assert "compressor.blocks.0.ff.0.weight" in muon_names
    assert "decoder.blocks.1.attn.in_proj_weight" in muon_names
    assert "node_encoder.sequence_transformer.blocks.0.ff.0.weight" in muon_names
    # Square hidden-space pointer projections are ordinary weight matrices.
    assert "pointer_key_proj.weight" in muon_names
    assert "pointer_slot_query_projs.0.weight" in muon_names

    # Lookup tables, 2-D learned token banks, norms, biases and the named
    # output head all belong to AdamW. The token banks are the cases a plain
    # ``p.ndim == 2`` split would get wrong.
    assert "root_node_embeddings.weight" in adam_names
    assert "node_encoder.sequence_transformer.node_type_embeddings.weight" in adam_names
    assert "node_encoder.sequence_transformer.position_embeddings" in adam_names
    assert "node_encoder.sequence_transformer.register_tokens" in adam_names
    assert "mask_token" in adam_names
    assert "node_type_predictor.weight" in adam_names
    assert "node_type_predictor.bias" in adam_names
    assert "compressor.blocks.0.attn_norm.weight" in adam_names
    assert "compressor.blocks.0.attn.in_proj_bias" in adam_names
    assert "compressor.blocks.0.ff.0.bias" in adam_names


def test_every_muon_parameter_is_a_2d_weight():
    """torch.optim.Muon rejects non-2-D parameters outright."""
    model = build_small_model()
    optimizer = build_optimizer_for(model)
    for group in optimizer.muon.param_groups:
        for param in group["params"]:
            assert param.ndim == 2, param.shape


def test_adam_module_names_match_paths_prefixes_and_patterns():
    model = build_small_model()
    exact = routed_names(AutoMuon(model, adam_module_names=("pointer_key_proj.weight",)))
    assert "pointer_key_proj.weight" in exact[1]
    assert "pointer_slot_query_projs.0.weight" in exact[0]

    subtree = routed_names(AutoMuon(model, adam_module_names=("pointer_slot_query_projs",)))
    assert "pointer_slot_query_projs.0.weight" in subtree[1]
    assert "pointer_key_proj.weight" in subtree[0]

    pattern = routed_names(AutoMuon(model, adam_module_names=("*.in_proj_weight",)))
    assert "compressor.blocks.0.attn.in_proj_weight" in pattern[1]
    assert "compressor.blocks.0.attn.out_proj.weight" in pattern[0]


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
    graphs = [
        make_random_graph_description(
            num_root_nodes=NUM_ROOTS,
            num_trunk_nodes=NUM_TRUNKS,
            num_output_nodes=NUM_OUTPUTS,
            num_trunk_node_types=NUM_TRUNK_TYPES,
            trunk_node_in_degrees=IN_DEGREES,
        )
        for _ in range(2)
    ]

    def training_step() -> float:
        losses = model.training_forward_batch(graphs)
        loss = (
            losses.node_classification_losses.mean()
            + losses.parent_pointer_losses.sum()
            / losses.parent_pointer_slot_mask.sum().clamp(min=1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    muon_probe = model.compressor.blocks[0].ff[0].weight
    adam_probe = model.root_node_embeddings.weight
    muon_before = muon_probe.detach().clone()
    adam_before = adam_probe.detach().clone()

    first_loss = training_step()
    assert not torch.allclose(muon_probe, muon_before), "Muon params did not move"
    assert not torch.allclose(adam_probe, adam_before), "AdamW params did not move"
    assert torch.isfinite(muon_probe).all() and torch.isfinite(adam_probe).all()

    for _ in range(4):
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
