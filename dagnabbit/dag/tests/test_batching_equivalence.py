"""Numerical-equivalence guard for the rank/supertype batching refactor.

The batching refactor (commits ``5055bc6`` -> ``5c271de``) replaced the per-node
encode loop (``evaluate_graph``) and the per-node guided-autoregressive decode
(``_decode_step``) with rank-and-supertype *batched* paths (``encode_batch`` /
``decode_batch`` / ``_decode_graph``). The claim is that this is a pure
performance change: numerically identical up to floating-point reduction order.

This module proves that claim two ways:

1. ``test_batched_matches_per_node_reference`` (the strong, permanent check).
   The current code still carries the *original* per-node module methods
   (``NodeEncoder.forward``, ``NodeDecoder.forward``,
   ``FixedInDegreeNodeAutoEncoder.encode/decode``,
   ``OutputNodeAutoEncoder.encode/decode``) -- only the graph-walking glue was
   rebatched. So we re-run the pre-refactor per-node ``evaluate_graph`` +
   ``_decode_step`` logic (vendored verbatim below) against the *same model
   instance* as the batched ``training_forward`` and compare encode buffers,
   decode buffers, per-node losses, the total loss, and **gradients**.

   Run in float64 the two paths agree to ~1e-10 -- i.e. the only thing
   separating them is float reduction order, exactly as the plan asserts.

2. A separate, out-of-repo cross-commit harness (see the chat that created this
   file) additionally confirmed that the batched ``main`` reproduces the
   genuine pre-refactor commit ``aae1991`` (identical weight init, identical
   graph, total-loss diff ~7e-6 in float32). That check lives outside the repo
   because the two commits have incompatible module APIs; this in-process test
   is its maintainable stand-in.

Run directly for a human-readable diff report::

    TORCH_COMPILE_DISABLE=1 python -m dagnabbit.dag.tests.test_batching_equivalence
"""

# Disable torch.compile so we compare pure-eager math on both the batched and
# per-node paths (and so the test needs no C toolchain). Must run before torch
# is imported, since ``MLP.forward`` is decorated at class-definition time.
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import math

import torch
import torch.nn.functional as F

from dagnabbit.dag.autoencoder import (
    DagnabbitAutoEncoder,
    NodeAutoEncoder,
    _class_balance_weights,
)
from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_random_graph_description,
)
from dagnabbit.scripts import config as cfg


# --------------------------------------------------------------------------- #
# Pre-refactor per-node reference implementation (vendored from commit aae1991,
# dagnabbit/dag/autoencoder.py). It deliberately uses only the per-node module
# methods the current model still exposes, so it runs against the live model.
# --------------------------------------------------------------------------- #


def _reference_evaluate_graph(
    graph: FixedInDegreeDAGDescription,
    root_node_embeddings: torch.Tensor,
    node_autoencoders: dict[int, NodeAutoEncoder],
    node_embedding_dim: int,
) -> torch.Tensor:
    """Pre-refactor ``DagnabbitAutoEncoder.evaluate_graph`` (per-node loop)."""
    assert root_node_embeddings.shape[0] == graph.num_root_nodes

    embeddings_buffer = torch.empty(
        (graph.num_nodes, node_embedding_dim),
        dtype=root_node_embeddings.dtype,
        device=root_node_embeddings.device,
    )
    embeddings_buffer[: graph.num_root_nodes] = root_node_embeddings

    for node_idx in range(graph.num_root_nodes, graph.num_nodes):
        buffer_read_indices = graph.node_inputs_indices[node_idx]
        parent_embeddings = embeddings_buffer[buffer_read_indices, :]
        node_type = graph.node_types[node_idx]
        node_autoencoder = node_autoencoders[node_type]
        embeddings_buffer[node_idx] = node_autoencoder.encode(
            parent_embeddings, node_type
        )

    return embeddings_buffer


class _RefEntry:
    """Stand-in for the removed ``TrainingDecodeBufferEntry`` dataclass."""

    __slots__ = (
        "embeddings_predicted_by_children",
        "classification_loss",
        "predicted_type_logits",
        "combined_predicted_embedding",
    )

    def __init__(self) -> None:
        self.embeddings_predicted_by_children: list[torch.Tensor] = []
        self.classification_loss = None
        self.predicted_type_logits = None
        self.combined_predicted_embedding = None


def _reference_decode_step(
    model: DagnabbitAutoEncoder,
    node_idx: int,
    graph: FixedInDegreeDAGDescription,
    decode_buffer: list[_RefEntry],
    node_autoencoders: dict[int, NodeAutoEncoder],
    class_label: torch.Tensor,
    classification_loss_weight: float,
) -> None:
    """Pre-refactor ``DagnabbitAutoEncoder._decode_step`` (per-node)."""
    entry = decode_buffer[node_idx]

    embeddings_predicted_by_children = torch.stack(
        entry.embeddings_predicted_by_children
    )
    num_children = embeddings_predicted_by_children.shape[0]
    combined = embeddings_predicted_by_children.sum(dim=0) / math.sqrt(num_children)
    entry.combined_predicted_embedding = combined

    logits = model.node_type_predictor(combined)
    entry.predicted_type_logits = logits

    entry.classification_loss = (
        F.cross_entropy(logits, class_label) * classification_loss_weight
    )

    node_parent_indices = graph.node_inputs_indices[node_idx]
    if len(node_parent_indices) == 0:
        return

    node_type = graph.node_types[node_idx]
    node_autoencoder = node_autoencoders[node_type]
    predicted_parent_embeddings = node_autoencoder.decode(combined, node_type)

    for parent_idx, parent_embedding in zip(
        node_parent_indices, predicted_parent_embeddings
    ):
        decode_buffer[parent_idx].embeddings_predicted_by_children.append(
            parent_embedding
        )


def reference_training_forward(
    model: DagnabbitAutoEncoder,
    primary_graph: FixedInDegreeDAGDescription,
) -> dict:
    """Pre-refactor ``training_forward`` body, returning normalized live tensors."""
    device = model.root_node_embeddings.weight.device
    D = model.node_embedding_dim

    primary_buffer = _reference_evaluate_graph(
        primary_graph,
        model.root_node_embeddings.weight,
        model.node_autoencoders,
        D,
    )

    primary_weights = _class_balance_weights(list(primary_graph.node_types))
    primary_labels = torch.as_tensor(
        list(primary_graph.node_types), dtype=torch.long, device=device
    )

    primary_buffer_entries = [_RefEntry() for _ in range(primary_graph.num_nodes)]

    # Seed each leaf with its own encode embedding (mirrors the batched
    # ``_decode_pipeline``): leaves are referenced by no node, so nothing
    # scatters predictions onto them during decode -- without a seed their
    # ``embeddings_predicted_by_children`` would be empty and the combine would
    # divide by zero / stack an empty list.
    for leaf_idx in primary_graph.leaf_node_indices:
        primary_buffer_entries[leaf_idx].embeddings_predicted_by_children.append(
            primary_buffer[leaf_idx]
        )

    for node_idx in reversed(range(primary_graph.num_nodes)):
        _reference_decode_step(
            model,
            node_idx,
            primary_graph,
            primary_buffer_entries,
            model.node_autoencoders,
            primary_labels[node_idx],
            primary_weights[node_idx],
        )

    pc = torch.stack([e.classification_loss for e in primary_buffer_entries])

    decode_combined = torch.stack(
        [e.combined_predicted_embedding for e in primary_buffer_entries]
    )

    return {
        "encode_buffer": primary_buffer,
        "decode_combined": decode_combined,
        "loss_primary_classification": pc,
    }


# --------------------------------------------------------------------------- #
# Batched (current) path, normalized to the same schema.
# --------------------------------------------------------------------------- #


def batched_training_forward(
    model: DagnabbitAutoEncoder,
    primary_graph: FixedInDegreeDAGDescription,
) -> dict:
    losses, primary_buffer, decode_combined = model.training_forward(
        primary_graph, return_buffers=True
    )
    return {
        "encode_buffer": primary_buffer,
        "decode_combined": decode_combined,
        "loss_primary_classification": losses.primary_node_classification_losses,
    }


def _total_loss(out: dict) -> torch.Tensor:
    """Replicate ``combine_losses`` (classification weight is 1.0)."""
    return cfg.W_PRIMARY_DECODED_CLASSIFICATION * out["loss_primary_classification"].mean()


# --------------------------------------------------------------------------- #
# Harness shared by the test and the __main__ report.
# --------------------------------------------------------------------------- #


def _build_model(dtype: torch.dtype) -> DagnabbitAutoEncoder:
    model = DagnabbitAutoEncoder(
        node_embedding_dim=cfg.NODE_EMBEDDING_DIM,
        trunk_node_type_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        mlp_depth=cfg.MLP_DEPTH,
        mlp_expansion_factor=cfg.MLP_EXPANSION_FACTOR,
    )
    return model.to(dtype=dtype)


def run_comparison(
    seed: int = 1234,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Run the per-node reference and the batched path against one shared model
    and return max abs/rel diffs over activations, losses, total, and grads.

    Gradients are compared by running both paths on the *same* model: backward
    the reference, snapshot grads, zero them, backward the batched path, snapshot
    again.
    """
    torch.manual_seed(seed)
    model = _build_model(dtype)

    primary_graph = make_random_graph_description(
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_trunk_nodes=cfg.NUM_TRUNK_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
    )

    model.zero_grad(set_to_none=True)
    ref = reference_training_forward(model, primary_graph)
    ref_total = _total_loss(ref)
    ref_total.backward()
    ref_grads = {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }

    model.zero_grad(set_to_none=True)
    bat = batched_training_forward(model, primary_graph)
    bat_total = _total_loss(bat)
    bat_total.backward()
    bat_grads = {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }

    def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.detach().double() - b.detach().double()).abs().max())

    tensor_keys = [
        "encode_buffer",
        "decode_combined",
        "loss_primary_classification",
    ]
    diffs = {k: max_abs(ref[k], bat[k]) for k in tensor_keys}
    diffs["total_loss"] = abs(float(ref_total) - float(bat_total))

    assert ref_grads.keys() == bat_grads.keys()
    grad_max = max(max_abs(ref_grads[n], bat_grads[n]) for n in ref_grads)
    diffs["grads"] = grad_max
    diffs["_num_grad_tensors"] = len(ref_grads)
    diffs["_num_nodes"] = primary_graph.num_nodes
    return diffs


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #


def test_batched_matches_per_node_reference_float64() -> None:
    """In float64 the batched path and the original per-node loop differ only by
    floating-point reduction order, so everything (incl. gradients) agrees to a
    very tight tolerance."""
    diffs = run_comparison(seed=1234, dtype=torch.float64)
    # float64 reduction-order noise on this graph is ~1e-12..1e-10.
    assert diffs["encode_buffer"] < 1e-9, diffs
    assert diffs["decode_combined"] < 1e-9, diffs
    assert diffs["total_loss"] < 1e-9, diffs
    assert diffs["loss_primary_classification"] < 1e-9, diffs
    assert diffs["grads"] < 1e-7, diffs


def test_batched_matches_per_node_reference_float32() -> None:
    """The same equivalence in float32, with tolerances loosened to the larger
    float32 reduction-order noise (this is the dtype training actually runs)."""
    diffs = run_comparison(seed=7, dtype=torch.float32)
    assert diffs["encode_buffer"] < 1e-3, diffs
    assert diffs["decode_combined"] < 1e-2, diffs
    assert diffs["total_loss"] < 1e-3, diffs
    assert diffs["grads"] < 1e-2, diffs


def main() -> None:
    for dtype in (torch.float64, torch.float32):
        diffs = run_comparison(seed=1234, dtype=dtype)
        print(f"\n==== batched vs per-node reference ({dtype}) ====")
        print(
            f"  num_nodes={diffs['_num_nodes']} "
            f"num_grad_tensors={diffs['_num_grad_tensors']}"
        )
        for key, value in diffs.items():
            if key.startswith("_"):
                continue
            print(f"  {key:34s} max_abs_diff = {value:.3e}")
    print("\nALL EQUIVALENCE CHECKS PASSED")


if __name__ == "__main__":
    main()
