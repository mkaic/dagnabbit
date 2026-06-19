"""Deterministic snapshot harness for the DAG autoencoder.

It builds a deterministic model + graph, runs ``training_forward`` + a backward
pass, and normalizes everything into a plain dict of CPU float32 tensors with a
fixed schema. The output is useful as a local regression snapshot when changing
the autoencoder internals.

Key determinism choices:
  * ``TORCH_COMPILE_DISABLE=1`` (set before importing torch) -> pure eager on
    this harness, so ``@torch.compile`` is not a confound and no C toolchain is
    needed.
  * CPU, float32, no autocast.
  * One ``manual_seed`` up front; model init then graph generation consume the
    global RNG in a fixed order.

Usage:  python harness.py <output_pickle_path> [seed]
"""

import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import pickle
import sys

import torch

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.scripts import config as cfg


def _stack_live(x) -> torch.Tensor:
    """Return a *live* (autograd-connected) 1-D tensor from a loss field that is
    either a list[Tensor] or a 1-D Tensor."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.stack(list(x))


def _as_2d(x, num_nodes: int) -> torch.Tensor:
    """Normalize a per-node 2-D field that is either a Tensor [N, k] (current)
    or a list (of per-node Tensors, or of decode-buffer entries) into a detached
    float32 tensor of shape [num_nodes, k]."""
    if isinstance(x, torch.Tensor):
        return x.detach().float().clone()
    rows = []
    for e in x:
        # Pre-refactor decode buffer entries carry the combined embedding on
        # ``.combined_predicted_embedding``; logits lists carry raw tensors.
        row = getattr(e, "combined_predicted_embedding", e)
        rows.append(row)
    return torch.stack(rows).detach().float().reshape(num_nodes, -1).clone()


def run_and_normalize(seed: int) -> dict:
    torch.manual_seed(seed)

    device = torch.device("cpu")
    model = DagnabbitAutoEncoder(
        node_embedding_dim=cfg.NODE_EMBEDDING_DIM,
        trunk_node_type_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        mlp_depth=cfg.MLP_DEPTH,
        mlp_expansion_factor=cfg.MLP_EXPANSION_FACTOR,
        transformer_num_layers=cfg.TRANSFORMER_NUM_LAYERS,
        transformer_mlp_depth=cfg.TRANSFORMER_MLP_DEPTH,
        transformer_num_register_tokens=cfg.TRANSFORMER_NUM_REGISTER_TOKENS,
        transformer_num_heads=cfg.TRANSFORMER_NUM_HEADS,
        transformer_dropout=cfg.TRANSFORMER_DROPOUT,
    ).to(device)
    model = model.float()

    # Fingerprint the freshly-initialized parameters so a weight-init RNG drift
    # between commits would show up directly (not only indirectly via buffers).
    param_fingerprint = {
        name: float(p.detach().double().sum().item())
        for name, p in model.named_parameters()
    }

    graph = make_random_graph_description(
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_trunk_nodes=cfg.NUM_TRUNK_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
    )

    model.zero_grad(set_to_none=True)
    losses, primary_buffer, decode_buffer = model.training_forward(
        graph, return_buffers=True
    )

    num_nodes = graph.num_nodes

    # Live (non-detached) loss tensor -> total, replicating ``combine_losses``
    # (classification weight is 1.0 in config), so we can backprop through it.
    live_pc = _stack_live(losses.primary_node_classification_losses)
    total = cfg.W_PRIMARY_DECODED_CLASSIFICATION * live_pc.mean()

    total.backward()

    # Detached, normalized copy of the per-node loss vector for comparison.
    pc = live_pc.detach().float().reshape(-1).clone()

    grads = {
        name: p.grad.detach().float().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }

    primary_logits = _as_2d(losses.primary_node_predicted_type_logits, num_nodes)

    return {
        "num_nodes": num_nodes,
        "node_types": list(graph.node_types),
        "param_fingerprint": param_fingerprint,
        "encode_buffer": primary_buffer.detach().float().clone(),
        "decode_combined": _as_2d(decode_buffer, num_nodes),
        "primary_logits": primary_logits,
        "loss_primary_classification": pc,
        "total_loss": float(total.detach().item()),
        "grads": grads,
    }


def main() -> None:
    out_path = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1234

    result = run_and_normalize(seed)
    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    print(f"wrote {out_path}")
    print(f"  seed={seed} num_nodes={result['num_nodes']}")
    print(f"  total_loss={result['total_loss']:.8f}")
    print(f"  encode_buffer={tuple(result['encode_buffer'].shape)}")
    print(f"  decode_combined={tuple(result['decode_combined'].shape)}")
    print(f"  num_grad_tensors={len(result['grads'])}")


if __name__ == "__main__":
    main()
