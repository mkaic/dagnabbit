"""Numerical-equivalence guard for the decode child-buffer scatter.

Easy-win #1 replaces the per-graph Python loop of ``index_add_`` calls in
``DagnabbitAutoEncoder._decode_graph`` with a single batched scatter over a
flattened ``[B*N, ...]`` view. Each graph maps to a disjoint index range, so the
per-slot accumulation order is unchanged and the result should be *bitwise*
identical on CPU (deterministic ``index_add_``).

This script captures a golden snapshot of every tensor returned by
``training_forward_batch`` (plus the encode/decode buffers) for a fixed seed,
then compares a later run against it::

    # 1. on the OLD implementation, write the golden file
    uv run python dagnabbit/dag/tests/test_decode_scatter_equivalence.py --save-golden /tmp/decode_golden.pt

    # 2. after editing autoencoder.py, compare against it
    uv run python dagnabbit/dag/tests/test_decode_scatter_equivalence.py --golden /tmp/decode_golden.pt

The committed default (no flags) regenerates a golden on the current code and
immediately re-runs against it, which guards determinism but NOT old-vs-new; the
old-vs-new check is the two-step flow above.
"""

import argparse

import torch

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.scripts import config as cfg

SEED = 1234
BATCH_SIZE = 8


def _build() -> tuple[DagnabbitAutoEncoder, list]:
    """Deterministically build the model and a batch of graphs.

    Both model init and graph generation draw from the global torch RNG, so a
    single ``manual_seed`` pins the entire setup.
    """
    torch.manual_seed(SEED)
    model = DagnabbitAutoEncoder(
        node_embedding_dim=cfg.NODE_EMBEDDING_DIM,
        trunk_node_type_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        mlp_expansion_factor=cfg.MLP_EXPANSION_FACTOR,
        reconstruction_detach_target=cfg.RECONSTRUCTION_DETACH_TARGET,
        # Force the reconstruction path on so the equivalence check also covers
        # the edge-flat recon losses, not just classification.
        compute_reconstruction_loss=True,
        transformer_num_layers=cfg.TRANSFORMER_NUM_LAYERS,
        transformer_mlp_depth=cfg.TRANSFORMER_MLP_DEPTH,
        transformer_num_register_tokens=cfg.TRANSFORMER_NUM_REGISTER_TOKENS,
        transformer_num_heads=cfg.TRANSFORMER_NUM_HEADS,
        transformer_dropout=0.0,
        transformer_residual_step_init=cfg.TRANSFORMER_RESIDUAL_STEP_INIT,
    )
    model.eval()
    graphs = [
        make_random_graph_description(
            num_root_nodes=cfg.NUM_ROOT_NODES,
            num_trunk_nodes=cfg.NUM_TRUNK_NODES,
            num_output_nodes=cfg.NUM_OUTPUT_NODES,
            trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
            num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        )
        for _ in range(BATCH_SIZE)
    ]
    return model, graphs


def _capture() -> dict[str, torch.Tensor]:
    model, graphs = _build()
    with torch.no_grad():
        losses, encode_buf, decode_buf = model.training_forward_batch(
            graphs, return_buffers=True
        )
    out = {
        "primary_class": losses.primary_node_classification_losses,
        "primary_logits": losses.primary_node_predicted_type_logits,
        "primary_true_types": losses.primary_node_true_types,
        "tf_class": losses.teacher_forced_primary_node_classification_losses,
        "tf_logits": losses.teacher_forced_primary_node_predicted_type_logits,
        "primary_recon": losses.primary_node_parent_reconstruction_losses,
        "tf_recon": losses.teacher_forced_primary_node_parent_reconstruction_losses,
        "primary_consistency": losses.primary_node_parent_consistency_losses,
        "tf_consistency": losses.teacher_forced_primary_node_parent_consistency_losses,
        "encode_buffer": encode_buf,
        "decode_buffer": decode_buf,
    }
    return {k: v.detach().clone() for k, v in out.items()}


def _compare(ref: dict[str, torch.Tensor], cur: dict[str, torch.Tensor]) -> bool:
    ok = True
    width = max(len(k) for k in ref)
    print(f"{'tensor':<{width}}  {'shape':<18}  {'exact':<6}  max_abs_diff")
    for key, ref_t in ref.items():
        cur_t = cur[key]
        if cur_t.shape != ref_t.shape:
            print(f"{key:<{width}}  SHAPE MISMATCH {tuple(ref_t.shape)} vs "
                  f"{tuple(cur_t.shape)}")
            ok = False
            continue
        exact = bool(torch.equal(ref_t, cur_t))
        if ref_t.is_floating_point():
            max_diff = (
                (ref_t - cur_t).abs().max().item() if ref_t.numel() else 0.0
            )
        else:
            max_diff = 0.0 if exact else float("nan")
        close = exact or (
            ref_t.is_floating_point()
            and torch.allclose(ref_t, cur_t, rtol=0.0, atol=1e-6)
        )
        flag = "yes" if exact else ("~" if close else "NO")
        print(f"{key:<{width}}  {str(tuple(ref_t.shape)):<18}  {flag:<6}  {max_diff:.3e}")
        if not close:
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-golden", metavar="PATH",
                        help="Run current code and write a golden snapshot here.")
    parser.add_argument("--golden", metavar="PATH",
                        help="Compare current code against this golden snapshot.")
    args = parser.parse_args()

    if args.save_golden:
        torch.save(_capture(), args.save_golden)
        print(f"wrote golden snapshot -> {args.save_golden}")
        return

    cur = _capture()
    if args.golden:
        ref = torch.load(args.golden)
        print(f"comparing current code against golden {args.golden}\n")
    else:
        # Self-consistency: prove determinism on the current code only.
        ref = _capture()
        print("no --golden given; checking run-to-run determinism only\n")

    if _compare(ref, cur):
        print("\nEQUIVALENCE OK")
    else:
        raise SystemExit("\nEQUIVALENCE FAILED")


if __name__ == "__main__":
    main()
