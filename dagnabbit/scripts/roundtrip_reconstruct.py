"""Roundtrip evaluation of the sequence pipeline: encode -> latent -> rebuild.

For a batch of freshly generated random graphs this script runs the full
training-time reconstruction (teacher-forced slot counts) to report node-type
and parent-pointer accuracy, then runs the genuine generation path
(``encode_to_latent`` -> ``generate``, predicted types deciding slot counts)
and reports the exact-graph match rate via ``graphs_match`` (ordered output
canonical ids; dead nodes ignored).

Usage::

    python -m dagnabbit.scripts.roundtrip_reconstruct --checkpoint runs/<run>
    python -m dagnabbit.scripts.roundtrip_reconstruct  # fresh (untrained) model
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import graphs_match, make_random_graph_description
from dagnabbit.scripts import config as cfg
from dagnabbit.scripts.logging_utils import (
    accuracy_summary,
    pointer_accuracy_summary,
    step_pointer_stats,
    step_preds_and_truth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint to evaluate: a .ckpt file or a run directory (its "
            "best.ckpt is preferred, falling back to latest.ckpt). Omit to "
            "evaluate a freshly initialized model."
        ),
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=64,
        help="Total random graphs to roundtrip.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Graphs per forward pass (default: cfg.GRAPH_BATCH_SIZE).",
    )
    return parser.parse_args()


def resolve_checkpoint(argument: str) -> Path:
    path = Path(argument)
    if path.is_dir():
        best = path / "best.ckpt"
        resolved = best if best.exists() else path / "latest.ckpt"
    else:
        resolved = path
    if not resolved.exists():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    return resolved


def build_model(device: torch.device) -> DagnabbitAutoEncoder:
    return DagnabbitAutoEncoder(
        node_embedding_dim=cfg.NODE_EMBEDDING_DIM,
        trunk_node_type_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_trunk_nodes=cfg.NUM_TRUNK_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        mlp_expansion_factor=cfg.MLP_EXPANSION_FACTOR,
        encoder_num_layers=cfg.ENCODER_NUM_LAYERS,
        compressor_num_layers=cfg.COMPRESSOR_NUM_LAYERS,
        decoder_num_layers=cfg.DECODER_NUM_LAYERS,
    ).to(device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    batch_size = args.batch_size or cfg.GRAPH_BATCH_SIZE

    model = build_model(device)
    if args.checkpoint is not None:
        checkpoint_path = resolve_checkpoint(args.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded {checkpoint_path} (graphs_seen={checkpoint.get('graphs_seen')})")
    else:
        print("no checkpoint given: evaluating a freshly initialized model")
    model.eval()

    type_preds: list[np.ndarray] = []
    type_truth: list[np.ndarray] = []
    pointer_correct: list[np.ndarray] = []
    pointer_is_output: list[np.ndarray] = []
    pointer_is_root_parent: list[np.ndarray] = []
    matches = 0
    total = 0

    with torch.no_grad():
        while total < args.num_graphs:
            count = min(batch_size, args.num_graphs - total)
            graphs = [
                make_random_graph_description(
                    num_root_nodes=cfg.NUM_ROOT_NODES,
                    num_trunk_nodes=cfg.NUM_TRUNK_NODES,
                    num_output_nodes=cfg.NUM_OUTPUT_NODES,
                    trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
                    num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
                )
                for _ in range(count)
            ]

            # Teacher-forced reconstruction metrics (ground-truth slot counts).
            losses = model.training_forward_batch(graphs)
            preds, truth = step_preds_and_truth(
                losses.node_predicted_type_logits,
                losses.node_true_types,
            )
            type_preds.append(preds)
            type_truth.append(truth)
            correct, is_output, is_root_parent = step_pointer_stats(
                losses.parent_pointer_logits,
                losses.parent_pointer_true_positions,
                losses.parent_pointer_slot_mask,
                output_start=model.output_start,
                num_root_nodes=model.num_root_nodes,
            )
            pointer_correct.append(correct)
            pointer_is_output.append(is_output)
            pointer_is_root_parent.append(is_root_parent)

            # Genuine generation path: predicted types decide slot counts.
            rebuilt = model.generate(model.encode_to_latent(graphs))
            matches += sum(
                graphs_match(graph, candidate)
                for graph, candidate in zip(graphs, rebuilt)
            )
            total += count

    type_accuracy, type_by_supertype = accuracy_summary(
        np.concatenate(type_preds),
        np.concatenate(type_truth),
        num_classes=model.num_trunk_node_types,
    )
    pointer_accuracy, pointer_by_supertype, root_parent_accuracy = (
        pointer_accuracy_summary(
            np.concatenate(pointer_correct),
            np.concatenate(pointer_is_output),
            np.concatenate(pointer_is_root_parent),
        )
    )

    print(f"graphs evaluated        {total}")
    print(f"type accuracy (mean of per-class recalls)   {type_accuracy:.4f}")
    for supertype, accuracy in type_by_supertype.items():
        print(f"  type accuracy [{supertype.value:6s}]  {accuracy:.4f}")
    print(f"pointer accuracy (valid slots)              {pointer_accuracy:.4f}")
    for supertype, accuracy in pointer_by_supertype.items():
        print(f"  pointer accuracy [{supertype.value:6s}]  {accuracy:.4f}")
    print(f"root-parent pointer accuracy                {root_parent_accuracy:.4f}")
    print(f"exact graph match rate                      {matches / total:.4f}"
          f"  ({matches}/{total})")


if __name__ == "__main__":
    main()
