"""Benchmark child-predicted parent embedding separation.

Run:

    uv run python -m dagnabbit.scripts.benchmark_parent_prediction_similarity \
        --ckpt "runs/20260623-190239-posenc-spread-0.08->0.02_0.1tfreconloss/checkpoints/graphs-003840000.ckpt" \
        --num-graphs 128
"""

from __future__ import annotations

# Keep this diagnostic pure-eager and comparable to the round-trip harness.
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import FixedInDegreeDAGDescription
from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.scripts import config as cfg
from dagnabbit.scripts.roundtrip_blind_decode import build_model, load_checkpoint


class SimilarityAccumulator:
    def __init__(self, name: str) -> None:
        self.name = name
        self._chunks: list[Tensor] = []

    def add(self, values: Tensor) -> None:
        if values.numel():
            self._chunks.append(values.detach().float().cpu())

    def tensor(self) -> Tensor:
        if not self._chunks:
            return torch.empty(0)
        return torch.cat(self._chunks)


def _make_graphs(num_graphs: int) -> list[FixedInDegreeDAGDescription]:
    return [
        make_random_graph_description(
            num_root_nodes=cfg.NUM_ROOT_NODES,
            num_trunk_nodes=cfg.NUM_TRUNK_NODES,
            num_output_nodes=cfg.NUM_OUTPUT_NODES,
            trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
            num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        )
        for _ in range(num_graphs)
    ]


def _sample_mask(
    mask: Tensor,
    *,
    max_pairs: int | None,
    generator: torch.Generator,
) -> Tensor:
    pair_count = int(mask.sum().item())
    if max_pairs is None or pair_count <= max_pairs:
        return mask

    flat_indices = mask.flatten().nonzero(as_tuple=False).squeeze(1)
    chosen = flat_indices[
        torch.randperm(flat_indices.numel(), generator=generator)[:max_pairs]
    ]
    sampled = torch.zeros_like(mask.flatten())
    sampled[chosen] = True
    return sampled.reshape_as(mask)


def _add_pair_stats(
    *,
    embeddings: Tensor,
    true_parent_embeddings: Tensor,
    parent_ids: Tensor,
    child_ids: Tensor,
    same_parent: SimilarityAccumulator,
    different_parent: SimilarityAccumulator,
    different_parent_true: SimilarityAccumulator,
    different_nonroot_parent: SimilarityAccumulator,
    root_count: int,
    max_different_pairs_per_graph: int | None,
    generator: torch.Generator,
) -> None:
    if embeddings.shape[0] < 2:
        return

    embeddings = F.normalize(embeddings.float(), dim=-1).cpu()
    true_parent_embeddings = F.normalize(true_parent_embeddings.float(), dim=-1).cpu()
    parent_ids = parent_ids.cpu()
    child_ids = child_ids.cpu()

    sims = embeddings @ embeddings.t()
    true_sims = true_parent_embeddings @ true_parent_embeddings.t()

    upper = torch.triu(
        torch.ones(
            embeddings.shape[0],
            embeddings.shape[0],
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    distinct_children = child_ids[:, None] != child_ids[None, :]
    same_mask = upper & distinct_children & (parent_ids[:, None] == parent_ids[None, :])
    diff_mask = upper & distinct_children & (parent_ids[:, None] != parent_ids[None, :])
    nonroot_mask = diff_mask & (parent_ids[:, None] >= root_count)
    nonroot_mask = nonroot_mask & (parent_ids[None, :] >= root_count)

    diff_mask = _sample_mask(
        diff_mask,
        max_pairs=max_different_pairs_per_graph,
        generator=generator,
    )
    nonroot_mask = _sample_mask(
        nonroot_mask,
        max_pairs=max_different_pairs_per_graph,
        generator=generator,
    )

    same_parent.add(sims[same_mask])
    different_parent.add(sims[diff_mask])
    different_parent_true.add(true_sims[diff_mask])
    different_nonroot_parent.add(sims[nonroot_mask])


@torch.no_grad()
def collect_parent_predictions(
    model: DagnabbitAutoEncoder,
    graphs: Sequence[FixedInDegreeDAGDescription],
    *,
    include_teacher_forced: bool,
) -> tuple[
    dict[int, dict[str, Tensor]],
    dict[int, dict[str, Tensor]] | None,
]:
    device = model.root_node_embeddings.weight.device
    rank_batches = model._make_batched_rank_cache(graphs, device)
    encoder_buffer = model.evaluate_graph_batch(graphs, rank_batches=rank_batches)
    batch_size, num_nodes, _ = encoder_buffer.shape

    def _zeros() -> tuple[Tensor, Tensor, Tensor]:
        return (
            torch.zeros_like(encoder_buffer),
            torch.zeros(batch_size, num_nodes, dtype=encoder_buffer.dtype, device=device),
            torch.zeros_like(encoder_buffer),
        )

    ar_sum, ar_count, ar_sumsq = _zeros()
    tf_sum, tf_count, tf_sumsq = _zeros()

    leaf_indices = torch.stack(
        [graph.leaf_node_indices_tensor for graph in graphs],
        dim=0,
    ).to(device=device, non_blocking=True)
    batch_rows = torch.arange(batch_size, dtype=torch.long, device=device)[:, None]
    leaf_embeddings_by_graph = encoder_buffer[batch_rows, leaf_indices]
    for child_sum, child_count, child_sumsq in (
        (ar_sum, ar_count, ar_sumsq),
        (tf_sum, tf_count, tf_sumsq),
    ):
        child_sum[batch_rows, leaf_indices] = leaf_embeddings_by_graph
        child_count[batch_rows, leaf_indices] = 1.0
        child_sumsq[batch_rows, leaf_indices] = leaf_embeddings_by_graph**2

    ar_combined = torch.zeros_like(encoder_buffer)
    ar_by_graph: dict[int, dict[str, list[Tensor]]] = {
        i: {
            "pred": [],
            "true_parent": [],
            "parent": [],
            "child": [],
        }
        for i in range(batch_size)
    }
    tf_by_graph: dict[int, dict[str, list[Tensor]]] | None = None
    if include_teacher_forced:
        tf_by_graph = {
            i: {
                "pred": [],
                "true_parent": [],
                "parent": [],
                "child": [],
            }
            for i in range(batch_size)
        }

    for rank in reversed(range(len(rank_batches))):
        rank_batch = rank_batches[rank]
        if rank_batch.node_indices.numel() == 0:
            continue

        rows = rank_batch.batch_indices
        nodes = rank_batch.node_indices

        ar_counts = ar_count[rows, nodes]
        ar_combined_rank = ar_sum[rows, nodes] / ar_counts.sqrt().unsqueeze(-1)
        ar_combined[rows, nodes] = ar_combined_rank

        if not rank_batch.has_valid_parents:
            continue

        if include_teacher_forced:
            decode_input = torch.cat(
                [ar_combined[rows, nodes], encoder_buffer[rows, nodes]],
                dim=0,
            )
            subtypes = torch.cat([rank_batch.subtypes, rank_batch.subtypes])
            leaf_embeddings = leaf_embeddings_by_graph[rows]
            leaf_embeddings = torch.cat([leaf_embeddings, leaf_embeddings], dim=0)
            predicted = model.node_decoder.forward_batch(
                decode_input,
                subtypes,
                leaf_embeddings,
            )
            ar_predicted = predicted[: nodes.shape[0]]
            tf_predicted = predicted[nodes.shape[0] :]
        else:
            ar_predicted = model.node_decoder.forward_batch(
                ar_combined[rows, nodes],
                rank_batch.subtypes,
                leaf_embeddings_by_graph[rows],
            )
            tf_predicted = None

        mask = rank_batch.valid_parent_mask
        flat_parent_indices = rank_batch.parent_indices[mask]
        flat_batch_indices = rows[:, None].expand_as(rank_batch.parent_indices)[mask]
        flat_child_indices = nodes[:, None].expand_as(rank_batch.parent_indices)[mask]
        ar_flat = ar_predicted[mask]
        true_flat = encoder_buffer[flat_batch_indices, flat_parent_indices]
        if tf_predicted is not None:
            tf_flat = tf_predicted[mask]

        for graph_idx in flat_batch_indices.unique().tolist():
            graph_mask = flat_batch_indices == graph_idx
            ar_by_graph[graph_idx]["pred"].append(ar_flat[graph_mask])
            ar_by_graph[graph_idx]["true_parent"].append(true_flat[graph_mask])
            ar_by_graph[graph_idx]["parent"].append(flat_parent_indices[graph_mask])
            ar_by_graph[graph_idx]["child"].append(flat_child_indices[graph_mask])
            if tf_by_graph is not None and tf_predicted is not None:
                tf_by_graph[graph_idx]["pred"].append(tf_flat[graph_mask])
                tf_by_graph[graph_idx]["true_parent"].append(true_flat[graph_mask])
                tf_by_graph[graph_idx]["parent"].append(flat_parent_indices[graph_mask])
                tf_by_graph[graph_idx]["child"].append(flat_child_indices[graph_mask])

        edge_weight = mask.reshape(-1).to(ar_count.dtype)
        global_parent_indices = (
            rows[:, None] * num_nodes + rank_batch.parent_indices
        ).reshape(-1)
        ar_contrib = ar_predicted.reshape(
            -1,
            model.node_embedding_dim,
        ) * edge_weight.unsqueeze(-1)
        ar_sum.view(batch_size * num_nodes, -1).index_add_(
            0,
            global_parent_indices,
            ar_contrib.to(ar_sum.dtype),
        )
        ar_count.view(-1).index_add_(0, global_parent_indices, edge_weight)
        ar_sumsq.view(batch_size * num_nodes, -1).index_add_(
            0,
            global_parent_indices,
            (ar_contrib**2).to(ar_sumsq.dtype),
        )

        if include_teacher_forced and tf_predicted is not None:
            tf_contrib = tf_predicted.reshape(
                -1,
                model.node_embedding_dim,
            ) * edge_weight.unsqueeze(-1)
            tf_sum.view(batch_size * num_nodes, -1).index_add_(
                0,
                global_parent_indices,
                tf_contrib.to(tf_sum.dtype),
            )
            tf_count.view(-1).index_add_(0, global_parent_indices, edge_weight)
            tf_sumsq.view(batch_size * num_nodes, -1).index_add_(
                0,
                global_parent_indices,
                (tf_contrib**2).to(tf_sumsq.dtype),
            )

    def _stack(
        by_graph: dict[int, dict[str, list[Tensor]]],
    ) -> dict[int, dict[str, Tensor]]:
        stacked = {}
        for graph_idx, chunks in by_graph.items():
            stacked[graph_idx] = {
                "pred": (
                    torch.cat(chunks["pred"])
                    if chunks["pred"]
                    else encoder_buffer.new_zeros((0, model.node_embedding_dim))
                ),
                "true_parent": (
                    torch.cat(chunks["true_parent"])
                    if chunks["true_parent"]
                    else encoder_buffer.new_zeros((0, model.node_embedding_dim))
                ),
                "parent": (
                    torch.cat(chunks["parent"])
                    if chunks["parent"]
                    else torch.empty(0, dtype=torch.long, device=device)
                ),
                "child": (
                    torch.cat(chunks["child"])
                    if chunks["child"]
                    else torch.empty(0, dtype=torch.long, device=device)
                ),
            }
        return stacked

    return _stack(ar_by_graph), _stack(tf_by_graph) if tf_by_graph is not None else None


def _print_stats(acc: SimilarityAccumulator) -> None:
    values = acc.tensor()
    print(f"\n{acc.name}")
    print(f"  pairs: {values.numel()}")
    if values.numel() == 0:
        return
    quantiles = torch.tensor([0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    q = torch.quantile(values, quantiles)
    print(
        "  mean/std: "
        f"{values.mean().item():.6f} / {values.std(unbiased=False).item():.6f}"
    )
    print(f"  min/max : {values.min().item():.6f} / {values.max().item():.6f}")
    print(
        "  q01/q05/q10/q50/q90/q95/q99: "
        + " / ".join(f"{x:.6f}" for x in q.tolist())
    )
    for threshold in (0.80, 0.85, 0.90, 0.95, 0.99):
        frac = (values >= threshold).float().mean().item()
        print(f"  frac >= {threshold:.2f}: {frac:.6f}")


def _make_accumulators(prefix: str) -> dict[str, SimilarityAccumulator]:
    return {
        "same": SimilarityAccumulator(
            f"{prefix}: same target parent, different children"
        ),
        "different": SimilarityAccumulator(
            f"{prefix}: different target parents, different children"
        ),
        "different_true": SimilarityAccumulator(
            f"{prefix}: true encoder embeddings for those different-parent pairs"
        ),
        "different_nonroot": SimilarityAccumulator(
            f"{prefix}: different non-root target parents, different children"
        ),
    }


def run_benchmark(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device or cfg.DEVICE)
    generator = torch.Generator().manual_seed(args.seed)

    print(f"device={device}  ckpt={args.ckpt}")
    model = build_model(device)
    step = load_checkpoint(model, Path(args.ckpt), device)
    model.eval()
    print(f"loaded checkpoint at step={step}")

    ar_acc = _make_accumulators("autoregressive")
    tf_acc = (
        _make_accumulators("teacher-forced") if args.include_teacher_forced else None
    )

    remaining = args.num_graphs
    with tqdm(total=args.num_graphs, desc="Benchmarking graphs") as progress:
        while remaining:
            batch_size = min(args.batch_size, remaining)
            graphs = _make_graphs(batch_size)
            ar_by_graph, tf_by_graph = collect_parent_predictions(
                model,
                graphs,
                include_teacher_forced=args.include_teacher_forced,
            )
            for graph_idx in range(batch_size):
                data = ar_by_graph[graph_idx]
                _add_pair_stats(
                    embeddings=data["pred"],
                    true_parent_embeddings=data["true_parent"],
                    parent_ids=data["parent"],
                    child_ids=data["child"],
                    same_parent=ar_acc["same"],
                    different_parent=ar_acc["different"],
                    different_parent_true=ar_acc["different_true"],
                    different_nonroot_parent=ar_acc["different_nonroot"],
                    root_count=cfg.NUM_ROOT_NODES,
                    max_different_pairs_per_graph=args.max_different_pairs_per_graph,
                    generator=generator,
                )
                if tf_acc is not None and tf_by_graph is not None:
                    data = tf_by_graph[graph_idx]
                    _add_pair_stats(
                        embeddings=data["pred"],
                        true_parent_embeddings=data["true_parent"],
                        parent_ids=data["parent"],
                        child_ids=data["child"],
                        same_parent=tf_acc["same"],
                        different_parent=tf_acc["different"],
                        different_parent_true=tf_acc["different_true"],
                        different_nonroot_parent=tf_acc["different_nonroot"],
                        root_count=cfg.NUM_ROOT_NODES,
                        max_different_pairs_per_graph=args.max_different_pairs_per_graph,
                        generator=generator,
                    )
            remaining -= batch_size
            progress.update(batch_size)

    print("\n================ PARENT-PREDICTION SIMILARITY ================")
    print(f"graphs: {args.num_graphs}")
    print(f"batch size: {args.batch_size}")
    print(f"seed: {args.seed}")
    print(f"max different pairs per graph: {args.max_different_pairs_per_graph}")

    for acc in ar_acc.values():
        _print_stats(acc)
    if tf_acc is not None:
        for acc in tf_acc.values():
            _print_stats(acc)

    print("\n==============================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        default="runs/best.ckpt",
        help="Path to a checkpoint trained with parent prediction losses.",
    )
    parser.add_argument("--num-graphs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=cfg.GRAPH_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (default: config.DEVICE).",
    )
    parser.add_argument(
        "--include-teacher-forced",
        action="store_true",
        help="Also report the teacher-forced decoder pass.",
    )
    parser.add_argument(
        "--max-different-pairs-per-graph",
        type=int,
        default=None,
        help="Subsample different-parent pairs per graph; default keeps all pairs.",
    )
    return parser.parse_args()


def main() -> None:
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
