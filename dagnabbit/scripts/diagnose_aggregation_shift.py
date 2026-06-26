"""Diagnose the single-embedding vs. aggregate domain shift in the decoder.

The model is trained on *aggregates*: a node's classification/decode input is the
sqrt-normalized sum of every child's prediction of it (``sum / sqrt(count)``; see
``DagnabbitAutoEncoder._decode_graph`` and the encoder pooling). Blind decode
instead classifies/expands each node from a *single* child-predicted embedding,
which is out of distribution.

This script isolates how much classification accuracy is lost purely from that
shift, holding graph structure fixed (ground-truth parent identities). For every
parent node it compares, using the type classifier masked exactly as blind decode
masks it (output types forbidden):

  * aggregate : classify ``sum(child_preds) / sqrt(count)``  -> reproduces the
                ~99% training metric.
  * single    : classify each individual child's prediction of the parent.
  * encoder   : classify the parent's own encoder embedding (a ceiling).

It reports both the teacher-forced pass (children decoded from clean true
embeddings -> pure aggregation shift) and the autoregressive pass (predictions
compound down the DAG -> shift + exposure bias). Accuracy is split by ROOT vs
TRUNK parents, and single-input accuracy is also bucketed by how many children
voted on the parent. Supporting distributions (within-parent prediction variance,
single-vs-aggregate cosine, embedding norms) quantify the "variance too high"
hypothesis directly.

Run:

    uv run python -m dagnabbit.scripts.diagnose_aggregation_shift \
        --ckpt "runs/20260623-190239-posenc-spread-0.08->0.02_0.1tfreconloss/checkpoints/graphs-003840000.ckpt" \
        --num-graphs 128
"""

from __future__ import annotations

# Keep this diagnostic pure-eager and comparable to the other harnesses.
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_random_graph_description,
)
from dagnabbit.scripts import config as cfg
from dagnabbit.scripts.benchmark_parent_prediction_similarity import (
    collect_parent_predictions,
)
from dagnabbit.scripts.roundtrip_blind_decode import build_model, load_checkpoint

ROOT_START = cfg.NUM_TRUNK_NODE_TYPES
OUTPUT_START = cfg.NUM_TRUNK_NODE_TYPES + cfg.NUM_ROOT_NODES


@dataclass
class AccuracyCounter:
    """Correct / total split by parent supertype ('root' / 'trunk')."""

    correct: dict[str, int] = field(default_factory=lambda: {"root": 0, "trunk": 0})
    total: dict[str, int] = field(default_factory=lambda: {"root": 0, "trunk": 0})

    def add(self, predicted_types: Tensor, true_types: Tensor) -> None:
        is_correct = predicted_types == true_types
        is_root = (true_types >= ROOT_START) & (true_types < OUTPUT_START)
        is_trunk = true_types < ROOT_START
        for name, mask in (("root", is_root), ("trunk", is_trunk)):
            self.total[name] += int(mask.sum())
            self.correct[name] += int((is_correct & mask).sum())

    def accuracy(self, name: str) -> float:
        total = self.total[name]
        return self.correct[name] / total if total else float("nan")

    def overall(self) -> float:
        total = self.total["root"] + self.total["trunk"]
        correct = self.correct["root"] + self.correct["trunk"]
        return correct / total if total else float("nan")


@dataclass
class Distributions:
    """CPU chunks of supporting per-parent / per-edge distributions."""

    within_parent_variance: list[Tensor] = field(default_factory=list)
    single_to_aggregate_cosine: list[Tensor] = field(default_factory=list)
    single_norm: list[Tensor] = field(default_factory=list)
    aggregate_norm: list[Tensor] = field(default_factory=list)
    encoder_norm: list[Tensor] = field(default_factory=list)

    def cat(self, attr: str) -> Tensor:
        chunks = getattr(self, attr)
        return torch.cat(chunks) if chunks else torch.empty(0)


def classify(model: DagnabbitAutoEncoder, embeddings: Tensor) -> Tensor:
    """Predict node subtype with output types forbidden, as blind decode does."""
    logits = model.node_type_predictor(embeddings).float()
    logits[:, OUTPUT_START:] = float("-inf")
    return logits.argmax(dim=-1)


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


def _process_graph_pass(
    model: DagnabbitAutoEncoder,
    graph: FixedInDegreeDAGDescription,
    data: dict[str, Tensor],
    *,
    aggregate_acc: AccuracyCounter,
    single_acc: AccuracyCounter,
    encoder_acc: AccuracyCounter,
    count_bucket_correct: dict[int, list[int]],
    distributions: Distributions,
) -> None:
    """Score one pass (ar or tf) of one graph's child->parent predictions.

    ``data`` carries edge-flat tensors: ``pred`` (child's prediction of parent),
    ``true_parent`` (parent's encoder embedding), and the integer ``parent``
    node indices. Edges are grouped by parent to rebuild the aggregate exactly.
    """
    pred = data["pred"]  # [E, D]
    if pred.shape[0] == 0:
        return
    parent = data["parent"]  # [E]
    true_parent_emb = data["true_parent"]  # [E, D]
    device = pred.device

    node_types = graph.node_types_tensor.to(device)
    true_type_edge = node_types[parent]  # [E]

    # ---- single-embedding classification (the OOD path blind decode uses) ----
    single_pred_type = classify(model, pred)
    single_acc.add(single_pred_type, true_type_edge)

    # ---- group edges by parent to rebuild aggregates exactly ----
    uniq_parent, inverse = torch.unique(parent, return_inverse=True)
    num_parents = uniq_parent.shape[0]
    counts = torch.zeros(num_parents, device=device).index_add_(
        0, inverse, torch.ones_like(inverse, dtype=pred.dtype)
    )
    sum_pred = torch.zeros(num_parents, pred.shape[1], device=device, dtype=pred.dtype)
    sum_pred.index_add_(0, inverse, pred)
    sumsq_pred = torch.zeros_like(sum_pred)
    sumsq_pred.index_add_(0, inverse, pred * pred)

    aggregate = sum_pred / counts.sqrt().unsqueeze(-1)  # sum / sqrt(count)
    true_type_parent = node_types[uniq_parent]
    aggregate_acc.add(classify(model, aggregate), true_type_parent)

    # Encoder-embedding ceiling: identical across a parent's edges; last write wins.
    encoder_emb = torch.zeros_like(sum_pred)
    encoder_emb[inverse] = true_parent_emb
    encoder_acc.add(classify(model, encoder_emb), true_type_parent)

    # ---- single accuracy bucketed by how many children voted on the parent ----
    parent_count_per_edge = counts[inverse]
    single_correct = single_pred_type == true_type_edge
    for count_value in parent_count_per_edge.unique().tolist():
        bucket = int(count_value)
        edge_mask = parent_count_per_edge == count_value
        stats = count_bucket_correct.setdefault(bucket, [0, 0])
        stats[0] += int((single_correct & edge_mask).sum())
        stats[1] += int(edge_mask.sum())

    # ---- supporting distributions ----
    multi_child = counts >= 2
    if int(multi_child.sum()):
        mean = sum_pred / counts.unsqueeze(-1)
        variance = (
            (sumsq_pred / counts.unsqueeze(-1) - mean**2).clamp(min=0).mean(dim=-1)
        )
        distributions.within_parent_variance.append(variance[multi_child].cpu())

        aggregate_per_edge = aggregate[inverse]
        cosine = F.cosine_similarity(pred.float(), aggregate_per_edge.float(), dim=-1)
        distributions.single_to_aggregate_cosine.append(
            cosine[multi_child[inverse]].cpu()
        )

    distributions.single_norm.append(pred.norm(dim=-1).cpu())
    distributions.aggregate_norm.append(aggregate.norm(dim=-1).cpu())
    distributions.encoder_norm.append(encoder_emb.norm(dim=-1).cpu())


def _print_accuracy(title: str, acc: AccuracyCounter) -> None:
    print(f"\n  {title}")
    print(
        f"    root : {acc.accuracy('root'):.4f}  "
        f"({acc.correct['root']} / {acc.total['root']})"
    )
    print(
        f"    trunk: {acc.accuracy('trunk'):.4f}  "
        f"({acc.correct['trunk']} / {acc.total['trunk']})"
    )
    print(f"    all  : {acc.overall():.4f}")


def _print_distribution(name: str, values: Tensor) -> None:
    if values.numel() == 0:
        print(f"  {name}: (no samples)")
        return
    quantiles = torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99])
    q = torch.quantile(values.float(), quantiles)
    print(
        f"  {name}: mean={values.mean():.4f} std={values.std(unbiased=False):.4f} "
        f"| q01/q10/q50/q90/q99 = " + " / ".join(f"{x:.4f}" for x in q.tolist())
    )


def _report_pass(
    label: str,
    aggregate_acc: AccuracyCounter,
    single_acc: AccuracyCounter,
    encoder_acc: AccuracyCounter,
    count_bucket_correct: dict[int, list[int]],
    distributions: Distributions,
) -> None:
    print(f"\n================ {label} ================")
    _print_accuracy("aggregate input  (sum / sqrt(count)) -- training regime", aggregate_acc)
    _print_accuracy("single-child input -- blind-decode regime", single_acc)
    _print_accuracy("encoder embedding -- classifier ceiling", encoder_acc)

    overall_gap = aggregate_acc.overall() - single_acc.overall()
    print(f"\n  >>> aggregation OOD gap (aggregate - single): {overall_gap:+.4f}")

    print("\n  single-input accuracy by #children voting on the parent:")
    for bucket in sorted(count_bucket_correct):
        correct, total = count_bucket_correct[bucket]
        rate = correct / total if total else float("nan")
        print(f"    count={bucket:>3}: {rate:.4f}  ({correct} / {total})")

    print("\n  supporting distributions (parents with >= 2 children unless noted):")
    _print_distribution(
        "within-parent pred variance (avg over D)",
        distributions.cat("within_parent_variance"),
    )
    _print_distribution(
        "cos(single pred, parent aggregate)",
        distributions.cat("single_to_aggregate_cosine"),
    )
    _print_distribution(
        "||single pred||           (all edges)",
        distributions.cat("single_norm"),
    )
    _print_distribution(
        "||aggregate||             (all parents)",
        distributions.cat("aggregate_norm"),
    )
    _print_distribution(
        "||encoder emb||           (all parents)",
        distributions.cat("encoder_norm"),
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device or cfg.DEVICE)
    print(f"device={device}  ckpt={args.ckpt}")

    model = build_model(device)
    step = load_checkpoint(model, Path(args.ckpt), device)
    model.eval()
    print(f"loaded checkpoint at step={step}")

    passes = {"AUTOREGRESSIVE": "ar", "TEACHER-FORCED": "tf"}
    aggregate_acc = {p: AccuracyCounter() for p in passes}
    single_acc = {p: AccuracyCounter() for p in passes}
    encoder_acc = {p: AccuracyCounter() for p in passes}
    count_buckets: dict[str, dict[int, list[int]]] = {p: {} for p in passes}
    distributions = {p: Distributions() for p in passes}

    remaining = args.num_graphs
    with tqdm(total=args.num_graphs, desc="Diagnosing graphs") as progress:
        while remaining:
            batch_size = min(args.batch_size, remaining)
            graphs = _make_graphs(batch_size)
            ar_by_graph, tf_by_graph = collect_parent_predictions(
                model, graphs, include_teacher_forced=True
            )
            by_pass = {"AUTOREGRESSIVE": ar_by_graph, "TEACHER-FORCED": tf_by_graph}
            for pass_label in passes:
                graph_data = by_pass[pass_label]
                assert graph_data is not None
                for graph_idx, graph in enumerate(graphs):
                    _process_graph_pass(
                        model,
                        graph,
                        graph_data[graph_idx],
                        aggregate_acc=aggregate_acc[pass_label],
                        single_acc=single_acc[pass_label],
                        encoder_acc=encoder_acc[pass_label],
                        count_bucket_correct=count_buckets[pass_label],
                        distributions=distributions[pass_label],
                    )
            remaining -= batch_size
            progress.update(batch_size)

    print("\n###############################################################")
    print("#  AGGREGATION DOMAIN-SHIFT DIAGNOSTIC")
    print(f"#  graphs={args.num_graphs}  seed={args.seed}")
    print("#  aggregate := sum(child preds) / sqrt(count)  (training regime)")
    print("#  single    := one child's prediction of the parent (blind-decode regime)")
    print("###############################################################")

    for pass_label in passes:
        _report_pass(
            pass_label,
            aggregate_acc[pass_label],
            single_acc[pass_label],
            encoder_acc[pass_label],
            count_buckets[pass_label],
            distributions[pass_label],
        )

    print("\nInterpretation:")
    print("  * TEACHER-FORCED gap isolates the *pure* aggregation shift (no compounding).")
    print("    Small gap  -> single embeddings are fine; blind-decode failure is")
    print("                  inconsistency/merging, not OOD. Favor consensus/merge.")
    print("    Large gap  -> the decoder genuinely needs aggregates; any single-")
    print("                  embedding scheme (incl. DFS) is fighting the training")
    print("                  distribution. Favor aggregate-matching / re-aggregation.")
    print("  * AUTOREGRESSIVE - TEACHER-FORCED gap measures exposure-bias compounding.")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="runs/best.ckpt")
    parser.add_argument("--num-graphs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=cfg.GRAPH_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Override config.DEVICE.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
