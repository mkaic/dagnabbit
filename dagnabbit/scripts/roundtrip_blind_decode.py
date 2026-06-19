"""Round-trip minimal DAGs through blind autoregressive global-pool decode.

Run:

    uv run python -m dagnabbit.scripts.roundtrip_blind_decode \
        --ckpt runs/<run>/best.ckpt --num-graphs 128 --similarity-threshold 0.99
"""

from __future__ import annotations

# Keep decode/eval pure-eager for this diagnostic harness.
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    canonicalize,
    graphs_match,
    make_random_graph_description,
)
from dagnabbit.scripts import config as cfg
from dagnabbit.scripts.diagnose_root_collapse import build_model, load_checkpoint


def _output_cone_indices(graph: FixedInDegreeDAGDescription) -> set[int]:
    output_start = graph.num_root_nodes + graph.num_trunk_nodes
    seen = set(range(output_start, graph.num_nodes))
    stack = list(seen)
    while stack:
        node_idx = stack.pop()
        for parent_idx in graph.node_inputs_indices[node_idx]:
            if parent_idx not in seen:
                seen.add(parent_idx)
                stack.append(parent_idx)
    return seen


def _canonical_output_cone_size(graph: FixedInDegreeDAGDescription) -> int:
    canonical_ids = canonicalize(graph)
    return len({canonical_ids[i] for i in _output_cone_indices(graph)})


def _size_bucket(num_nodes: int) -> str:
    for upper in (16, 32, 64, 128, 256, 512, 1024, 2048):
        if num_nodes <= upper:
            return f"<= {upper}"
    return "> 2048"


@torch.no_grad()
def run_roundtrip(
    model: DagnabbitAutoEncoder,
    *,
    num_graphs: int,
    seed: int,
    similarity_threshold: float,
    root_match_margin: float | None,
    max_nodes: int,
) -> dict:
    torch.manual_seed(seed)
    model.eval()

    total = 0
    survived = 0
    over_merge = 0
    under_merge = 0
    equal_size_miss = 0
    failures: Counter[str] = Counter()
    termination_reasons: Counter[str] = Counter()
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "survived": 0})

    for _ in tqdm(range(num_graphs), desc="Round-tripping graphs"):
        graph = make_random_graph_description(
            num_root_nodes=cfg.NUM_ROOT_NODES,
            num_trunk_nodes=cfg.NUM_TRUNK_NODES,
            num_output_nodes=cfg.NUM_OUTPUT_NODES,
            trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
            num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        )

        try:
            buffer = DagnabbitAutoEncoder.evaluate_graph(
                graph=graph,
                root_node_embeddings=model.root_node_embeddings.weight,
                node_autoencoders=model.node_autoencoders,
                node_embedding_dim=model.node_embedding_dim,
            )
            output_embeddings = buffer[-graph.num_output_nodes :]
            recovered, diagnostics = model.blind_autoregressive_decode(
                output_embeddings,
                similarity_threshold=similarity_threshold,
                root_match_margin=root_match_margin,
                max_nodes=max_nodes,
                return_diagnostics=True,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics should keep sweeping.
            failures[type(exc).__name__] += 1
            continue

        total += 1
        termination_reasons[str(diagnostics["termination_reason"])] += 1

        matched = graphs_match(graph, recovered)
        survived += int(matched)

        bucket = _size_bucket(graph.num_nodes)
        buckets[bucket]["total"] += 1
        buckets[bucket]["survived"] += int(matched)

        source_size = _canonical_output_cone_size(graph)
        recovered_size = _canonical_output_cone_size(recovered)
        if recovered_size < source_size:
            over_merge += 1
        elif recovered_size > source_size:
            under_merge += 1
        elif not matched:
            equal_size_miss += 1

    return {
        "attempted": num_graphs,
        "completed": total,
        "survived": survived,
        "over_merge": over_merge,
        "under_merge": under_merge,
        "equal_size_miss": equal_size_miss,
        "failures": failures,
        "termination_reasons": termination_reasons,
        "buckets": dict(sorted(buckets.items())),
    }


def report(results: dict) -> None:
    completed = results["completed"]
    survived = results["survived"]
    survival_rate = survived / completed if completed else float("nan")

    print("\n================ BLIND-DECODE ROUND TRIP ================\n")
    print(f"attempted graphs : {results['attempted']}")
    print(f"completed graphs : {completed}")
    print(f"survived         : {survived} ({survival_rate:.3f})")
    print(f"over-merge       : {results['over_merge']}")
    print(f"under-merge      : {results['under_merge']}")
    print(f"equal-size misses: {results['equal_size_miss']}")

    print("\nSurvival by source node count:")
    for bucket, counts in results["buckets"].items():
        rate = counts["survived"] / counts["total"] if counts["total"] else float("nan")
        print(
            f"  {bucket:>6}: {counts['survived']:5d} / {counts['total']:<5d} "
            f"({rate:.3f})"
        )

    print("\nTermination reasons:")
    for reason, count in results["termination_reasons"].most_common():
        print(f"  {reason}: {count}")

    if results["failures"]:
        print("\nDecode failures:")
        for reason, count in results["failures"].most_common():
            print(f"  {reason}: {count}")

    print("\n=========================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        default="runs/best.ckpt",
        help="Path to a checkpoint trained with parent-reconstruction losses.",
    )
    parser.add_argument("--num-graphs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (default: config.DEVICE).",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.99)
    parser.add_argument("--root-match-margin", type=float, default=None)
    parser.add_argument("--max-nodes", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or cfg.DEVICE)
    print(f"device={device}  ckpt={args.ckpt}")

    model = build_model(device)
    step = load_checkpoint(model, Path(args.ckpt), device)
    print(f"loaded checkpoint at step={step}")

    results = run_roundtrip(
        model,
        num_graphs=args.num_graphs,
        seed=args.seed,
        similarity_threshold=args.similarity_threshold,
        root_match_margin=args.root_match_margin,
        max_nodes=args.max_nodes,
    )
    report(results)


if __name__ == "__main__":
    main()
