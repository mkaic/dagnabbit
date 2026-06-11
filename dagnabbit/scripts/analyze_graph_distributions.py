"""Sample random DAG descriptions and plot rank / batching statistics."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    NodeSupertype,
    make_random_graph_description,
)
from dagnabbit.scripts import config as cfg

NUM_GRAPHS = 4096
NUM_ROOT_NODES = 4
NUM_OUTPUT_NODES = 4
MIN_TRUNK_NODES = 64
MAX_TRUNK_NODES = 1024
MIN_TRUNK_NODE_TYPES = 1
MAX_TRUNK_NODE_TYPES = 8
OUTPUT_DIR = Path("graph_distribution_plots")


@dataclass
class GraphStats:
    num_nodes: int
    num_trunk_node_types: int
    max_rank: int
    num_supertype_groups: int
    num_execution_groups: int
    avg_execution_group_size: float
    supertype_group_ratio: float


def _batch_key(
    group_supertype: NodeSupertype, subtypes: np.ndarray
) -> tuple[NodeSupertype, int | None]:
    if group_supertype is NodeSupertype.TRUNK:
        return group_supertype, int(subtypes[0])
    return group_supertype, None


def collect_graph_stats(
    graph: FixedInDegreeDAGDescription,
) -> tuple[GraphStats, list[int]]:
    group_sizes: list[int] = []
    batch_keys: set[tuple[NodeSupertype, int | None]] = set()

    for groups in graph.rank_groups:
        for group in groups:
            size = int(group.node_buffer_indices.numel())
            group_sizes.append(size)
            batch_keys.add(_batch_key(group.supertype, group.subtypes.numpy()))

    num_nodes = graph.num_nodes
    num_supertype_groups = len(batch_keys)
    stats = GraphStats(
        num_nodes=num_nodes,
        num_trunk_node_types=graph.num_trunk_node_types,
        max_rank=max(graph.node_ranks, default=0),
        num_supertype_groups=num_supertype_groups,
        num_execution_groups=len(group_sizes),
        avg_execution_group_size=float(np.mean(group_sizes)),
        supertype_group_ratio=num_supertype_groups / num_nodes,
    )
    return stats, group_sizes


def sample_graphs(num_graphs: int, seed: int) -> tuple[list[GraphStats], list[int]]:
    rng = random.Random(seed)
    stats: list[GraphStats] = []
    all_group_sizes: list[int] = []

    for _ in tqdm(range(num_graphs), desc="Sampling graphs"):
        num_trunk_nodes = rng.randint(MIN_TRUNK_NODES, MAX_TRUNK_NODES)
        num_trunk_node_types = rng.randint(MIN_TRUNK_NODE_TYPES, MAX_TRUNK_NODE_TYPES)
        graph = make_random_graph_description(
            num_root_nodes=NUM_ROOT_NODES,
            num_trunk_nodes=num_trunk_nodes,
            num_output_nodes=NUM_OUTPUT_NODES,
            trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
            num_trunk_node_types=num_trunk_node_types,
        )
        graph_stats, group_sizes = collect_graph_stats(graph)
        stats.append(graph_stats)
        all_group_sizes.extend(group_sizes)

    return stats, all_group_sizes


def _save_distribution_plots(
    stats: list[GraphStats], all_group_sizes: list[int], output_dir: Path
) -> None:
    max_ranks = [s.max_rank for s in stats]
    num_supertype_groups = [s.num_supertype_groups for s in stats]
    num_execution_groups = [s.num_execution_groups for s in stats]
    trunk_node_types = [s.num_trunk_node_types for s in stats]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Rank and supertype-group distributions ({NUM_GRAPHS:,} graphs)")

    axes[0, 0].hist(max_ranks, bins=50, color="steelblue", edgecolor="white")
    axes[0, 0].set_title("Max topological rank per graph")
    axes[0, 0].set_xlabel("max rank")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(
        num_supertype_groups,
        bins=range(min(num_supertype_groups), max(num_supertype_groups) + 2),
        color="darkorange",
        edgecolor="white",
        align="left",
    )
    axes[0, 1].set_title("Supertype groups per graph")
    axes[0, 1].set_xlabel("num supertype groups (unique batch keys)")
    axes[0, 1].set_ylabel("count")

    axes[1, 0].hist(num_execution_groups, bins=50, color="seagreen", edgecolor="white")
    axes[1, 0].set_title("Execution groups (RankGroups) per graph")
    axes[1, 0].set_xlabel("num execution groups")
    axes[1, 0].set_ylabel("count")

    axes[1, 1].hist(
        all_group_sizes,
        bins=80,
        color="mediumpurple",
        edgecolor="white",
        log=True,
    )
    axes[1, 1].set_title("Execution group sizes (all graphs, log count)")
    axes[1, 1].set_xlabel("group size (nodes per RankGroup)")
    axes[1, 1].set_ylabel("count (log)")

    fig.tight_layout()
    fig.savefig(output_dir / "distributions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        trunk_node_types,
        bins=range(MIN_TRUNK_NODE_TYPES, MAX_TRUNK_NODE_TYPES + 2),
        color="gray",
        edgecolor="white",
        align="left",
    )
    ax.set_title("Sampled trunk node types per graph")
    ax.set_xlabel("num_trunk_node_types")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_dir / "trunk_node_types_sampled.png", dpi=150)
    plt.close(fig)


def _save_scatter_plots(stats: list[GraphStats], output_dir: Path) -> None:
    avg_sizes = np.array([s.avg_execution_group_size for s in stats])
    num_nodes = np.array([s.num_nodes for s in stats])
    num_supertype_groups = np.array([s.num_supertype_groups for s in stats])
    ratios = np.array([s.supertype_group_ratio for s in stats])

    scatter_style = dict(alpha=0.25, s=8, linewidths=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Average execution group size vs graph shape")

    axes[0].scatter(num_supertype_groups, avg_sizes, c="steelblue", **scatter_style)
    axes[0].set_xlabel("num supertype groups")
    axes[0].set_ylabel("avg execution group size")
    axes[0].set_title("vs supertype groups")

    axes[1].scatter(num_nodes, avg_sizes, c="darkorange", **scatter_style)
    axes[1].set_xlabel("num nodes")
    axes[1].set_ylabel("avg execution group size")
    axes[1].set_title("vs num nodes")

    axes[2].scatter(ratios, avg_sizes, c="seagreen", **scatter_style)
    axes[2].set_xlabel("supertype groups / num nodes")
    axes[2].set_ylabel("avg execution group size")
    axes[2].set_title("vs supertype-group ratio")

    fig.tight_layout()
    fig.savefig(output_dir / "avg_group_size_scatter.png", dpi=150)
    plt.close(fig)


def _print_summary(stats: list[GraphStats]) -> None:
    avg_sizes = [s.avg_execution_group_size for s in stats]
    max_ranks = [s.max_rank for s in stats]
    num_supertype_groups = [s.num_supertype_groups for s in stats]
    num_nodes = [s.num_nodes for s in stats]

    print(f"Generated {len(stats):,} graphs")
    print(
        f"  nodes: min={min(num_nodes)}, max={max(num_nodes)}, mean={np.mean(num_nodes):.1f}"
    )
    print(
        f"  max rank: min={min(max_ranks)}, max={max(max_ranks)}, "
        f"mean={np.mean(max_ranks):.1f}"
    )
    print(
        f"  supertype groups: min={min(num_supertype_groups)}, "
        f"max={max(num_supertype_groups)}, mean={np.mean(num_supertype_groups):.1f}"
    )
    print(
        f"  avg execution group size: min={min(avg_sizes):.2f}, "
        f"max={max(avg_sizes):.2f}, mean={np.mean(avg_sizes):.2f}"
    )
    print(f"Plots saved to {OUTPUT_DIR.resolve()}/")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats, all_group_sizes = sample_graphs(NUM_GRAPHS, seed=cfg.SEED)
    _save_distribution_plots(stats, all_group_sizes, OUTPUT_DIR)
    _save_scatter_plots(stats, OUTPUT_DIR)
    _print_summary(stats)


if __name__ == "__main__":
    main()
