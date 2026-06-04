"""Signal-propagation sanity check for the DAG autoencoder.

Because ``evaluate_graph`` recursively composes a ``NodeEncoder`` MLP at every
non-root node, the effective depth of the computation equals the depth of the
DAG -- so a freshly-initialized model is really a very deep, weight-shared,
residual-free network. With the Scaled-Weight-Standardization linears
(``StandardizedLinear``) and the variance-preserving ``GammaScaledGELU``, the
per-node embedding norm should stay flat around ``sqrt(node_embedding_dim)`` as
DAG depth increases, and the per-element mean should stay near zero (no
mean-shift), *without* any brute-force normalization.

Run it directly to eyeball the depth-vs-statistics table::

    python -m dagnabbit.dag.tests.test_signal_propagation

The ``test_*`` functions additionally assert that norms stay bounded (i.e. do
not explode or collapse) so this doubles as a regression guard.
"""

import math

import torch

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_random_graph_description,
)
from dagnabbit.scripts import config as cfg


def build_model() -> DagnabbitAutoEncoder:
    return DagnabbitAutoEncoder(
        node_embedding_dim=cfg.NODE_EMBEDDING_DIM,
        trunk_node_type_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        condenser_node_type_in_degree=cfg.CONDENSER_NODE_TYPE_IN_DEGREE,
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        mlp_depth=cfg.MLP_DEPTH,
        mlp_expansion_factor=cfg.MLP_EXPANSION_FACTOR,
    )


def sample_graph() -> FixedInDegreeDAGDescription:
    return make_random_graph_description(
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_trunk_nodes=cfg.NUM_TRUNK_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
    )


def node_depths(graph: FixedInDegreeDAGDescription) -> list[int]:
    """Longest-path depth of each node (roots are depth 0).

    Nodes are stored in topological order (roots first, every trunk/output node
    only references earlier indices), so a single forward sweep suffices.
    """
    depths = [0] * graph.num_nodes
    for node_idx in range(graph.num_root_nodes, graph.num_nodes):
        parents = graph.node_inputs_indices[node_idx]
        depths[node_idx] = 1 + max((depths[p] for p in parents), default=0)
    return depths


def _bucket_stats_by_depth(
    buffer: torch.Tensor,
    depths: list[int],
) -> dict[int, dict[str, float]]:
    """Accumulate per-node norm / element-mean / element-std, bucketed by depth."""
    per_depth: dict[int, list[torch.Tensor]] = {}
    for node_idx, depth in enumerate(depths):
        per_depth.setdefault(depth, []).append(buffer[node_idx])

    stats: dict[int, dict[str, float]] = {}
    for depth, rows in sorted(per_depth.items()):
        stacked = torch.stack(rows)  # [count, D]
        norms = stacked.norm(dim=-1)
        stats[depth] = {
            "count": float(stacked.shape[0]),
            "norm_mean": float(norms.mean()),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "elem_mean": float(stacked.mean()),
            "elem_std": float(stacked.std()),
        }
    return stats


def measure_signal_propagation(
    num_models: int = 4,
    graphs_per_model: int = 4,
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    """Run several freshly-initialized models on several random graphs and return
    primary-buffer statistics aggregated (mean over samples) per DAG depth."""
    torch.manual_seed(seed)

    accum: dict[int, dict[str, list[float]]] = {}
    with torch.no_grad():
        for _ in range(num_models):
            model = build_model()
            for _ in range(graphs_per_model):
                graph = sample_graph()
                _, _, primary_buffer, _ = model.encode_graph_with_condenser(
                    graph, return_buffers=True
                )
                depths = node_depths(graph)
                for depth, stats in _bucket_stats_by_depth(
                    primary_buffer, depths
                ).items():
                    bucket = accum.setdefault(
                        depth, {k: [] for k in stats if k != "count"}
                    )
                    for key, value in stats.items():
                        if key == "count":
                            continue
                        bucket[key].append(value)

    return {
        depth: {key: float(sum(vals) / len(vals)) for key, vals in keyed.items()}
        for depth, keyed in sorted(accum.items())
    }


def format_table(stats_by_depth: dict[int, dict[str, float]]) -> str:
    target = math.sqrt(cfg.NODE_EMBEDDING_DIM)
    header = (
        f"target norm = sqrt(D) = {target:.2f}  (D = {cfg.NODE_EMBEDDING_DIM})\n"
        f"{'depth':>5}  {'norm_mean':>9}  {'norm_min':>9}  {'norm_max':>9}  "
        f"{'elem_mean':>9}  {'elem_std':>9}"
    )
    rows = [
        f"{depth:>5}  {s['norm_mean']:>9.3f}  {s['norm_min']:>9.3f}  "
        f"{s['norm_max']:>9.3f}  {s['elem_mean']:>9.4f}  {s['elem_std']:>9.4f}"
        for depth, s in stats_by_depth.items()
    ]
    return "\n".join([header, *rows])


def test_norms_stay_bounded() -> None:
    """Norms should neither explode nor collapse as DAG depth grows."""
    target = math.sqrt(cfg.NODE_EMBEDDING_DIM)
    stats_by_depth = measure_signal_propagation()

    for depth, s in stats_by_depth.items():
        assert math.isfinite(s["norm_mean"]), f"non-finite norm at depth {depth}"
        # Generous band: the original failure mode was norms exploding by orders
        # of magnitude with depth; this catches that (and collapse) robustly.
        assert 0.25 * target <= s["norm_mean"] <= 4.0 * target, (
            f"depth {depth}: norm_mean {s['norm_mean']:.3f} outside "
            f"[{0.25 * target:.3f}, {4.0 * target:.3f}]"
        )


def test_no_mean_shift() -> None:
    """Per-element mean should stay near zero (Scaled WS removes mean-shift)."""
    stats_by_depth = measure_signal_propagation()
    for depth, s in stats_by_depth.items():
        assert abs(s["elem_mean"]) < 0.5, (
            f"depth {depth}: element mean {s['elem_mean']:.4f} drifted from 0"
        )


def main() -> None:
    stats_by_depth = measure_signal_propagation()
    print("Signal propagation through the primary DAG (fresh init, no training):")
    print(format_table(stats_by_depth))

    test_norms_stay_bounded()
    test_no_mean_shift()
    print("\nALL SIGNAL-PROPAGATION CHECKS PASSED")


if __name__ == "__main__":
    main()
