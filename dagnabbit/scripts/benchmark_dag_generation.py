"""Benchmark random DAG generation throughput.

Run directly:

    uv run python -m dagnabbit.scripts.benchmark_dag_generation
"""

import argparse
import time

import torch

from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.scripts import config as cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--num-root-nodes", type=int, default=cfg.NUM_ROOT_NODES)
    parser.add_argument("--num-trunk-nodes", type=int, default=cfg.NUM_TRUNK_NODES)
    parser.add_argument("--num-output-nodes", type=int, default=cfg.NUM_OUTPUT_NODES)
    parser.add_argument(
        "--trunk-node-in-degrees",
        type=int,
        nargs="+",
        default=None,
        help=(
            "One integer for all trunk types, or one integer per trunk type. "
            "Defaults to config.TRUNK_NODE_TYPE_IN_DEGREES."
        ),
    )
    parser.add_argument(
        "--num-trunk-node-types",
        type=int,
        default=cfg.NUM_TRUNK_NODE_TYPES,
    )
    parser.add_argument("--warmup-graphs", type=int, default=5)
    parser.add_argument("--min-graphs", type=int, default=20)
    parser.add_argument("--min-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trunk_node_in_degrees: int | list[int]
    if args.trunk_node_in_degrees is None:
        trunk_node_in_degrees = cfg.TRUNK_NODE_TYPE_IN_DEGREES
    elif len(args.trunk_node_in_degrees) == 1:
        trunk_node_in_degrees = args.trunk_node_in_degrees[0]
    else:
        trunk_node_in_degrees = args.trunk_node_in_degrees

    params = dict(
        num_root_nodes=args.num_root_nodes,
        num_trunk_nodes=args.num_trunk_nodes,
        num_output_nodes=args.num_output_nodes,
        trunk_node_in_degrees=trunk_node_in_degrees,
        num_trunk_node_types=args.num_trunk_node_types,
    )

    def make_graph() -> None:
        make_random_graph_description(**params)

    torch.manual_seed(args.seed)
    for _ in range(args.warmup_graphs):
        make_graph()

    count = 0
    start = time.perf_counter()
    while True:
        make_graph()
        count += 1
        elapsed = time.perf_counter() - start
        if count >= args.min_graphs and elapsed >= args.min_seconds:
            break

    seconds_per_graph = elapsed / count
    graphs_per_second = count / elapsed
    print("DAG generation benchmark")
    print(f"  seed:                 {args.seed}")
    print(f"  roots:                {args.num_root_nodes}")
    print(f"  trunks:               {args.num_trunk_nodes}")
    print(f"  outputs:              {args.num_output_nodes}")
    print(f"  trunk types:          {args.num_trunk_node_types}")
    print(f"  trunk in-degrees:     {trunk_node_in_degrees}")
    print(f"  warmup graphs:        {args.warmup_graphs}")
    print(f"  measured graphs:      {count}")
    print(f"  elapsed seconds:      {elapsed:.6f}")
    print(f"  seconds per graph:    {seconds_per_graph:.6f}")
    print(f"  graphs per second:    {graphs_per_second:.2f}")


if __name__ == "__main__":
    main()
