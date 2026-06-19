"""Render a random fixed-in-degree DAG to ``dag_primary.png`` in the repo root.

Example:

    uv run python -m dagnabbit.scripts.render --seed 7 --roots 4 --trunks 32 --outputs 4
"""

import argparse

import torch

from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.dag.render import render_dag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--roots", type=int, default=4)
    parser.add_argument("--trunks", type=int, default=32)
    parser.add_argument("--outputs", type=int, default=4)
    parser.add_argument("--in-degrees", type=int, default=2)
    parser.add_argument("--types", type=int, default=2)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    dag = make_random_graph_description(
        num_root_nodes=args.roots,
        num_trunk_nodes=args.trunks,
        num_output_nodes=args.outputs,
        trunk_node_in_degrees=args.in_degrees,
        num_trunk_node_types=args.types,
    )

    path = render_dag(dag, output_path="dag_primary", fmt="png")
    print(f"Rendered graph to {path} ({dag.num_trunk_nodes} trunk nodes)")


if __name__ == "__main__":
    main()
