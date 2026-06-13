import argparse

import torch

from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.dag.render import render_dag


def main():
    parser = argparse.ArgumentParser(
        description="Render a random edge-splitting DAG."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-root-nodes", type=int, default=8)
    parser.add_argument("--num-trunk-nodes", type=int, default=128)
    parser.add_argument("--num-output-nodes", type=int, default=4)
    parser.add_argument("--trunk-node-in-degrees", type=int, default=2)
    parser.add_argument("--num-trunk-node-types", type=int, default=2)
    args = parser.parse_args()

    kwargs = dict(
        num_root_nodes=args.num_root_nodes,
        num_trunk_nodes=args.num_trunk_nodes,
        num_output_nodes=args.num_output_nodes,
        trunk_node_in_degrees=args.trunk_node_in_degrees,
        num_trunk_node_types=args.num_trunk_node_types,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
    primary = make_random_graph_description(**kwargs)

    primary_path = render_dag(primary, output_path="dag_primary", fmt="png")
    print(
        f"Rendered graph to {primary_path} "
        f"({primary.num_trunk_nodes} trunk nodes)"
    )


if __name__ == "__main__":
    main()
