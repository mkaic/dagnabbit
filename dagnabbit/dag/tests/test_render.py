import argparse

import torch

from dagnabbit.dag.description import (
    make_minimal_random_dag,
    make_random_graph_description,
)
from dagnabbit.dag.render import render_dag


def main():
    parser = argparse.ArgumentParser(
        description="Render a random DAG and its output-ancestor-pruned counterpart."
    )
    parser.add_argument("--seed", type=int, default=0)
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

    # Seed before each construction so the pruned graph is the pruned version of
    # the exact same random full graph (make_minimal_random_dag builds the full
    # graph internally with the same RNG draws, then prunes it).
    torch.manual_seed(args.seed)
    primary = make_random_graph_description(**kwargs)

    torch.manual_seed(args.seed)
    pruned = make_minimal_random_dag(**kwargs)

    primary_path = render_dag(primary, output_path="dag_primary", fmt="png")
    print(
        f"Rendered full graph to {primary_path} "
        f"({primary.num_trunk_nodes} trunk nodes)"
    )

    pruned_path = render_dag(pruned, output_path="dag_pruned", fmt="png")
    print(
        f"Rendered pruned graph to {pruned_path} "
        f"({pruned.num_trunk_nodes} trunk nodes kept)"
    )


if __name__ == "__main__":
    main()
