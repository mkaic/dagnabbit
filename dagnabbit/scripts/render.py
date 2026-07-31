"""Render a sampled DAG, or a reference adder, to a PNG in the repo root.

Examples::

    uv run python -m dagnabbit.scripts.render --seed 7 --trunks 32
    uv run python -m dagnabbit.scripts.render --circuit nand
    uv run python -m dagnabbit.scripts.render --circuit mixed --format svg

The training geometry (128 gates) draws a legible but dense picture; pass a
smaller ``--trunks`` when the point is to read individual wires.
"""

import argparse

import torch

from dagnabbit.dag.graphs import Geometry, sample
from dagnabbit.dag.render import describe, render_dag
from dagnabbit.scripts import config as cfg
from dagnabbit.tasks.logic_gates.reference_circuits import (
    mixed_ripple_carry_adder,
    nand_ripple_carry_adder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--circuit",
        choices=("random", "nand", "mixed"),
        default="random",
        help="a fresh sample, or one of the hand-built reference adders",
    )
    parser.add_argument("--roots", type=int, default=cfg.GEOMETRY.num_root_nodes)
    parser.add_argument("--trunks", type=int, default=cfg.GEOMETRY.num_trunk_nodes)
    parser.add_argument("--outputs", type=int, default=cfg.GEOMETRY.num_output_nodes)
    parser.add_argument("--out", default=None)
    parser.add_argument("--format", default="png")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    geometry = Geometry(
        num_root_nodes=args.roots,
        num_trunk_nodes=args.trunks,
        num_output_nodes=args.outputs,
        num_trunk_node_types=cfg.GATES.num_types,
        trunk_node_in_degrees=cfg.GATES.in_degrees,
    )
    if args.circuit == "random":
        graphs = sample(1, geometry, sampling=cfg.SAMPLING)
    elif args.circuit == "nand":
        graphs = nand_ripple_carry_adder(geometry, cfg.GATES)
    else:
        graphs = mixed_ripple_carry_adder(geometry, cfg.GATES)

    stem = args.out or f"dag_{args.circuit}"
    path = render_dag(graphs, output_path=stem, fmt=args.format)
    print(f"wrote {path}")
    print(f"  {describe(graphs, gate_names=cfg.GATES.names)}")


if __name__ == "__main__":
    main()
