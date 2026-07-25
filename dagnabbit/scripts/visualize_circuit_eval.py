"""End-to-end visual check of bitpacked circuit evaluation.

Runs three circuits against the 8-bit adder truth table and renders their
output bit planes as 256x256 images (input ``a`` across, input ``b`` down):

* the **target** -- the truth table's own sum bits;
* a **known-correct adder** -- 68 hand-wired NAND gates, which must reproduce
  the target exactly and score 1.0;
* a **random circuit** -- the chance baseline a search starts from.

If the packing, slot mapping, gate semantics or rank evaluation were wrong
anywhere, the adder row would visibly diverge from the target row.

Usage::

    python -m dagnabbit.scripts.visualize_circuit_eval
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.scripts import config as cfg
from dagnabbit.tasks.logic_gates.evaluate import (
    adder_task,
    bit_accuracy,
    evaluate_graphs,
)
from dagnabbit.tasks.logic_gates.reference_circuits import nand_ripple_carry_adder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="circuit_eval_check.png",
        help="Path for the rendered figure.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the random circuit."
    )
    return parser.parse_args()


def to_bit_planes(packed: torch.Tensor) -> np.ndarray:
    """[num_outputs, num_words] uint8 -> [num_outputs, 256, 256] of 0/1.

    The truth table flattens its 256x256 grid row-major, so reshaping recovers
    an image whose columns index ``a`` and whose rows index ``b``.
    """
    return np.unpackbits(packed.cpu().numpy(), axis=-1).reshape(-1, 256, 256)


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(args.seed)
    task = adder_task()

    adder = nand_ripple_carry_adder(
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_trunk_nodes=cfg.NUM_TRUNK_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
    )
    random_circuit = make_random_graph_description(
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_trunk_nodes=cfg.NUM_TRUNK_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
    )

    outputs = evaluate_graphs([adder, random_circuit], task)
    overall, per_output = bit_accuracy(outputs, task)

    print(f"adder  fitness {overall[0].item():.10f}  depth {max(adder.node_ranks)}")
    print(f"random fitness {overall[1].item():.10f}")
    exact = torch.equal(outputs[0], task.target_values)
    print(f"adder output is bit-identical to the target: {exact}")

    rows = [
        ("target\n(a + b) mod 256", to_bit_planes(task.target_values), None),
        ("68-NAND adder", to_bit_planes(outputs[0]), per_output[0]),
        ("random circuit", to_bit_planes(outputs[1]), per_output[1]),
    ]

    num_outputs = task.num_output_nodes
    figure, axes = plt.subplots(
        len(rows), num_outputs, figsize=(2.0 * num_outputs, 2.0 * len(rows) + 1.0)
    )
    figure.suptitle(
        "Circuit evaluation end-to-end check\n"
        "each tile is one output bit over all 65536 (a, b) pairs "
        "-- a across, b down",
        fontsize=13,
    )

    for row, (label, planes, accuracies) in enumerate(rows):
        for output_slot in range(num_outputs):
            axis = axes[row, output_slot]
            axis.imshow(
                planes[output_slot],
                cmap="binary",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(
                    f"output {output_slot}\n(sum bit {num_outputs - 1 - output_slot})",
                    fontsize=9,
                )
            if accuracies is not None:
                axis.set_xlabel(f"{accuracies[output_slot].item():.4f}", fontsize=9)
            if output_slot == 0:
                axis.set_ylabel(label, fontsize=10)

    figure.tight_layout()
    path = Path(args.output)
    figure.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
