"""Eyeball what a proposal network would actually be conditioned on.

Renders the truth-table images (see
:mod:`dagnabbit.tasks.logic_gates.truth_table_image`) of a handful of random
graphs alongside the two reference circuits and the adder target. The question
this answers, before any model is built: **do random graphs produce structured
images, or noise?**

Structure means a conditioning network has something to learn from. A random
128-gate NAND/NOR circuit cannot compute an arbitrary boolean function -- it
tends to depend on a few input bits -- so its bit planes should come out
blocky and striped rather than static. If they come out as static, the whole
"train on random graphs, generalize to structured targets" plan is in trouble
and it is much cheaper to find that out here than after training something.

The second figure compares binary against Gray-coded axis ordering, for both a
random graph and the target. That comparison is what set the default in
:mod:`dagnabbit.tasks.logic_gates.truth_table_image`: Gray coding is the right
prior for a boolean target, but these targets are *arithmetic*, and binary
ordering is what makes them simple.

Writes PNGs; nothing here trains or loads a checkpoint.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.scripts import config as cfg
from dagnabbit.tasks.logic_gates.evaluate import adder_task
from dagnabbit.tasks.logic_gates.proposer import behaviour_images
from dagnabbit.tasks.logic_gates.roundtrip_probe import reference_circuits
from dagnabbit.tasks.logic_gates.truth_table_image import (
    image_dimensions,
    task_target_image,
)


def bit_plane_figure(
    images: list[torch.Tensor],
    labels: list[str],
    title: str,
) -> plt.Figure:
    """Grid of every output bit plane: one row per circuit, one column per bit."""
    num_rows = len(images)
    num_channels = images[0].shape[0]
    figure, axes = plt.subplots(
        num_rows,
        num_channels,
        figsize=(1.5 * num_channels, 1.6 * num_rows),
        squeeze=False,
    )
    for row, (image, label) in enumerate(zip(images, labels)):
        for channel in range(num_channels):
            axis = axes[row][channel]
            axis.imshow(
                image[channel].cpu().numpy(),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"bit {channel}", fontsize=8)
            if channel == 0:
                axis.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    return figure


def density_summary(image: torch.Tensor) -> str:
    """Per-bit fraction of set pixels -- a constant plane is a dead output."""
    densities = image.float().mean(dim=(-2, -1))
    return " ".join(f"{value:.2f}" for value in densities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-random", type=int, default=6)
    parser.add_argument("--output-dir", default="profile_out/truth_table_images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = adder_task("cpu")
    height, width = image_dimensions(task.root_values.shape[0])
    print(f"truth-table grid is {height}x{width}, {task.target_values.shape[0]} bits")

    graphs = [
        make_random_graph_description(
            num_root_nodes=cfg.NUM_ROOT_NODES,
            num_trunk_nodes=cfg.NUM_TRUNK_NODES,
            num_output_nodes=cfg.NUM_OUTPUT_NODES,
            trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
            num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        )
        for _ in range(args.num_random)
    ]
    random_images = behaviour_images(graphs, task)

    # The reference circuits are evaluated on the adder's inputs so every image
    # in the figure shares one coordinate system.
    circuits = reference_circuits("cpu")
    reference_images = behaviour_images([circuit.graph for circuit in circuits], task)

    images = list(random_images) + list(reference_images) + [task_target_image(task)]
    labels = (
        [f"random {index}" for index in range(len(random_images))]
        + [f"{circuit.name} circuit" for circuit in circuits]
        + ["ADDER TARGET"]
    )

    print("\nper-bit set-pixel density (0.50 = balanced, 0.00/1.00 = dead output)")
    for label, image in zip(labels, images):
        print(f"  {label:>16}: {density_summary(image)}")

    figure = bit_plane_figure(
        images,
        labels,
        "Truth-table images, binary axis ordering (the training layout)",
    )
    path = output_dir / "bit_planes.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    print(f"\nwrote {path}")

    # Which layout to train in is decided by which one makes random graphs and
    # the target look most alike -- that gap is what the network has to cross.
    # Both rows of each block share a layout so the comparison is like-for-like.
    channels = [0, 2, 4, 7]
    comparison, axes = plt.subplots(4, len(channels), figsize=(10, 10.8), squeeze=False)
    for column, channel in enumerate(channels):
        for layout_index, gray in enumerate([False, True]):
            target = task_target_image(task, gray=gray)
            sample = behaviour_images(graphs[:1], task, gray=gray)[0]
            layout = "gray" if gray else "binary"
            for offset, (image, name) in enumerate(
                [(target, "target"), (sample, "random")]
            ):
                axis = axes[2 * layout_index + offset][column]
                axis.imshow(
                    image[channel].numpy(),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_title(f"bit {channel} {name}, {layout}", fontsize=9)
    comparison.suptitle(
        "Binary vs Gray axis ordering: adder target against a random graph",
        fontsize=11,
    )
    comparison.tight_layout()
    path = output_dir / "gray_vs_binary.png"
    comparison.savefig(path, dpi=110)
    plt.close(comparison)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
