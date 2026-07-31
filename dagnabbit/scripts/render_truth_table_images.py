"""Eyeball what the simulator is actually being asked to predict.

Renders the truth-table images (see
:mod:`dagnabbit.tasks.logic_gates.truth_table_image`) of a handful of random
graphs alongside the two reference adders. The question this answers, before
reading any training curve: **do random graphs produce structured images, or
static?**

Structure means there is something to learn. A random 128-gate circuit cannot
compute an arbitrary boolean function -- it tends to depend on a few input bits
-- so its bit planes should come out blocky and striped rather than as noise. If
they come out as noise, "train on random graphs, generalize to structured
targets" is in trouble, and it is much cheaper to find that out here than after
training something.

The adder rows are the contrast: every sum bit is a function of ``a + b``, so in
binary axis ordering they are clean diagonal stripes.

Writes a PNG; nothing here trains or loads a checkpoint.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dagnabbit.dag.graphs import sample
from dagnabbit.scripts import config as cfg
from dagnabbit.tasks.logic_gates.evaluate import (
    evaluate_graphs,
    exhaustive_root_values,
)
from dagnabbit.tasks.logic_gates.reference_circuits import (
    mixed_ripple_carry_adder,
    nand_ripple_carry_adder,
)
from dagnabbit.tasks.logic_gates.truth_table_image import (
    image_dimensions,
    outputs_to_image,
)


def bit_plane_figure(
    images: list[torch.Tensor], labels: list[str], title: str
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
    """Per-bit fraction of set pixels -- 0.00 or 1.00 is a constant output."""
    return " ".join(f"{value:.2f}" for value in image.float().mean(dim=(-2, -1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-random", type=int, default=6)
    parser.add_argument("--output-dir", default="profile_out/truth_table_images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = cfg.GEOMETRY
    height, width = image_dimensions(geometry.num_root_nodes)
    roots = exhaustive_root_values(geometry.num_root_nodes)
    print(
        f"truth-table grid is {height}x{width}, "
        f"{geometry.num_output_nodes} bits per circuit"
    )

    def planes(graphs) -> list[torch.Tensor]:
        return list(outputs_to_image(evaluate_graphs(graphs, roots), height, width))

    # Every image shares one coordinate system: the exhaustive input enumeration.
    images = planes(sample(args.num_random, geometry, sampling=cfg.SAMPLING))
    labels = [f"random {index}" for index in range(len(images))]
    for name, build in (
        ("NAND adder", nand_ripple_carry_adder),
        ("NAND+XOR adder", mixed_ripple_carry_adder),
    ):
        images += planes(build(geometry))
        labels.append(name)

    print("\nper-bit set-pixel density (0.50 = balanced, 0.00/1.00 = constant)")
    for label, image in zip(labels, images):
        print(f"  {label:>16}: {density_summary(image)}")

    figure = bit_plane_figure(
        images, labels, "Truth-table images, binary axis ordering (the training layout)"
    )
    path = output_dir / "bit_planes.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
