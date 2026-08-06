"""Eyeball how far a trained simulator is from the truth on a reference circuit.

Loads a Phase 0 checkpoint, decodes the *full* predicted truth table for each
reference adder the configured gate set can express, and renders it against the
exact table: one figure per (circuit, padding) with a row of ground-truth bit
planes, a row of predicted probabilities, and a row of absolute error, each bit
annotated with its output rank and Matthews correlation.

Both paddings of each circuit are rendered on purpose. The mask-padded minimal
core is in-distribution under the variable-gate-count sampler; the historical
buffer-padded build reaches ranks the training distribution never produces. The
difference between their two figures is therefore a picture of pure depth
extrapolation, with the computed function held exactly fixed.

Usage::

    uv run python -m dagnabbit.scripts.render_predicted_truth_tables \\
        --checkpoint runs/<run>/step_XXXXXXXX.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dagnabbit.dag.metrics import confusion_counts, matthews_correlation
from dagnabbit.dag.model import GraphSimulator, patch_targets
from dagnabbit.scripts import config as cfg
from dagnabbit.scripts.checkpoints import load_simulator
from dagnabbit.scripts.train_simulator import available_probes
from dagnabbit.tasks.logic_gates.evaluate import (
    adder_task,
    bit_accuracy,
    evaluate_graphs,
)
from dagnabbit.tasks.logic_gates.truth_table_image import (
    image_dimensions,
    outputs_to_image,
)


@torch.no_grad()
def predict_full_table(model: GraphSimulator, graphs) -> torch.Tensor:
    """``[1, K, C, rows_per_patch]`` logits over every patch, in patch order."""
    patch_indices = torch.arange(model.config.num_patches, device=graphs.device)
    return model.forward_graphs(graphs, patch_indices).float()


def logits_to_image(logits: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """``[1, K, C, rows_per_patch]`` -> ``[C, H, W]`` probabilities.

    Patches are contiguous row blocks in truth-table order, so concatenating
    them along the row axis and folding is exactly the transform
    :func:`outputs_to_image` applies to the packed exact table.
    """
    probabilities = torch.sigmoid(logits[0])  # [K, C, rows_per_patch]
    flat = probabilities.permute(1, 0, 2).reshape(probabilities.shape[1], -1)
    return flat.reshape(-1, height, width)


def comparison_figure(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    ranks: list[int],
    mcc: list[float],
    title: str,
) -> plt.Figure:
    """Rows: exact bit plane, predicted probability, absolute error."""
    num_bits = truth.shape[0]
    figure, axes = plt.subplots(
        3, num_bits, figsize=(1.6 * num_bits, 5.4), squeeze=False
    )
    row_labels = ("exact", "predicted", "|error|")
    images = (truth.float(), predicted, (predicted - truth.float()).abs())
    for row, (label, stack) in enumerate(zip(row_labels, images)):
        for bit in range(num_bits):
            axis = axes[row][bit]
            axis.imshow(
                stack[bit].cpu().numpy(),
                cmap="gray" if row < 2 else "magma",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(
                    f"bit {bit}\nrank {ranks[bit]}\nmcc {mcc[bit]:+.2f}", fontsize=7
                )
            if bit == 0:
                axis.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", default="profile_out/predicted_truth_tables")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, step = load_simulator(args.checkpoint, device, cfg.GATES.num_types)
    model.eval()
    rows_per_patch = model.decoder.rows_per_patch
    geometry = model.geometry
    height, width = image_dimensions(geometry.num_root_nodes)
    task = adder_task(device)
    all_patches = torch.arange(model.config.num_patches, device=device)

    print(f"checkpoint {args.checkpoint} (step {step}) on {device}")
    header = f"{'circuit':>18}  {'live':>4}  {'acc':>6}  {'mcc':>6}  per-bit mcc"
    print(header)

    for name, build in available_probes(cfg.GATES):
        for padding in ("mask", "buffers"):
            graphs = build(geometry, cfg.GATES, padding=padding).to(device)
            packed = evaluate_graphs(graphs, task.root_values, cfg.GATES.operators)
            exact, _ = bit_accuracy(packed, task)
            assert float(exact[0]) == 1.0, f"the {name} reference is not an adder"

            logits = predict_full_table(model, graphs)
            targets = patch_targets(packed, all_patches, rows_per_patch)
            correct = ((logits > 0).float() == targets).float().mean()
            mcc, _ = matthews_correlation(confusion_counts(logits, targets))
            per_bit = [float(value) for value in mcc[0]]
            ranks = graphs.output_ranks[0].tolist()

            label = f"{name} ({padding})"
            print(
                f"{label:>18}  {int(graphs.num_live_trunk_nodes[0]):>4}  "
                f"{float(correct):.4f}  {float(mcc.mean()):+.3f}  "
                + " ".join(f"{value:+.2f}" for value in per_bit)
            )

            truth_image = outputs_to_image(packed, height, width)[0]
            predicted_image = logits_to_image(logits, height, width)
            figure = comparison_figure(
                truth_image,
                predicted_image,
                ranks,
                per_bit,
                f"{name} adder, {padding}-padded -- step {step}, "
                f"mean MCC {float(mcc.mean()):+.3f}",
            )
            path = output_dir / f"{name}_{padding}.png"
            figure.savefig(path, dpi=110)
            plt.close(figure)
            print(f"{'':>18}  wrote {path}")


if __name__ == "__main__":
    main()
