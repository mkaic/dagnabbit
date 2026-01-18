"""
Visualize the bitpacking process for the 8-bit adder truth table.
This script generates images showing each stage of the packing/unpacking process.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dagnabbit.bitarrays import get_8bit_adder_truth_table

# Create output directory for images
OUTPUT_DIR = Path(__file__).parent / "bitpacking_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_figure(fig, name: str):
    """Save figure and close it."""
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_grid(grid: np.ndarray):
    """
    Visualize the input grid (256, 256, 2).
    Shows input_a values and input_b values as separate heatmaps.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Input Grid (256×256×2)\nEach cell contains [a, b] where sum = a + b", fontsize=14)

    # Input A (varies along columns due to meshgrid)
    im0 = axes[0].imshow(grid[:, :, 0], cmap="viridis", aspect="equal")
    axes[0].set_title("Input A values\ngrid[:, :, 0]")
    axes[0].set_xlabel("j (column)")
    axes[0].set_ylabel("i (row)")
    plt.colorbar(im0, ax=axes[0], label="Value (0-255)")

    # Input B (varies along rows due to meshgrid)
    im1 = axes[1].imshow(grid[:, :, 1], cmap="plasma", aspect="equal")
    axes[1].set_title("Input B values\ngrid[:, :, 1]")
    axes[1].set_xlabel("j (column)")
    axes[1].set_ylabel("i (row)")
    plt.colorbar(im1, ax=axes[1], label="Value (0-255)")

    save_figure(fig, "01_input_grid")


def visualize_sums(sums: np.ndarray):
    """
    Visualize the sum output (256, 256).
    Shows the raw uint8 sum values (with overflow wrapping).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sum Output (256×256)\nsum = (a + b) mod 256 (uint8 overflow)", fontsize=14)

    # Raw sum values
    im0 = axes[0].imshow(sums.squeeze(), cmap="magma", aspect="equal")
    axes[0].set_title("Sum values")
    axes[0].set_xlabel("j (column) = input_a")
    axes[0].set_ylabel("i (row) = input_b")
    plt.colorbar(im0, ax=axes[0], label="Sum (0-255)")

    # Show the diagonal pattern from overflow
    # Highlight where overflow occurs (a + b >= 256)
    overflow_mask = (np.arange(256)[:, None] + np.arange(256)[None, :]) >= 256
    im1 = axes[1].imshow(overflow_mask.astype(float), cmap="RdYlGn_r", aspect="equal")
    axes[1].set_title("Overflow regions\n(where a + b ≥ 256)")
    axes[1].set_xlabel("j (column) = input_a")
    axes[1].set_ylabel("i (row) = input_b")
    plt.colorbar(im1, ax=axes[1], label="Overflow (1=yes)")

    save_figure(fig, "02_sums")


def visualize_unpacked_inputs(unpacked_inputs: np.ndarray):
    """
    Visualize the unpacked input bits (16, 256, 256) before reshaping.
    Each of the 16 planes represents one bit position.
    """
    # Show all 16 bit planes in a 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle(
        "Unpacked Input Bit Planes (16×256×256)\n"
        "Bits 0-7: Input A (MSB→LSB) | Bits 8-15: Input B (MSB→LSB)",
        fontsize=14,
    )

    for bit_idx in range(16):
        ax = axes[bit_idx // 4, bit_idx % 4]
        plane = unpacked_inputs[bit_idx]

        ax.imshow(plane, cmap="binary", aspect="equal", vmin=0, vmax=1)

        if bit_idx < 8:
            label = f"Bit {bit_idx}\nInput A bit {7-bit_idx}"
        else:
            label = f"Bit {bit_idx}\nInput B bit {15-bit_idx}"

        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_figure(fig, "03_unpacked_inputs_bitplanes")


def visualize_unpacked_sums(unpacked_sums: np.ndarray):
    """
    Visualize the unpacked sum bits (8, 256, 256) before reshaping.
    Each of the 8 planes represents one bit position of the output.
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(
        "Unpacked Sum Bit Planes (8×256×256)\n"
        "Each plane is one bit of the 8-bit sum output (MSB→LSB)",
        fontsize=14,
    )

    for bit_idx in range(8):
        ax = axes[bit_idx // 4, bit_idx % 4]
        plane = unpacked_sums[bit_idx]

        ax.imshow(plane, cmap="binary", aspect="equal", vmin=0, vmax=1)
        ax.set_title(f"Bit {bit_idx} (weight={2**(7-bit_idx)})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_figure(fig, "04_unpacked_sums_bitplanes")


def visualize_packed_data(packed_inputs: np.ndarray, packed_sums: np.ndarray):
    """
    Visualize the final packed bit arrays.
    packed_inputs: (16, 8192)
    packed_sums: (8, 8192)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Packed Bit Arrays (final truth table format)", fontsize=14)

    # Packed inputs
    im0 = axes[0].imshow(packed_inputs, cmap="viridis", aspect="auto")
    axes[0].set_title(f"Packed Inputs: shape {packed_inputs.shape}\n16 bitvectors × 8192 bytes = 65536 samples")
    axes[0].set_xlabel("Byte index (0-8191)")
    axes[0].set_ylabel("Bit index (0-15)")
    plt.colorbar(im0, ax=axes[0], label="Byte value (0-255)")

    # Packed sums
    im1 = axes[1].imshow(packed_sums, cmap="plasma", aspect="auto")
    axes[1].set_title(f"Packed Sums: shape {packed_sums.shape}\n8 bitvectors × 8192 bytes = 65536 samples")
    axes[1].set_xlabel("Byte index (0-8191)")
    axes[1].set_ylabel("Bit index (0-7)")
    plt.colorbar(im1, ax=axes[1], label="Byte value (0-255)")

    plt.tight_layout()
    save_figure(fig, "05_packed_arrays")


def visualize_sample_correspondence(grid, sums, unpacked_inputs, unpacked_sums):
    """
    Show that sample ordering is consistent between inputs and outputs.
    Pick a few specific samples and verify the bits match.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Sample Correspondence Check\nVerifying input/output alignment", fontsize=14)

    # Show sample index mapping
    ax = axes[0, 0]
    sample_indices = np.arange(256 * 256).reshape(256, 256)
    im = ax.imshow(sample_indices, cmap="turbo", aspect="equal")
    ax.set_title("Sample index = i×256 + j\n(row-major flattening)")
    ax.set_xlabel("j (column)")
    ax.set_ylabel("i (row)")
    plt.colorbar(im, ax=ax, label="Sample index")

    # Test specific samples
    ax = axes[0, 1]
    test_cases = [
        (0, 0),
        (0, 255),
        (255, 0),
        (255, 255),
        (128, 64),
        (100, 200),
    ]

    text_lines = ["Sample verification (i, j) → (a, b) → sum:\n"]
    for i, j in test_cases:
        sample_idx = i * 256 + j
        a = grid[i, j, 0]
        b = grid[i, j, 1]
        s = sums[i, j, 0]
        expected_sum = (int(a) + int(b)) % 256

        # Check bits match
        input_bits = unpacked_inputs[:, sample_idx]
        sum_bits = unpacked_sums[:, sample_idx]

        a_reconstructed = sum(input_bits[k] * (2 ** (7 - k)) for k in range(8))
        b_reconstructed = sum(input_bits[k + 8] * (2 ** (7 - k)) for k in range(8))
        s_reconstructed = sum(sum_bits[k] * (2 ** (7 - k)) for k in range(8))

        status = "✓" if (a == a_reconstructed and b == b_reconstructed and s == s_reconstructed) else "✗"
        text_lines.append(
            f"{status} ({i:3d},{j:3d}): a={a:3d}, b={b:3d}, sum={s:3d} "
            f"(reconstructed: a={a_reconstructed}, b={b_reconstructed}, s={s_reconstructed})"
        )

    ax.text(0.05, 0.95, "\n".join(text_lines), transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Sample verification")

    # Show bit patterns for one example
    ax = axes[1, 0]
    i, j = 100, 200
    sample_idx = i * 256 + j
    input_bits = unpacked_inputs[:, sample_idx]
    a, b = grid[i, j, 0], grid[i, j, 1]

    ax.barh(range(16), input_bits, color=["tab:blue"] * 8 + ["tab:orange"] * 8)
    ax.set_yticks(range(16))
    ax.set_yticklabels([f"A bit {7-k}" for k in range(8)] + [f"B bit {7-k}" for k in range(8)])
    ax.set_xlabel("Bit value (0 or 1)")
    ax.set_title(f"Input bits for sample ({i}, {j})\na={a} (0b{a:08b}), b={b} (0b{b:08b})")
    ax.invert_yaxis()

    # Sum bits for same example
    ax = axes[1, 1]
    sum_bits = unpacked_sums[:, sample_idx]
    s = sums[i, j, 0]

    ax.barh(range(8), sum_bits, color="tab:green")
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"Sum bit {7-k}" for k in range(8)])
    ax.set_xlabel("Bit value (0 or 1)")
    ax.set_title(f"Sum bits for sample ({i}, {j})\nsum={s} (0b{s:08b})")
    ax.invert_yaxis()

    plt.tight_layout()
    save_figure(fig, "06_sample_correspondence")


def main():
    print(f"Saving visualizations to: {OUTPUT_DIR}\n")

    # Get all intermediate values from the actual function
    data = get_8bit_adder_truth_table(return_intermediates=True)

    grid = data["grid"]
    sums = data["sums"]
    unpacked_inputs_2d = data["unpacked_inputs_2d"]
    unpacked_sums_2d = data["unpacked_sums_2d"]
    unpacked_inputs = data["unpacked_inputs"]
    unpacked_sums = data["unpacked_sums"]
    packed_inputs = data["packed_inputs"]
    packed_sums = data["packed_sums"]

    print(f"grid shape: {grid.shape}")
    print(f"sums shape: {sums.shape}")
    print(f"unpacked_inputs_2d shape: {unpacked_inputs_2d.shape}")
    print(f"unpacked_sums_2d shape: {unpacked_sums_2d.shape}")
    print(f"unpacked_inputs shape: {unpacked_inputs.shape}")
    print(f"unpacked_sums shape: {unpacked_sums.shape}")
    print(f"packed_inputs shape: {packed_inputs.shape}")
    print(f"packed_sums shape: {packed_sums.shape}")

    print("\nGenerating visualizations...")

    # Generate all visualizations
    visualize_grid(grid)
    visualize_sums(sums)
    visualize_unpacked_inputs(unpacked_inputs_2d)
    visualize_unpacked_sums(unpacked_sums_2d)
    visualize_packed_data(packed_inputs, packed_sums)
    visualize_sample_correspondence(grid, sums, unpacked_inputs, unpacked_sums)

    print(f"\nDone! All images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
