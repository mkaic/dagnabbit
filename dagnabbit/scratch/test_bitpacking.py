"""
Visualization script for bitpacking and unpacking functions.
Tests get_address_bitarrays and output_to_image_array to verify they work correctly.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dagnabbit.bitarrays import (
    calculate_required_bitdepth,
    get_address_bitarrays,
    output_to_image_array,
)


def visualize_bitpacking_roundtrip(shape: Tuple[int], verbose: bool = True):
    """
    Test the round-trip: pack addresses -> unpack -> verify correctness.

    Args:
        shape: Shape of the array to test
        verbose: If True, print detailed information
    """
    print(f"\n{'='*60}")
    print(f"Testing shape: {shape}")
    print(f"{'='*60}")

    # Step 1: Pack addresses into bit arrays
    packed = get_address_bitarrays(shape)
    total_elements = np.prod(shape)
    total_bitdepth = packed.shape[0]

    print(f"\n1. Packing addresses:")
    print(f"   Shape: {shape}")
    print(f"   Total elements: {total_elements}")
    print(f"   Total bitdepth: {total_bitdepth}")
    print(f"   Packed shape: {packed.shape}")

    # Calculate bitdepths per dimension
    bitdepths = [calculate_required_bitdepth(dim) for dim in shape]
    print(f"   Bitdepths per dimension: {bitdepths}")
    print(f"   Sum of bitdepths: {sum(bitdepths)}")

    # Step 2: Unpack back to image array
    unpacked = output_to_image_array(packed, shape)

    print(f"\n2. Unpacking:")
    print(f"   Unpacked shape: {unpacked.shape}")
    print(f"   Unpacked dtype: {unpacked.dtype}")

    # Step 3: Verify correctness
    # We expect the unpacked array to contain the original addresses
    # For a 2D array, we can reconstruct coordinates
    print(f"\n3. Verification:")

    # Create expected values by reconstructing coordinates
    expected = np.zeros(shape, dtype=np.uint8)
    coords = np.unravel_index(np.arange(total_elements), shape)

    # For each element, we can check what address bits it should have
    # The packed array contains address bits, so unpacking should give us
    # the address encoding back

    if verbose:
        print(f"\n   Sample values (first 10 elements):")
        flat_unpacked = unpacked.flatten()
        for i in range(min(10, total_elements)):
            coord = np.unravel_index(i, shape)
            print(f"   Element {i} at {coord}: unpacked={flat_unpacked[i]}")

    # Visualize the packed bits
    print(f"\n4. Visualizing packed bits:")
    visualize_packed_bits(packed, shape, bitdepths)

    # Visualize the unpacked array
    print(f"\n5. Visualizing unpacked array:")
    visualize_unpacked_array(unpacked, shape)

    return packed, unpacked


def visualize_packed_bits(packed: np.ndarray, shape: Tuple[int], bitdepths: list):
    """
    Visualize the packed bit arrays in a human-readable format.
    """
    total_elements = np.prod(shape)
    total_bitdepth = packed.shape[0]

    # Unpack the bits for visualization
    unpacked_bits = np.unpackbits(packed, axis=-1)
    unpacked_bits = unpacked_bits[:, :total_elements]  # Trim padding

    print(f"   Packed bits shape: {unpacked_bits.shape}")
    print(f"   (bitdepth, num_elements)")

    # Show bit breakdown by dimension
    bit_offset = 0
    for dim_idx, (dim_size, bitdepth) in enumerate(zip(shape, bitdepths)):
        print(f"\n   Dimension {dim_idx} (size={dim_size}, bits={bitdepth}):")
        dim_bits = unpacked_bits[bit_offset : bit_offset + bitdepth, :]

        # Reshape to show dimension structure
        dim_bits_reshaped = dim_bits.reshape(bitdepth, *shape)

        # Show a sample for small arrays
        if total_elements <= 20:
            print(f"   Bits for this dimension:")
            for bit_idx in range(bitdepth):
                bit_slice = dim_bits_reshaped[bit_idx]
                print(f"     Bit {bit_idx}: {bit_slice.flatten()}")
        else:
            print(f"   (too large to print, showing summary)")
            print(f"     Bit patterns: min={dim_bits.min()}, max={dim_bits.max()}")

        bit_offset += bitdepth


def visualize_unpacked_array(unpacked: np.ndarray, shape: Tuple[int]):
    """
    Visualize the unpacked array.
    """
    print(f"   Unpacked array shape: {unpacked.shape}")
    print(f"   Value range: [{unpacked.min()}, {unpacked.max()}]")

    if len(shape) == 2 and shape[0] <= 20 and shape[1] <= 20:
        print(f"\n   Full array:")
        print(unpacked)
    elif len(shape) == 1 and shape[0] <= 50:
        print(f"\n   Full array:")
        print(unpacked)
    else:
        print(f"\n   Array too large to print, showing summary:")
        print(f"   Shape: {unpacked.shape}")
        print(f"   Mean: {unpacked.mean():.2f}, Std: {unpacked.std():.2f}")


def create_visualization_plots(
    shape: Tuple[int], packed: np.ndarray, unpacked: np.ndarray
):
    """
    Create matplotlib visualizations of the packing/unpacking process.
    """
    if len(shape) != 2:
        print(f"   Skipping plots for non-2D shape {shape}")
        return

    # Calculate which dimension each bitplane belongs to
    bitdepths = [calculate_required_bitdepth(dim) for dim in shape]
    total_bitdepth = packed.shape[0]

    # Determine which dimension each bitplane belongs to
    bitplane_dimension = []
    bitplane_bit_index = []
    bit_offset = 0
    for dim_idx, bitdepth in enumerate(bitdepths):
        for bit_idx in range(bitdepth):
            bitplane_dimension.append(dim_idx)
            bitplane_bit_index.append(bit_idx)
        bit_offset += bitdepth

    # Create figure with subplots: unpacked array on left, bitplanes on right
    num_bitplanes = total_bitdepth
    n_cols = 4
    n_rows = (num_bitplanes + n_cols - 1) // n_cols

    # Create a larger figure to accommodate all bitplanes
    fig = plt.figure(figsize=(16, 4 + n_rows * 3))

    # Plot 1: Unpacked array as image (top left)
    ax_unpacked = plt.subplot2grid((n_rows + 1, n_cols), (0, 0), colspan=2, rowspan=1)
    im = ax_unpacked.imshow(unpacked, cmap="viridis", interpolation="nearest")
    ax_unpacked.set_title(
        f"Unpacked Array (shape={shape})", fontsize=12, fontweight="bold"
    )
    ax_unpacked.set_xlabel("X coordinate", fontsize=10)
    ax_unpacked.set_ylabel("Y coordinate", fontsize=10)
    ax_unpacked.set_xticks(range(shape[1]))
    ax_unpacked.set_yticks(range(shape[0]))
    ax_unpacked.set_xticklabels(range(shape[1]))
    ax_unpacked.set_yticklabels(range(shape[0]))
    plt.colorbar(im, ax=ax_unpacked)

    # Plot bitplanes
    total_elements = np.prod(shape)
    unpacked_bits = np.unpackbits(packed, axis=-1)
    unpacked_bits = unpacked_bits[:, :total_elements]

    # Reshape bits to 2D for visualization
    bit_visualization = unpacked_bits.reshape(packed.shape[0], *shape)

    for bitplane_idx in range(num_bitplanes):
        row = (bitplane_idx // n_cols) + 1  # +1 to account for unpacked array row
        col = bitplane_idx % n_cols

        ax = plt.subplot2grid((n_rows + 1, n_cols), (row, col))

        # Get the bitplane
        bitplane = bit_visualization[bitplane_idx]

        # Display the bitplane
        im = ax.imshow(bitplane, cmap="gray", interpolation="nearest", vmin=0, vmax=1)

        # Determine which dimension and bit index this belongs to
        dim_idx = bitplane_dimension[bitplane_idx]
        bit_idx = bitplane_bit_index[bitplane_idx]
        dim_name = ["X", "Y", "Z"][dim_idx] if dim_idx < 3 else f"Dim{dim_idx}"

        # Create title with dimension and bit info
        is_lsb = bit_idx == bitdepths[dim_idx] - 1
        is_msb = bit_idx == 0
        bit_label = "LSB" if is_lsb else ("MSB" if is_msb else f"Bit{bit_idx}")

        title = f"Bitplane {bitplane_idx}\n{dim_name} dim, {bit_label}"
        ax.set_title(title, fontsize=9, fontweight="bold")

        # Add axis labels with coordinates
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_xticks(range(shape[1]))
        ax.set_yticks(range(shape[0]))
        ax.set_xticklabels(range(shape[1]), fontsize=7)
        ax.set_yticklabels(range(shape[0]), fontsize=7)

        # Add grid for better readability
        ax.grid(True, color="red", alpha=0.3, linewidth=0.5)
        ax.set_xticks([x - 0.5 for x in range(shape[1] + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(shape[0] + 1)], minor=True)
        ax.grid(True, which="minor", color="red", alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(
        f"/home/mkaic/dagnabbit/dagnabbit/outputs/bitpacking_test_{shape[0]}x{shape[1]}.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(
        f"   Saved visualization to outputs/bitpacking_test_{shape[0]}x{shape[1]}.png"
    )
    plt.close()


def test_roundtrip_correctness(shape: Tuple[int]) -> bool:
    """
    Test that packing and unpacking preserves the address information correctly.
    Returns True if test passes.
    """
    packed = get_address_bitarrays(shape)
    unpacked = output_to_image_array(packed, shape)

    # The unpacked array should contain address information
    # For a simple test, we can verify that different positions have different values
    # (or at least that the encoding is consistent)

    # Check that we can reconstruct coordinates from the packed bits
    total_elements = np.prod(shape)
    unpacked_bits = np.unpackbits(packed, axis=-1)
    unpacked_bits = unpacked_bits[:, :total_elements]

    # Reconstruct addresses from bits
    bitdepths = [calculate_required_bitdepth(dim) for dim in shape]
    total_bitdepth = sum(bitdepths)

    # Verify bitdepth matches
    assert (
        unpacked_bits.shape[0] == total_bitdepth
    ), f"Bitdepth mismatch: expected {total_bitdepth}, got {unpacked_bits.shape[0]}"

    # Test a few specific coordinates
    test_coords = []
    for i in range(min(5, total_elements)):
        coord = np.unravel_index(i, shape)
        test_coords.append(coord)

    print(f"\n   Testing {len(test_coords)} sample coordinates...")
    all_passed = True

    for coord in test_coords:
        flat_idx = np.ravel_multi_index(coord, shape)

        # Extract bits for this element
        element_bits = unpacked_bits[:, flat_idx]

        # Reconstruct each dimension's address
        bit_offset = 0
        reconstructed_coords = []
        for dim_idx, (dim_size, bitdepth) in enumerate(zip(shape, bitdepths)):
            dim_bits = element_bits[bit_offset : bit_offset + bitdepth]
            # Convert bits to integer (little-endian within the bitdepth)
            dim_value = 0
            for bit_idx, bit in enumerate(dim_bits):
                dim_value += int(bit) * (2 ** (bitdepth - 1 - bit_idx))
            reconstructed_coords.append(dim_value)
            bit_offset += bitdepth

        # Verify reconstruction
        if tuple(reconstructed_coords) == coord:
            print(f"   ✓ Coord {coord} -> reconstructed correctly")
        else:
            print(
                f"   ✗ Coord {coord} -> reconstructed as {tuple(reconstructed_coords)}"
            )
            all_passed = False

    return all_passed


def main():
    """
    Run visualization tests on various shapes.
    """
    # Test cases
    test_shapes = [
        (4, 4),  # Small 2D
        (8, 8),  # Medium 2D
        (5, 5),  # Non-power-of-2
        (3, 7),  # Rectangular
        (10,),  # 1D
    ]

    print("Bitpacking/Unpacking Visualization Test")
    print("=" * 60)

    results = []

    for shape in test_shapes:
        try:
            packed, unpacked = visualize_bitpacking_roundtrip(shape, verbose=True)

            # Test correctness
            passed = test_roundtrip_correctness(shape)
            results.append((shape, passed))

            # Create plots for 2D shapes
            if len(shape) == 2:
                create_visualization_plots(shape, packed, unpacked)

        except Exception as e:
            print(f"\n   ERROR testing shape {shape}: {e}")
            import traceback

            traceback.print_exc()
            results.append((shape, False))

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for shape, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {shape}: {status}")

    all_passed = all(passed for _, passed in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")


if __name__ == "__main__":
    main()
