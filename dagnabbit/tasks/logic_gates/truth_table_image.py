"""Truth tables as multi-channel images, for conditioning a proposal network.

A circuit's behaviour is a function of its 16 input bits. Splitting those into
two 8-bit halves lays the whole truth table out as a 256x256 grid with one
channel per output bit: pixel ``(u, v)`` holds what the circuit produced for
that pair of inputs. Position *is* the input, so a network reading this image
needs no separate addressing scheme -- its position embedding already is one.

Axis ordering
-------------
Indexing each axis by Gray code, ``gray(i) = i ^ (i >> 1)``, makes adjacent
pixels differ in exactly one input bit -- a Karnaugh map, and the obviously
correct prior for a *boolean* function, whose natural metric is Hamming
distance.

It is nonetheless the wrong default here, and
``dagnabbit.scripts.render_truth_table_images`` shows why: the targets in this
package are **arithmetic**. Every output bit of ``a + b`` is a function of the
integer sum, so in plain binary ordering the target's bit planes are clean
diagonal stripes -- constant along anti-diagonals -- while Gray ordering
scrambles that into a Sierpinski-like fractal. Random graphs, meanwhile, come
out as axis-aligned bands in *either* layout, so the ordering only moves the
target. Binary therefore leaves the smaller gap between what a conditioning
network trains on and what it is asked for at inference.

Hence ``gray=False`` by default. The flag stays because the reasoning inverts
for a genuinely boolean target, where Hamming locality is the real structure.

Only :func:`image_dimensions` knows anything about the 8-bit adder's shape;
everything else takes the grid size as an argument.
"""

import torch
from torch import Tensor

BITS_PER_WORD = 8


def gray_permutation(size: int, device: torch.device | str = "cpu") -> Tensor:
    """``[size]`` tensor whose ``i``-th entry is the value shown at index ``i``.

    Consecutive entries differ in exactly one bit, which is the entire point.
    ``size`` must be a power of two, or the code wraps and stops being a
    bijection.
    """
    if size & (size - 1) != 0:
        raise ValueError(f"gray coding needs a power-of-two axis; got {size}")
    indices = torch.arange(size, dtype=torch.long, device=device)
    return indices ^ (indices >> 1)


def unpack_bits(words: Tensor) -> Tensor:
    """``[..., W]`` uint8 -> ``[..., W * 8]`` uint8 of 0/1, big-endian.

    Matches ``np.packbits``' default bit order, which is how every truth table
    in :mod:`.bitarrays` is packed. Done with shifts rather than a numpy round
    trip so this stays on whatever device the batch already lives on.
    """
    if words.dtype != torch.uint8:
        raise TypeError(f"expected uint8, got {words.dtype}")
    shifts = torch.arange(
        BITS_PER_WORD - 1,
        -1,
        -1,
        dtype=torch.uint8,
        device=words.device,
    )
    bits = (words.unsqueeze(-1) >> shifts) & 1
    return bits.flatten(start_dim=-2)


def outputs_to_image(
    packed: Tensor,
    height: int,
    width: int,
    gray: bool = False,
) -> Tensor:
    """``[B, C, W]`` packed output columns -> ``[B, C, H, W]`` uint8 bit planes.

    ``packed`` is what :func:`~dagnabbit.tasks.logic_gates.evaluate.evaluate_graphs`
    returns, or a task's ``target_values`` with a batch axis added. Rows are
    consumed in truth-table order and folded into ``height`` rows of ``width``
    columns, matching how :func:`.bitarrays.get_8bit_adder_truth_table`
    flattens its ``(256, 256)`` meshgrid.
    """
    if packed.ndim != 3:
        raise ValueError("packed must have shape [B, C, num_words]")
    num_rows = height * width
    bits = unpack_bits(packed)
    if bits.shape[-1] < num_rows:
        raise ValueError(
            f"{packed.shape[-1]} words hold {bits.shape[-1]} rows, "
            f"too few for a {height}x{width} grid"
        )
    image = bits[..., :num_rows].unflatten(-1, (height, width))

    if gray:
        rows = gray_permutation(height, image.device)
        columns = gray_permutation(width, image.device)
        image = image[..., rows[:, None], columns[None, :]]
    return image


def image_dimensions(num_root_nodes: int) -> tuple[int, int]:
    """Grid shape for a truth table over ``num_root_nodes`` input bits.

    Splits the input bits evenly between the two axes. The 16-input tasks in
    this package give the 256x256 grid the rest of the module assumes.
    """
    if num_root_nodes % 2 != 0:
        raise ValueError(
            f"need an even number of input bits to split across two axes; "
            f"got {num_root_nodes}"
        )
    half = num_root_nodes // 2
    return 1 << half, 1 << half


def task_target_image(task, gray: bool = False) -> Tensor:
    """A task's *desired* outputs as a ``[C, H, W]`` image.

    The same transform the training images go through, so what the proposal
    network is asked for at inference time is laid out exactly like what it was
    trained on.
    """
    height, width = image_dimensions(task.root_values.shape[0])
    return outputs_to_image(
        task.target_values.unsqueeze(0),
        height,
        width,
        gray=gray,
    )[0]
