"""Bitpacked evaluation of node-index circuits.

A circuit is run by pushing every row of its truth table through the DAG at
once: each node's value is a packed bit vector with one bit per row, so one
``uint8`` word carries 8 rows and a full 2^16-row table fits in 8192 bytes per
node.

Row ordering
------------
Row ``r`` *is* the integer ``r``, with root node 0 holding the most significant
input bit. So for the 16-input geometry, row ``r`` has ``a = r >> 8`` on roots
0-7 and ``b = r & 255`` on roots 8-15, and folding the rows into a 256x256 grid
gives ``a`` down one axis and ``b`` along the other. One enumeration is used
everywhere -- random graphs during training, the reference adder at probe time --
because the model's truth-table patch queries are learned per row block and
would otherwise mean different things in the two settings.

Padding bits
------------
When ``num_rows`` is not a multiple of 8, the final word holds padding bits past
the end of the table. These are *deliberately left to rot*: every operation here
is bitwise-elementwise, so output bit i depends only on input bit i of its
parents. Padding bits therefore evolve independently of real rows and can never
contaminate them, no matter how deep the circuit. Masking once at scoring time
is sufficient, and is what :func:`bit_accuracy` does.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from dagnabbit.tasks.logic_gates.operators import GATE_OPERATORS, GateOperator

BITS_PER_WORD = 8


def make_valid_bit_mask(num_rows: int, num_words: int) -> Tensor:
    """A ``[num_words]`` uint8 mask whose set bits are exactly the real rows.

    Uses ``np.packbits``' default big-endian bit order, matching how everything
    here is packed, so bit i of word w corresponds to row ``w * 8 + i``.
    """
    if num_rows > num_words * BITS_PER_WORD:
        raise ValueError(
            f"{num_rows} rows do not fit in {num_words} words "
            f"({num_words * BITS_PER_WORD} bits)"
        )
    bits = np.zeros(num_words * BITS_PER_WORD, dtype=np.uint8)
    bits[:num_rows] = 1
    return torch.from_numpy(np.packbits(bits))


def unpack_bits(words: Tensor) -> Tensor:
    """``[..., W]`` uint8 -> ``[..., W * 8]`` uint8 of 0/1, big-endian.

    Matches ``np.packbits``' default bit order. Done with shifts rather than a
    numpy round trip so this stays on whatever device the batch already lives on.
    """
    if words.dtype != torch.uint8:
        raise TypeError(f"expected uint8, got {words.dtype}")
    shifts = torch.arange(
        BITS_PER_WORD - 1, -1, -1, dtype=torch.uint8, device=words.device
    )
    return ((words.unsqueeze(-1) >> shifts) & 1).flatten(start_dim=-2)


def exhaustive_root_values(num_root_nodes: int) -> Tensor:
    """Every input combination, packed: ``[R, num_words]`` uint8.

    Root ``i`` carries bit ``R - 1 - i`` of the row index, so row ``r`` presents
    the integer ``r`` with root 0 as the most significant bit.
    """
    num_rows = 1 << num_root_nodes
    rows = np.arange(num_rows, dtype=np.uint32)
    shifts = np.arange(num_root_nodes - 1, -1, -1, dtype=np.uint32)
    bits = ((rows[None, :] >> shifts[:, None]) & 1).astype(np.uint8)
    return torch.from_numpy(np.packbits(bits, axis=-1))


@dataclass(frozen=True)
class BitpackedTask:
    """A *target* behaviour in packed form, resident on one device.

    ``root_values[i]`` is the input column driving root node ``i`` and
    ``target_values[j]`` is the desired output of output node ``j``. Both index
    by node slot, matching the node layout: roots occupy positions
    ``[0, R)`` and outputs the final ``num_output_nodes`` positions.

    Only Phase-1 inverse design and the reference probes need a task. Simulator
    training runs circuits against :func:`exhaustive_root_values` and takes
    whatever they compute as the label, so it never builds one.
    """

    root_values: Tensor  # [num_root_nodes, num_words] uint8
    target_values: Tensor  # [num_output_nodes, num_words] uint8
    num_rows: int
    valid_bit_mask: Tensor  # [num_words] uint8

    def __post_init__(self) -> None:
        if self.root_values.dtype != torch.uint8:
            raise TypeError(f"root_values must be uint8, got {self.root_values.dtype}")
        if self.target_values.dtype != torch.uint8:
            raise TypeError(
                f"target_values must be uint8, got {self.target_values.dtype}"
            )
        if self.root_values.shape[1] != self.target_values.shape[1]:
            raise ValueError(
                f"root_values has {self.root_values.shape[1]} words but "
                f"target_values has {self.target_values.shape[1]}"
            )
        if self.valid_bit_mask.shape != (self.num_words,):
            raise ValueError(
                f"valid_bit_mask must be [{self.num_words}], got "
                f"{tuple(self.valid_bit_mask.shape)}"
            )

    @property
    def num_words(self) -> int:
        return self.root_values.shape[1]

    @property
    def num_root_nodes(self) -> int:
        return self.root_values.shape[0]

    @property
    def num_output_nodes(self) -> int:
        return self.target_values.shape[0]

    def to(self, device: torch.device | str) -> "BitpackedTask":
        return BitpackedTask(
            root_values=self.root_values.to(device),
            target_values=self.target_values.to(device),
            num_rows=self.num_rows,
            valid_bit_mask=self.valid_bit_mask.to(device),
        )


def adder_task(device: torch.device | str = "cpu") -> BitpackedTask:
    """The 8-bit adder: 16 input bits -> 8 sum bits, all 65536 rows.

    Roots 0-7 carry the bits of ``a`` (most significant first), roots 8-15 the
    bits of ``b``, and output j the corresponding bit of ``(a + b) mod 256``.
    Derived from the row index rather than a meshgrid so it lands on the same
    enumeration as :func:`exhaustive_root_values`.
    """
    num_root_nodes = 16
    root_values = exhaustive_root_values(num_root_nodes)
    num_rows = 1 << num_root_nodes

    rows = np.arange(num_rows, dtype=np.uint32)
    # uint8 addition wraps, which is the mod 256 the task wants.
    sums = (rows >> 8).astype(np.uint8) + (rows & 0xFF).astype(np.uint8)
    shifts = np.arange(7, -1, -1, dtype=np.uint8)
    target_bits = ((sums[None, :] >> shifts[:, None]) & 1).astype(np.uint8)

    return BitpackedTask(
        root_values=root_values,
        target_values=torch.from_numpy(np.packbits(target_bits, axis=-1)),
        num_rows=num_rows,
        valid_bit_mask=make_valid_bit_mask(num_rows, root_values.shape[1]),
    ).to(device)


@torch.no_grad()
def evaluate_choices(
    trunk_types: Tensor,
    parent_indices: Tensor,
    root_values: Tensor,
    num_output_nodes: int,
    trunk_node_in_degrees: Sequence[int],
    gate_operators: Sequence[GateOperator] = GATE_OPERATORS,
) -> Tensor:
    """Run a batch of circuits, on-device, straight from graph tensors.

    ``trunk_types`` is ``[B, T]`` trunk class ids and ``parent_indices`` is
    ``[B, N, S]`` parent node indices -- exactly what
    :class:`~dagnabbit.dag.graphs.GraphBatch` holds, and exactly what a
    generator's straight-through argmax produces. Returns packed outputs
    ``[B, num_output_nodes, num_words]`` uint8, in output-slot order.

    One sequential sweep over trunk positions, each a batched gather plus gate
    over all B circuits. Positions cannot be batched by rank without per-graph
    topology, but a position step is a handful of kernels over ``[B, num_words]``,
    so the sweep is launch-cheap and bandwidth-bound. Routing through per-graph
    Python objects instead cost ~0.5 ms per graph with the device idle throughout.
    """
    if trunk_types.ndim != 2 or parent_indices.ndim != 3:
        raise ValueError(
            f"expected [B, T] types and [B, N, S] positions; got "
            f"{tuple(trunk_types.shape)} and {tuple(parent_indices.shape)}"
        )
    device = root_values.device
    trunk_types = trunk_types.to(device)
    parent_indices = parent_indices.to(device)

    batch_size, num_trunk_nodes = trunk_types.shape
    num_root_nodes, num_words = root_values.shape
    output_start = num_root_nodes + num_trunk_nodes
    num_nodes = output_start + num_output_nodes
    if parent_indices.shape[:2] != (batch_size, num_nodes):
        raise ValueError(
            f"parent_indices is {tuple(parent_indices.shape)}, expected "
            f"[{batch_size}, {num_nodes}, S]"
        )
    if len(gate_operators) < len(trunk_node_in_degrees):
        raise ValueError(
            f"{len(gate_operators)} gate operators for "
            f"{len(trunk_node_in_degrees)} trunk node types"
        )
    max_in_degree = max(trunk_node_in_degrees)
    if parent_indices.shape[2] < max_in_degree:
        raise ValueError(
            f"parent_indices has {parent_indices.shape[2]} slots, but the "
            f"widest trunk type needs {max_in_degree}"
        )

    # Legality, checked once and vectorized: every slot read below points
    # strictly earlier and never at an output. A garbage index would otherwise
    # silently alias a real node rather than raising.
    positions = torch.arange(num_nodes, device=device)
    read_slots = parent_indices[:, num_root_nodes:, :max_in_degree]
    if not bool(
        (
            (read_slots < positions[num_root_nodes:, None])
            & (read_slots < output_start)
        ).all()
    ):
        raise ValueError(
            "parent_indices contains an index at or after its consumer "
            "(or pointing into the output block)"
        )

    # MPS rejects gather sources above 2^31 elements; chunk the batch there.
    chunk = batch_size
    if device.type == "mps":
        chunk = max(1, (2**31 - 1) // (output_start * num_words))

    outputs = []
    for start in range(0, batch_size, chunk):
        chunk_types = trunk_types[start : start + chunk]
        chunk_positions = parent_indices[start : start + chunk]
        rows = chunk_types.shape[0]

        # Producer values only: outputs are leaves and are never gathered from.
        buffer = torch.empty(
            rows, output_start, num_words, dtype=torch.uint8, device=device
        )
        buffer[:, :num_root_nodes] = root_values

        for position in range(num_root_nodes, output_start):
            slot_indices = chunk_positions[:, position, :max_in_degree]
            parents = buffer.gather(
                1, slot_indices.unsqueeze(-1).expand(rows, max_in_degree, num_words)
            )
            value = None
            for node_type, in_degree in enumerate(trunk_node_in_degrees):
                candidate = gate_operators[node_type](parents[:, :in_degree])
                if value is None:
                    value = candidate
                else:
                    is_type = chunk_types[:, position - num_root_nodes] == node_type
                    value = torch.where(is_type[:, None], candidate, value)
            buffer[:, position] = value

        # Output nodes are in-degree-1 pass-throughs of their slot-0 parent.
        output_indices = chunk_positions[:, output_start:, 0]
        outputs.append(
            buffer.gather(
                1,
                output_indices.unsqueeze(-1).expand(rows, num_output_nodes, num_words),
            )
        )

    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


def evaluate_graphs(
    graphs,
    root_values: Tensor,
    gate_operators: Sequence[GateOperator] = GATE_OPERATORS,
) -> Tensor:
    """:func:`evaluate_choices` for a :class:`GraphBatch`.

    ``gate_operators`` is explicit only so tests and experiments can score a
    geometry against a gate set other than the configured one; training always
    takes the default.
    """
    geometry = graphs.geometry
    return evaluate_choices(
        graphs.trunk_types,
        graphs.parent_indices,
        root_values,
        geometry.num_output_nodes,
        geometry.trunk_node_in_degrees,
        gate_operators,
    )


def popcount(words: Tensor) -> Tensor:
    """Per-byte set-bit count of a uint8 tensor, returned as uint8 (0-8).

    SWAR rather than a lookup table: a table index would widen the tensor to
    int64 first, which for a full-truth-table batch is gigabytes of temporary.
    Every intermediate here stays uint8.
    """
    if words.dtype != torch.uint8:
        raise TypeError(f"popcount expects uint8, got {words.dtype}")
    counts = words - ((words >> 1) & 0x55)
    counts = (counts & 0x33) + ((counts >> 2) & 0x33)
    return (counts + (counts >> 4)) & 0x0F


def bit_accuracy(predicted: Tensor, task: BitpackedTask) -> tuple[Tensor, Tensor]:
    """Score packed outputs against a task's targets.

    Returns ``(overall, per_output)``: overall fitness per graph as ``[B]`` in
    [0, 1], and per-output-bit accuracy as ``[B, num_output_nodes]``. Only the
    first ``task.num_rows`` bits are counted, so padding cannot influence it.

    Mismatches are counted exactly, as int64, on whatever device the outputs
    live on. The ratio is then taken in float64 on the CPU: the count tensor is
    only ``[B, num_output_nodes]``, so the transfer is free next to the
    evaluation, and it keeps the score exact for tables larger than float32's
    2^24 integer range while sidestepping MPS, which has no float64 at all.
    """
    if predicted.dtype != torch.uint8:
        raise TypeError(f"predicted must be uint8, got {predicted.dtype}")
    if predicted.shape[1:] != task.target_values.shape:
        raise ValueError(
            f"predicted has shape {tuple(predicted.shape)}, incompatible with "
            f"targets {tuple(task.target_values.shape)}"
        )

    mismatched = (predicted ^ task.target_values) & task.valid_bit_mask
    mismatches = popcount(mismatched).sum(dim=-1, dtype=torch.int64).cpu()

    per_output = 1.0 - mismatches.double() / task.num_rows
    overall = 1.0 - mismatches.sum(dim=-1).double() / (
        task.num_rows * task.num_output_nodes
    )
    return overall, per_output
