"""Bitpacked evaluation of :class:`FixedInDegreeDAGDescription` circuits.

A circuit is scored against a truth table by pushing every row of that table
through the DAG at once: each node's value is a packed bit vector with one bit
per truth-table row, so one ``uint8`` word carries 8 rows and the full 2^16-row
8-bit adder table fits in 8192 bytes per node.

Evaluation is batched two ways at once, mirroring
:meth:`DagnabbitAutoEncoder.evaluate_graph_batch`:

* **across nodes**, by topological rank -- every node at the same rank is
  evaluated in one vectorized step, since none of them can depend on another;
* **across graphs**, by concatenating each graph's rank metadata into flat
  index tensors, so a batch of B circuits costs one pass over ranks rather
  than B passes.

Padding bits
------------
When ``num_rows`` is not a multiple of 8, the final word holds padding bits
past the end of the truth table. These are *deliberately left to rot*: every
operation here (AND, OR, NOT) is bitwise-elementwise, so output bit i depends
only on input bit i of its parents. Padding bits therefore evolve entirely
independently of real rows and can never contaminate them, no matter how deep
the circuit. Masking once at scoring time is sufficient, and is what
:func:`bit_accuracy` does -- there is no need to re-mask after all 128 gates.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from dagnabbit.dag.description import FixedInDegreeDAGDescription
from dagnabbit.tasks.logic_gates.bitarrays import get_8bit_adder_truth_table
from dagnabbit.tasks.logic_gates.operators import GATE_OPERATORS, GateOperator

BITS_PER_WORD = 8


def make_valid_bit_mask(num_rows: int, num_words: int) -> Tensor:
    """A [num_words] uint8 mask whose set bits are exactly the real rows.

    Uses ``np.packbits``' default big-endian bit order, matching how the truth
    tables in :mod:`.bitarrays` are packed, so bit i of word w corresponds to
    row ``w * 8 + i`` in both.
    """
    if num_rows > num_words * BITS_PER_WORD:
        raise ValueError(
            f"{num_rows} rows do not fit in {num_words} words "
            f"({num_words * BITS_PER_WORD} bits)"
        )
    bits = np.zeros(num_words * BITS_PER_WORD, dtype=np.uint8)
    bits[:num_rows] = 1
    return torch.from_numpy(np.packbits(bits))


@dataclass(frozen=True)
class BitpackedTask:
    """A truth table in packed form, resident on one device.

    ``root_values[i]`` is the input column driving root node ``i``, and
    ``target_values[j]`` is the desired output of output node ``j``. Both index
    by node slot, matching the description's fixed layout: roots occupy node
    indices ``[0, R)`` and outputs the final ``num_output_nodes`` indices.
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
    """
    packed_inputs, packed_sums = get_8bit_adder_truth_table()
    num_words = packed_inputs.shape[1]
    num_rows = 256 * 256  # every (a, b) pair, exhaustively
    if num_rows != num_words * BITS_PER_WORD:
        raise ValueError(
            f"adder table packs {num_words} words for {num_rows} rows; "
            "the truth table generator changed shape"
        )
    return BitpackedTask(
        root_values=torch.from_numpy(packed_inputs.copy()),
        target_values=torch.from_numpy(packed_sums.copy()),
        num_rows=num_rows,
        valid_bit_mask=make_valid_bit_mask(num_rows, num_words),
    ).to(device)


@dataclass
class _BatchedBitRank:
    """One topological rank, flattened across every graph in the batch."""

    batch_indices: Tensor  # [rows] which graph each node belongs to
    node_indices: Tensor  # [rows] node index within that graph
    parent_indices: Tensor  # [rows, maximum_indegree]
    valid_parent_mask: Tensor  # [rows, maximum_indegree]
    subtypes: Tensor  # [rows] raw node_type index


def _build_batched_ranks(
    graphs: Sequence[FixedInDegreeDAGDescription],
    device: torch.device,
) -> list[_BatchedBitRank]:
    """Concatenate every graph's precomputed rank metadata, rank by rank.

    Reuses ``graph.rank_batches`` (built once when the description is created)
    rather than recomputing topology here, and transfers each rank's tensors to
    the device in one concatenated copy per field.
    """
    max_ranks = max(len(graph.rank_batches) for graph in graphs)
    ranks: list[_BatchedBitRank] = []
    # Async host-to-device copies are only requested on CUDA. On MPS a
    # non-blocking copy of an index tensor can be read by the subsequent gather
    # before it lands, which silently yields out-of-range garbage indices
    # rather than an error; from unpinned host memory CUDA's non_blocking is
    # synchronous anyway, so nothing is lost by gating it.
    non_blocking = device.type == "cuda"

    for rank in range(max_ranks):
        batch_indices: list[Tensor] = []
        node_indices: list[Tensor] = []
        parent_indices: list[Tensor] = []
        valid_parent_masks: list[Tensor] = []
        subtypes: list[Tensor] = []

        for batch_index, graph in enumerate(graphs):
            if rank >= len(graph.rank_batches):
                continue
            rank_batch = graph.rank_batches[rank]
            num_rows = rank_batch.node_indices.shape[0]
            if num_rows == 0:
                continue
            batch_indices.append(torch.full((num_rows,), batch_index, dtype=torch.long))
            node_indices.append(rank_batch.node_indices)
            parent_indices.append(rank_batch.parent_indices)
            valid_parent_masks.append(rank_batch.valid_parent_mask)
            subtypes.append(rank_batch.subtypes)

        if not node_indices:
            empty = torch.empty(0, dtype=torch.long, device=device)
            maximum_indegree = graphs[0].maximum_indegree
            ranks.append(
                _BatchedBitRank(
                    batch_indices=empty,
                    node_indices=empty,
                    parent_indices=torch.empty(
                        0, maximum_indegree, dtype=torch.long, device=device
                    ),
                    valid_parent_mask=torch.empty(
                        0, maximum_indegree, dtype=torch.bool, device=device
                    ),
                    subtypes=empty,
                )
            )
            continue

        ranks.append(
            _BatchedBitRank(
                batch_indices=torch.cat(batch_indices).to(
                    device, non_blocking=non_blocking
                ),
                node_indices=torch.cat(node_indices).to(
                    device, non_blocking=non_blocking
                ),
                parent_indices=torch.cat(parent_indices).to(
                    device, non_blocking=non_blocking
                ),
                valid_parent_mask=torch.cat(valid_parent_masks).to(
                    device, non_blocking=non_blocking
                ),
                subtypes=torch.cat(subtypes).to(device, non_blocking=non_blocking),
            )
        )

    return ranks


def _validate_batch(
    graphs: Sequence[FixedInDegreeDAGDescription],
    task: BitpackedTask,
    gate_operators: Sequence[GateOperator],
) -> FixedInDegreeDAGDescription:
    """Check that the batch is homogeneous and matches the task, return graph 0."""
    if not graphs:
        raise ValueError("cannot evaluate an empty batch of graphs")
    reference = graphs[0]
    layout = (
        reference.num_root_nodes,
        reference.num_trunk_nodes,
        reference.num_output_nodes,
        reference.num_trunk_node_types,
        reference.maximum_indegree,
    )
    for index, graph in enumerate(graphs[1:], start=1):
        other = (
            graph.num_root_nodes,
            graph.num_trunk_nodes,
            graph.num_output_nodes,
            graph.num_trunk_node_types,
            graph.maximum_indegree,
        )
        if other != layout:
            raise ValueError(
                f"graph {index} has layout {other}, but graph 0 has {layout}; "
                "batched evaluation requires a homogeneous batch"
            )
    if reference.num_root_nodes != task.num_root_nodes:
        raise ValueError(
            f"graphs have {reference.num_root_nodes} roots but the task "
            f"supplies {task.num_root_nodes} input columns"
        )
    if reference.num_output_nodes != task.num_output_nodes:
        raise ValueError(
            f"graphs have {reference.num_output_nodes} outputs but the task "
            f"supplies {task.num_output_nodes} target columns"
        )
    if len(gate_operators) < reference.num_trunk_node_types:
        raise ValueError(
            f"{len(gate_operators)} gate operators for "
            f"{reference.num_trunk_node_types} trunk node types"
        )
    return reference


@torch.no_grad()
def evaluate_graphs(
    graphs: Sequence[FixedInDegreeDAGDescription],
    task: BitpackedTask,
    gate_operators: Sequence[GateOperator] = GATE_OPERATORS,
) -> Tensor:
    """Evaluate a batch of circuits against a task's inputs.

    Returns the packed output values, ``[B, num_output_nodes, num_words]``
    uint8, in output-slot order. Bits past ``task.num_rows`` are unspecified;
    see the module docstring.
    """
    reference = _validate_batch(graphs, task, gate_operators)
    device = task.root_values.device
    num_trunk_node_types = reference.num_trunk_node_types
    output_type = reference.output_node_types_start
    output_start = reference.num_root_nodes + reference.num_trunk_nodes

    buffer = torch.zeros(
        (len(graphs), reference.num_nodes, task.num_words),
        dtype=torch.uint8,
        device=device,
    )
    buffer[:, : reference.num_root_nodes] = task.root_values

    ranks = _build_batched_ranks(graphs, device)
    # Rank 0 is exactly the roots, which are already seeded; compute_node_ranks
    # gives every non-root node a rank of 1 + max(parent ranks) >= 1.
    for rank in ranks[1:]:
        if rank.node_indices.numel() == 0:
            continue

        # [rows, maximum_indegree, num_words]: every parent of every node at
        # this rank, gathered in one shot.
        parent_values = buffer[rank.batch_indices[:, None], rank.parent_indices]

        # Nodes at one rank can differ in type, and each type has its own gate
        # and in-degree, so dispatch per type over the rows of that type. With
        # two trunk types this is a handful of large vectorized ops per rank.
        for subtype in rank.subtypes.unique().tolist():
            rows = rank.subtypes == subtype
            selected = parent_values[rows]

            if subtype < num_trunk_node_types:
                in_degree = reference.trunk_node_in_degrees[subtype]
                operator = gate_operators[subtype]
            elif subtype == output_type:
                # Output nodes are pass-throughs with in-degree 1.
                in_degree = 1
                operator = None
            else:
                raise ValueError(
                    f"node type {subtype} appears at rank > 0; roots are "
                    "rank 0 and cannot be recomputed"
                )

            # The padded slots beyond a type's in-degree hold index 0, which is
            # a real (root) node -- reading them would silently produce wrong
            # values rather than an error, so confirm the slots we do read are
            # the ones the description marked valid.
            if not bool(rank.valid_parent_mask[rows][:, :in_degree].all()):
                raise ValueError(
                    f"node type {subtype} claims in-degree {in_degree}, but the "
                    "graph's parent slot mask disagrees"
                )

            values = selected[:, :in_degree]
            value = values[:, 0] if operator is None else operator(values)
            buffer[rank.batch_indices[rows], rank.node_indices[rows]] = value

    return buffer[:, output_start:]


def popcount(words: Tensor) -> Tensor:
    """Per-byte set-bit count of a uint8 tensor, returned as uint8 (0-8).

    SWAR rather than a lookup table: a table index would widen the tensor to
    int64 first, which for a full-truth-table batch is gigabytes of temporary.
    Every intermediate here stays uint8. Exhaustively tested over all 256 byte
    values in the test suite.
    """
    if words.dtype != torch.uint8:
        raise TypeError(f"popcount expects uint8, got {words.dtype}")
    counts = words - ((words >> 1) & 0x55)
    counts = (counts & 0x33) + ((counts >> 2) & 0x33)
    return (counts + (counts >> 4)) & 0x0F


def bit_accuracy(
    predicted: Tensor,
    task: BitpackedTask,
) -> tuple[Tensor, Tensor]:
    """Score packed outputs against a task's targets.

    Returns ``(overall, per_output)``: overall fitness per graph as ``[B]`` in
    [0, 1], and per-output-bit accuracy as ``[B, num_output_nodes]``. Only the
    first ``task.num_rows`` bits are counted, so the padding described in the
    module docstring cannot influence the score.

    Mismatches are counted exactly, as int64, on whatever device the outputs
    live on. The ratio is then taken in float64 on the CPU: the count tensor is
    only ``[B, num_output_nodes]``, so the transfer is free next to the
    evaluation, and it keeps the score exact for truth tables larger than
    float32's 2^24 integer range while sidestepping MPS, which has no float64
    at all. Move the result back to the device if you want to select on it
    there.
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


def evaluate_and_score(
    graphs: Sequence[FixedInDegreeDAGDescription],
    task: BitpackedTask,
    gate_operators: Sequence[GateOperator] = GATE_OPERATORS,
) -> tuple[Tensor, Tensor]:
    """Evaluate then score in one call. Returns ``(overall, per_output)``."""
    return bit_accuracy(evaluate_graphs(graphs, task, gate_operators), task)
