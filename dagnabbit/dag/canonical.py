"""The one graph representation the model and the evaluator both read.

A graph is four batch-wide tensors in *canonical order*, so sequence position
``i`` and node index ``i`` are the same thing everywhere downstream. That
identity is what lets the model's token sequence and
:func:`~dagnabbit.tasks.logic_gates.evaluate.evaluate_choices` consume the exact
same tensors with no marshalling between them.

Two construction paths, which must agree:

* :func:`sample` -- the training path, straight from the compiled generator in
  :mod:`dagnabbit.dag.generate` through a handful of vectorized gathers. No
  per-graph Python object is ever built; a batch of 256 costs one numba call
  plus ~six numpy ops.
* :func:`from_lists` -- for hand-built circuits (the reference adder, tests),
  where the parent lists are written by hand and the canonical order has to be
  derived with the Kahn sweep in :func:`canonical_order_from_lists`.

Canonical position layout, fixed by construction: the ``num_root_nodes`` roots
occupy positions ``0..R-1`` in slot order, the ``num_output_nodes`` outputs
occupy the final positions in slot order, and trunk nodes fill the middle.
"""

import heapq
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from dagnabbit.dag.generate import check_geometry, generate_arrays


@dataclass(frozen=True)
class Geometry:
    """The shape of the graphs being sampled.

    Node *types* live in a single index space: trunk types first, then one type
    per root slot, then a single shared output type. Outputs share one class
    because their fixed final positions already identify them.
    """

    num_root_nodes: int
    num_trunk_nodes: int
    num_output_nodes: int
    num_trunk_node_types: int
    trunk_node_in_degrees: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.trunk_node_in_degrees) != self.num_trunk_node_types:
            raise ValueError(
                f"{len(self.trunk_node_in_degrees)} in-degrees for "
                f"{self.num_trunk_node_types} trunk types"
            )
        if min(self.trunk_node_in_degrees, default=1) < 1:
            raise ValueError("every trunk type needs in-degree >= 1")
        if self.num_root_nodes < 1 or self.num_output_nodes < 1:
            raise ValueError("need at least one root and one output")
        # Note that the *sampler's* extra preconditions are not checked here.
        # They are about whether random wiring can guarantee coverage, which a
        # hand-built circuit has no reason to satisfy; :func:`sample` checks them.

    @property
    def num_nodes(self) -> int:
        return self.num_root_nodes + self.num_trunk_nodes + self.num_output_nodes

    @property
    def output_start(self) -> int:
        """First output position. Also the number of *producer* positions."""
        return self.num_root_nodes + self.num_trunk_nodes

    @property
    def maximum_indegree(self) -> int:
        return max([1, *self.trunk_node_in_degrees])

    @property
    def root_type_start(self) -> int:
        return self.num_trunk_node_types

    @property
    def output_type(self) -> int:
        return self.num_trunk_node_types + self.num_root_nodes

    @property
    def num_node_types(self) -> int:
        return self.num_trunk_node_types + self.num_root_nodes + 1

    @property
    def num_truth_table_rows(self) -> int:
        return 1 << self.num_root_nodes


@dataclass
class CanonicalGraphs:
    """A batch of DAGs, every tensor in canonical order.

    ``parent_positions[b, i, s]`` is the canonical position of node ``i``'s
    slot-``s`` parent, always strictly less than ``i`` and strictly less than
    ``geometry.output_start`` (outputs are leaves). Padded slots hold 0 and are
    marked false in ``parent_slot_mask``; 0 rather than a stale index so that a
    consumer which reads through the mask still sees a legal position.
    """

    node_types: Tensor  # [B, N] long
    parent_positions: Tensor  # [B, N, S] long
    parent_slot_mask: Tensor  # [B, N, S] bool
    ranks: Tensor  # [B, N] long, longest-path depth
    geometry: Geometry

    def __len__(self) -> int:
        return self.node_types.shape[0]

    @property
    def device(self) -> torch.device:
        return self.node_types.device

    @property
    def trunk_types(self) -> Tensor:
        """``[B, T]`` raw trunk class ids, what ``evaluate_choices`` wants."""
        geometry = self.geometry
        return self.node_types[:, geometry.num_root_nodes : geometry.output_start]

    @property
    def output_ranks(self) -> Tensor:
        """``[B, num_output_nodes]`` longest-path depth of each output node."""
        return self.ranks[:, self.geometry.output_start :]

    def to(self, device: torch.device | str) -> "CanonicalGraphs":
        return CanonicalGraphs(
            node_types=self.node_types.to(device),
            parent_positions=self.parent_positions.to(device),
            parent_slot_mask=self.parent_slot_mask.to(device),
            ranks=self.ranks.to(device),
            geometry=self.geometry,
        )


def _canonicalize_arrays(
    node_types: np.ndarray,
    in_degrees: np.ndarray,
    padded_parents: np.ndarray,
    ranks: np.ndarray,
    order: np.ndarray,
    geometry: Geometry,
) -> CanonicalGraphs:
    """Reindex a batch of node-index-space arrays into canonical position space.

    Every step is one vectorized op over the whole batch. The old path built a
    Python object per graph to do this and spent more time marshalling the
    result than the compiled sampler spent producing it.
    """
    batch_size = node_types.shape[0]
    slots = geometry.maximum_indegree

    # order maps position -> node index, so its inverse permutation maps node
    # index -> position. argsort of a permutation is exactly that inverse.
    positions = np.argsort(order, axis=1)

    slot_mask = np.arange(slots) < in_degrees[..., None]
    # Parent *node indices* become parent *positions*, still in node-index row
    # order; padded slots are forced to 0 before the mask is applied so they can
    # never carry a stale position.
    parent_positions = positions[np.arange(batch_size)[:, None, None], padded_parents]
    parent_positions = np.where(slot_mask, parent_positions, 0)

    rows = order[:, :, None].repeat(slots, axis=2)
    return CanonicalGraphs(
        node_types=torch.from_numpy(np.take_along_axis(node_types, order, axis=1)),
        parent_positions=torch.from_numpy(
            np.take_along_axis(parent_positions, rows, axis=1)
        ),
        parent_slot_mask=torch.from_numpy(np.take_along_axis(slot_mask, rows, axis=1)),
        ranks=torch.from_numpy(np.take_along_axis(ranks, order, axis=1)),
        geometry=geometry,
    )


def sample(
    count: int,
    geometry: Geometry,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> CanonicalGraphs:
    """``count`` freshly sampled random DAGs, canonicalized, on ``device``.

    Seeded from ``torch.randint`` when ``seed`` is None, so ``torch.manual_seed``
    still determines the whole stream. numba keeps its own RNG state, which is
    why the seed is passed in explicitly rather than inherited.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    check_geometry(
        geometry.num_root_nodes,
        geometry.num_trunk_nodes,
        geometry.num_output_nodes,
        geometry.num_trunk_node_types,
        list(geometry.trunk_node_in_degrees),
    )
    if seed is None:
        # numba's seed is a uint32, so draw in that range rather than truncating.
        seed = int(torch.randint(0, 2**32, (1,), dtype=torch.int64).item())

    node_types, in_degrees, padded_parents, ranks, order = generate_arrays(
        count,
        geometry.num_root_nodes,
        geometry.num_trunk_nodes,
        geometry.num_output_nodes,
        geometry.num_trunk_node_types,
        np.asarray(geometry.trunk_node_in_degrees, dtype=np.int64),
        geometry.maximum_indegree,
        seed,
    )
    graphs = _canonicalize_arrays(
        node_types, in_degrees, padded_parents, ranks, order, geometry
    )
    return graphs.to(device)


def canonical_order_from_lists(
    node_inputs_indices: list[list[int]],
    node_types: list[int],
    geometry: Geometry,
) -> list[int]:
    """Structure-canonical topological order (position -> node index).

    Roots are pinned to the first positions in slot order and outputs to the
    last in slot order. Trunk order comes from Kahn's algorithm with a
    deterministic tie-break: among ready nodes, emit the one with the
    lexicographically smallest ``(parent canonical positions in slot order,
    node_type)``. Parent positions are compared in slot order, not sorted,
    because input-slot order is semantically meaningful.

    The key depends only on structure, so isomorphic graphs canonicalize
    identically, with one residual ambiguity: exact structural twins (same type,
    same parents in the same slots) compare equal and fall back to original node
    index. Twins are interchangeable as producers, but a consumer referencing
    both in different slots can see its parent tuple swap under relabeling. That
    is accepted rather than paying for full graph canonization.

    This is the slow path, for hand-built circuits only. The compiled generator
    reimplements it and there is a differential test pinning the two together.
    """
    num_root_nodes = geometry.num_root_nodes
    output_start = geometry.output_start
    num_nodes = geometry.num_nodes

    positions: list[int | None] = [None] * num_nodes
    for root_idx in range(num_root_nodes):
        positions[root_idx] = root_idx

    children: list[list[int]] = [[] for _ in range(output_start)]
    blocking_parents = [0] * output_start
    for node_idx in range(num_root_nodes, output_start):
        for parent in node_inputs_indices[node_idx]:
            if parent >= num_root_nodes:
                blocking_parents[node_idx] += 1
                children[parent].append(node_idx)

    def ready_key(node_idx: int) -> tuple:
        # All parents are placed by the time a node enters the heap: roots are
        # pre-placed and trunk parents gate readiness, so no position is None.
        return (
            tuple(positions[parent] for parent in node_inputs_indices[node_idx]),
            node_types[node_idx],
            node_idx,
        )

    heap = [
        ready_key(node_idx)
        for node_idx in range(num_root_nodes, output_start)
        if blocking_parents[node_idx] == 0
    ]
    heapq.heapify(heap)

    order = list(range(num_root_nodes))
    while heap:
        node_idx = heapq.heappop(heap)[-1]
        positions[node_idx] = len(order)
        order.append(node_idx)
        for child in children[node_idx]:
            blocking_parents[child] -= 1
            if blocking_parents[child] == 0:
                heapq.heappush(heap, ready_key(child))

    assert len(order) == output_start, "canonical sort left unplaced trunk nodes"
    order.extend(range(output_start, num_nodes))
    return order


def ranks_from_lists(
    node_inputs_indices: list[list[int]], geometry: Geometry
) -> list[int]:
    """Longest-path depth of each node, roots at 0.

    Nodes are stored in topological order (every non-root references only
    earlier indices), so one forward sweep suffices.
    """
    ranks = [0] * geometry.num_nodes
    for node_idx in range(geometry.num_root_nodes, geometry.num_nodes):
        ranks[node_idx] = 1 + max(
            ranks[parent] for parent in node_inputs_indices[node_idx]
        )
    return ranks


def from_lists(
    node_inputs_indices: list[list[int]],
    node_types: list[int],
    geometry: Geometry,
    device: torch.device | str = "cpu",
) -> CanonicalGraphs:
    """One hand-built circuit as a batch of size 1.

    ``node_inputs_indices`` is ragged and in node-index order, with roots first
    and outputs last, matching the layout ``geometry`` declares.
    """
    validate_lists(node_inputs_indices, node_types, geometry)

    num_nodes = geometry.num_nodes
    slots = geometry.maximum_indegree
    in_degrees = np.fromiter(map(len, node_inputs_indices), np.int64, num_nodes)
    padded_parents = np.zeros((num_nodes, slots), dtype=np.int64)
    for node_idx, parents in enumerate(node_inputs_indices):
        padded_parents[node_idx, : len(parents)] = parents

    order = np.asarray(
        canonical_order_from_lists(node_inputs_indices, node_types, geometry),
        dtype=np.int64,
    )
    graphs = _canonicalize_arrays(
        np.asarray(node_types, dtype=np.int64)[None],
        in_degrees[None],
        padded_parents[None],
        np.asarray(ranks_from_lists(node_inputs_indices, geometry), dtype=np.int64)[
            None
        ],
        order[None],
        geometry,
    )
    return graphs.to(device)


def validate_lists(
    node_inputs_indices: list[list[int]],
    node_types: list[int],
    geometry: Geometry,
) -> None:
    """Check a hand-built circuit against its declared geometry.

    Only the list path runs this; :func:`sample` skips it because the compiled
    generator constructs the same invariants by hand and re-checking them per
    node would cost more than generating the graph did.
    """
    if len(node_types) != geometry.num_nodes:
        raise ValueError(f"{len(node_types)} node types for {geometry.num_nodes} nodes")
    if len(node_inputs_indices) != geometry.num_nodes:
        raise ValueError(
            f"{len(node_inputs_indices)} parent lists for {geometry.num_nodes} nodes"
        )

    for i in range(geometry.num_root_nodes):
        if node_inputs_indices[i] or node_types[i] != geometry.root_type_start + i:
            raise ValueError(f"node {i} is not root slot {i}")

    for i in range(geometry.num_trunk_nodes):
        node_idx = geometry.num_root_nodes + i
        trunk_type = node_types[node_idx]
        if not 0 <= trunk_type < geometry.num_trunk_node_types:
            raise ValueError(f"node {node_idx} has non-trunk type {trunk_type}")
        expected = geometry.trunk_node_in_degrees[trunk_type]
        if len(node_inputs_indices[node_idx]) != expected:
            raise ValueError(
                f"node {node_idx} has {len(node_inputs_indices[node_idx])} parents, "
                f"but type {trunk_type} has in-degree {expected}"
            )

    for i in range(geometry.num_output_nodes):
        node_idx = geometry.output_start + i
        if (
            len(node_inputs_indices[node_idx]) != 1
            or node_types[node_idx] != geometry.output_type
        ):
            raise ValueError(f"node {node_idx} is not an in-degree-1 output")

    for node_idx, parents in enumerate(node_inputs_indices):
        for parent in parents:
            if not 0 <= parent < node_idx:
                raise ValueError(
                    f"node {node_idx} references {parent}, which is not earlier"
                )
            if parent >= geometry.output_start:
                raise ValueError(
                    f"node {node_idx} references output node {parent}; "
                    "outputs are leaves"
                )
