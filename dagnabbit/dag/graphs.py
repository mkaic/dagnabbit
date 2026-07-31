"""The one graph representation the model and the evaluator both read.

A graph is four batch-wide tensors indexed by node. Node index order is already
a valid topological order -- roots occupy ``[0, R)``, outputs the final
positions, and every non-root references only earlier indices -- so there is no
reordering step and no separate notion of "position". Sequence position ``i``,
node index ``i``, and parent pointer value ``i`` all mean the same thing, which
is what lets the model's token sequence and
:func:`~dagnabbit.tasks.logic_gates.evaluate.evaluate_choices` consume the exact
same tensors with no marshalling between them.

An earlier version canonicalized the order with a Kahn sweep so that isomorphic
graphs produced identical sequences. That mattered when the model was trained to
reconstruct a sequence; it does not now, and it was actively harmful -- under a
structure-derived ordering a node's index leaks information about its structure,
which a model can exploit instead of reading parent pointers.

Two construction paths:

* :func:`sample` -- the training path, straight from the compiled generator in
  :mod:`dagnabbit.dag.generate`. No per-graph Python object is ever built.
* :func:`from_lists` -- for hand-built circuits (the reference adder, tests),
  where the parent lists are written by hand.
"""

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
        # The *sampler's* extra preconditions are deliberately not checked here.
        # They are about whether random wiring can guarantee coverage, which a
        # hand-built circuit has no reason to satisfy; :func:`sample` checks them.

    @property
    def num_nodes(self) -> int:
        return self.num_root_nodes + self.num_trunk_nodes + self.num_output_nodes

    @property
    def output_start(self) -> int:
        """First output index. Also the number of *producer* nodes."""
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
class GraphBatch:
    """A batch of DAGs, every tensor indexed by node.

    ``parent_indices[b, i, s]`` is the node index of node ``i``'s slot-``s``
    parent, always strictly less than ``i`` and strictly less than
    ``geometry.output_start`` (outputs are leaves). A node may hold the same
    parent in both slots -- ``NAND(x, x) = NOT x`` is how inverters exist.
    Padded slots hold 0 and are marked false in ``parent_slot_mask``; 0 rather
    than a stale index so a consumer reading through the mask still sees a
    legal node.
    """

    node_types: Tensor  # [B, N] long
    parent_indices: Tensor  # [B, N, S] long
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

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(
            node_types=self.node_types.to(device),
            parent_indices=self.parent_indices.to(device),
            parent_slot_mask=self.parent_slot_mask.to(device),
            ranks=self.ranks.to(device),
            geometry=self.geometry,
        )


def _wrap(
    node_types: np.ndarray,
    in_degrees: np.ndarray,
    padded_parents: np.ndarray,
    ranks: np.ndarray,
    geometry: Geometry,
) -> GraphBatch:
    """Wrap the generator's arrays as tensors. One vectorized op, no reordering."""
    slot_mask = np.arange(geometry.maximum_indegree) < in_degrees[..., None]
    # Padded slots are forced to 0 so they can never carry a stale index.
    parents = np.where(slot_mask, padded_parents, 0)
    return GraphBatch(
        node_types=torch.from_numpy(np.ascontiguousarray(node_types)),
        parent_indices=torch.from_numpy(parents),
        parent_slot_mask=torch.from_numpy(slot_mask),
        ranks=torch.from_numpy(np.ascontiguousarray(ranks)),
        geometry=geometry,
    )


def sample(
    count: int,
    geometry: Geometry,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> GraphBatch:
    """``count`` freshly sampled random DAGs on ``device``.

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

    arrays = generate_arrays(
        count,
        geometry.num_root_nodes,
        geometry.num_trunk_nodes,
        geometry.num_output_nodes,
        geometry.num_trunk_node_types,
        np.asarray(geometry.trunk_node_in_degrees, dtype=np.int64),
        geometry.maximum_indegree,
        seed,
    )
    return _wrap(*arrays, geometry).to(device)


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
) -> GraphBatch:
    """One hand-built circuit as a batch of size 1.

    ``node_inputs_indices`` is ragged and in node-index order, with roots first
    and outputs last, matching the layout ``geometry`` declares.
    """
    validate_lists(node_inputs_indices, node_types, geometry)

    num_nodes = geometry.num_nodes
    in_degrees = np.fromiter(map(len, node_inputs_indices), np.int64, num_nodes)
    padded_parents = np.zeros((num_nodes, geometry.maximum_indegree), dtype=np.int64)
    for node_idx, parents in enumerate(node_inputs_indices):
        padded_parents[node_idx, : len(parents)] = parents

    return _wrap(
        np.asarray(node_types, dtype=np.int64)[None],
        in_degrees[None],
        padded_parents[None],
        np.asarray(ranks_from_lists(node_inputs_indices, geometry), dtype=np.int64)[
            None
        ],
        geometry,
    ).to(device)


def validate_lists(
    node_inputs_indices: list[list[int]],
    node_types: list[int],
    geometry: Geometry,
) -> None:
    """Check a hand-built circuit against its declared geometry.

    Only the list path runs this; :func:`sample` skips it because the compiled
    generator constructs the same invariants by hand and re-checking them per
    node would cost more than generating the graph did.

    Repeated parents are legal and deliberately not rejected.
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
