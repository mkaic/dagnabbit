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

The trunk block is a fixed number of *positions*, not a fixed number of gates.
Trailing trunk positions may hold the ``<MASK>`` type: in-degree 0, referenced
by nothing, carrying no value. That is how a graph smaller than the budget is
represented without changing the sequence length. See :class:`SamplingConfig`.

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
    per root slot, then a single shared output type, then ``<MASK>``. Outputs
    share one class because their fixed final positions already identify them.

    ``num_trunk_nodes`` is the number of trunk *positions*, which is fixed --
    it is what sets the sequence length. How many of them hold a gate rather
    than ``<MASK>`` is a sampling question, and lives in :class:`SamplingConfig`.
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
    def mask_type(self) -> int:
        """The type held by a trunk position that carries no gate.

        Deliberately outside ``[0, num_trunk_node_types)``: it names no
        operator and has no in-degree, so putting it in the gate range would
        force every consumer of ``trunk_node_in_degrees`` to special-case it.
        """
        return self.num_trunk_node_types + self.num_root_nodes + 1

    @property
    def num_node_types(self) -> int:
        return self.num_trunk_node_types + self.num_root_nodes + 2

    @property
    def num_truth_table_rows(self) -> int:
        return 1 << self.num_root_nodes


@dataclass(frozen=True)
class SamplingConfig:
    """How :func:`sample` draws *within* a fixed :class:`Geometry`.

    Both fields default to ``None``, which reproduces the original sampler
    exactly: every graph fills the whole trunk budget, and every gate type is
    drawn uniformly. They are separate from :class:`Geometry` because they
    describe the training prior, not the tensor shapes -- a hand-built circuit
    has a geometry but no sampling distribution.

    ``minimum_trunk_nodes`` makes ``geometry.num_trunk_nodes`` a *maximum*: each
    graph draws its live gate count uniformly from ``[minimum, maximum]`` and
    the remaining trunk positions become ``<MASK>``. Set it below the reference
    circuits' core sizes (35 gates for the NAND+XOR adder, 67 for the all-NAND
    one) if the point is for those to be reachable without buffer padding.

    ``trunk_type_concentration`` is the symmetric Dirichlet parameter for a
    graph's own mixture over gate types. 1.0 is uniform over the simplex, which
    is the useful default: a batch then spans everything from an even split to a
    near-single-type circuit, without changing the batch-wide marginal. Below 1
    pushes harder toward single-type graphs, above 1 back toward even mixtures.
    """

    minimum_trunk_nodes: int | None = None
    trunk_type_concentration: float | None = None

    def resolve_minimum_trunk_nodes(self, geometry: "Geometry") -> int:
        """The live-gate floor for ``geometry``, clamped to its trunk budget.

        Clamping rather than raising is what lets one config be reused across
        geometries -- the render scripts hand the training prior to whatever
        small geometry the flags describe, and "at least 32 of these 16
        positions" should mean 16, not an error.
        """
        if self.minimum_trunk_nodes is None:
            return geometry.num_trunk_nodes
        return min(self.minimum_trunk_nodes, geometry.num_trunk_nodes)

    @property
    def concentration(self) -> float:
        """The value the compiled sampler wants; 0 disables the Dirichlet draw."""
        return (
            0.0
            if self.trunk_type_concentration is None
            else float(self.trunk_type_concentration)
        )


DEFAULT_SAMPLING = SamplingConfig()


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

    @property
    def trunk_is_masked(self) -> Tensor:
        """``[B, T]`` true where a trunk position holds ``<MASK>``, not a gate.

        Masked positions have in-degree 0 and are referenced by nothing, so the
        evaluator's value for them is arbitrary and inert.
        """
        return self.trunk_types == self.geometry.mask_type

    @property
    def num_live_trunk_nodes(self) -> Tensor:
        """``[B]`` how many trunk positions each graph actually spends on gates."""
        return (~self.trunk_is_masked).sum(dim=1)

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
    sampling: SamplingConfig = DEFAULT_SAMPLING,
) -> GraphBatch:
    """``count`` freshly sampled random DAGs on ``device``.

    Seeded from ``torch.randint`` when ``seed`` is None, so ``torch.manual_seed``
    still determines the whole stream. numba keeps its own RNG state, which is
    why the seed is passed in explicitly rather than inherited.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    minimum_trunk_nodes = sampling.resolve_minimum_trunk_nodes(geometry)
    check_geometry(
        geometry.num_root_nodes,
        minimum_trunk_nodes,
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
        minimum_trunk_nodes,
        geometry.num_trunk_nodes,
        geometry.num_output_nodes,
        geometry.num_trunk_node_types,
        sampling.concentration,
        geometry.mask_type,
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
    earlier indices), so one forward sweep suffices. A parentless non-root is a
    ``<MASK>`` position, which sits on no path and stays at 0.
    """
    ranks = [0] * geometry.num_nodes
    for node_idx in range(geometry.num_root_nodes, geometry.num_nodes):
        parents = node_inputs_indices[node_idx]
        ranks[node_idx] = 1 + max(ranks[parent] for parent in parents) if parents else 0
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

    Repeated parents are legal and deliberately not rejected. ``<MASK>`` trunk
    positions are legal too, but only as a trailing block that nothing reads --
    the same layout :func:`sample` produces, so the two paths still describe the
    same space of graphs.
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

    producer_limit = geometry.output_start
    for i in range(geometry.num_trunk_nodes):
        node_idx = geometry.num_root_nodes + i
        trunk_type = node_types[node_idx]
        if trunk_type == geometry.mask_type:
            if node_inputs_indices[node_idx]:
                raise ValueError(f"node {node_idx} is <MASK> but has parents")
            producer_limit = min(producer_limit, node_idx)
            continue
        if not 0 <= trunk_type < geometry.num_trunk_node_types:
            raise ValueError(f"node {node_idx} has non-trunk type {trunk_type}")
        if node_idx > producer_limit:
            raise ValueError(
                f"node {node_idx} is a gate after the <MASK> block starts at "
                f"{producer_limit}; masked positions must be a trailing block"
            )
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
            if parent >= producer_limit:
                raise ValueError(
                    f"node {node_idx} references <MASK> node {parent}, which "
                    "holds no value"
                )
