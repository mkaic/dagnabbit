import functools
import heapq
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch


class NodeSupertype(Enum):
    """Module-level role of a node, used as a batch key for grouped evaluation."""

    ROOT = "root"
    TRUNK = "trunk"
    OUTPUT = "output"


def subtype_to_supertype(
    node_type: int,
    num_trunk_node_types: int | None = None,
    num_root_nodes: int | None = None,
) -> NodeSupertype:
    """Map a raw ``node_type`` subtype index to its :class:`NodeSupertype`.

    Uses the unified type-index layout:
        [0, num_trunk_node_types)                        -> TRUNK
        [root_node_types_start, output_node_types_start) -> ROOT
        [output_node_types_start]                        -> OUTPUT (single class)

    All output nodes share one output class -- individual outputs are already
    identifiable by their fixed slot positions, so the classifier does not
    distinguish between them.

    Any layout argument left as ``None`` falls back to the corresponding value
    in ``dagnabbit.scripts.config``, so training callsites can omit them.
    """
    if num_trunk_node_types is None or num_root_nodes is None:
        from dagnabbit.scripts import config as cfg

        if num_trunk_node_types is None:
            num_trunk_node_types = cfg.NUM_TRUNK_NODE_TYPES
        if num_root_nodes is None:
            num_root_nodes = cfg.NUM_ROOT_NODES

    root_node_types_start = num_trunk_node_types
    output_node_types_start = num_trunk_node_types + num_root_nodes
    num_node_types = num_trunk_node_types + num_root_nodes + 1

    if node_type < root_node_types_start:
        return NodeSupertype.TRUNK
    if node_type < output_node_types_start:
        return NodeSupertype.ROOT
    if node_type < num_node_types:
        return NodeSupertype.OUTPUT
    raise ValueError(f"unknown node type subtype index: {node_type}")


@dataclass
class PreparedRankBatch:
    """CPU-side padded tensors for one topological rank of one graph."""

    node_indices: torch.Tensor
    parent_indices: torch.Tensor
    valid_parent_mask: torch.Tensor
    subtypes: torch.Tensor
    has_valid_parents: bool = False


@dataclass
class RankPartition:
    """One graph's whole rank partition, flat.

    Rows are ordered rank-major, ascending by node index within each rank, so
    rank ``r`` occupies a contiguous slice whose length is ``counts[r]``.

    Held flat rather than as a tensor per rank because *every* consumer
    immediately concatenates across the batch: splitting the partition into ~4
    views per rank per graph, only for the model to re-join them, cost more than
    building the partition in the first place. :func:`collate_rank_partitions`
    is the batch-side counterpart. :attr:`FixedInDegreeDAGDescription.rank_batches`
    still offers the per-rank view for code that reads one graph at a time.

    A pleasant side effect: flat contiguous tensors pickle by value at their true
    size, so shipping a description to a worker process needs no special handling
    (a *slice* would drag its whole base storage along, once per rank).
    """

    # [N] long: node index of each row, within its own graph.
    node_indices: torch.Tensor
    # [N, maximum_indegree] long, zero-padded past each node's in-degree.
    parent_indices: torch.Tensor
    # [N, maximum_indegree] bool: which parent slots are real.
    valid_parent_mask: torch.Tensor
    # [N] long: raw node_type index of each row.
    subtypes: torch.Tensor
    # [N] long: which rank each row belongs to. Redundant with ``counts``, but it
    # is what lets a batch of graphs be regrouped by rank in one stable argsort.
    rank_of_row: torch.Tensor
    # Rows per rank, indexed by rank.
    counts: tuple[int, ...]
    # Whether any node at that rank has at least one parent, indexed by rank.
    has_valid_parents: tuple[bool, ...]

    @property
    def num_ranks(self) -> int:
        return len(self.counts)


class FixedInDegreeDAGDescription:
    """A fixed-in-degree DAG, plus the overlays the model reads.

    Two construction paths, which must produce indistinguishable objects:

    * :meth:`__init__`, from ragged Python lists -- what hand-built reference
      circuits, blind decode and most tests use.
    * :meth:`from_arrays`, from the padded arrays the compiled generator in
      :mod:`dagnabbit.dag.generate` emits -- what the training loop uses.

    Numpy arrays are the source of truth for everything on the hot path, and the
    ragged Python views of them (``node_inputs_indices``, ``node_types``,
    ``node_ranks``, ``canonical_order``, ``canonical_positions``,
    ``leaf_node_indices``) are ``cached_property``, so the array path never pays
    to materialize what nothing on that path reads. That matters more than it
    sounds: building ``node_inputs_indices`` alone costs ~54us per graph, about
    five times the compiled generator that produced the arrays in the first
    place. Assigning one of those names in ``__init__`` shadows its property,
    which is exactly how the list path avoids deriving its own inputs.
    """

    def __init__(
        self,
        num_root_nodes: int,
        num_trunk_nodes: int,
        num_output_nodes: int,
        num_trunk_node_types: int,
        trunk_node_in_degrees: int | list[int],
        node_inputs_indices: list[list[int]],
        node_types: list[int],
    ):
        self._init_layout(
            num_root_nodes,
            num_trunk_nodes,
            num_output_nodes,
            num_trunk_node_types,
            trunk_node_in_degrees,
        )
        # Given, not derived: these shadow the cached_property views below.
        self.node_inputs_indices = node_inputs_indices
        self.node_types = node_types
        self._validate_against_layout()
        self._build_overlays()

    @classmethod
    def from_arrays(
        cls,
        *,
        num_root_nodes: int,
        num_trunk_nodes: int,
        num_output_nodes: int,
        num_trunk_node_types: int,
        trunk_node_in_degrees: int | list[int],
        node_types_array: np.ndarray,
        padded_parents: np.ndarray,
        parent_slot_mask: np.ndarray,
        ranks_array: np.ndarray,
        canonical_order_array: np.ndarray,
    ) -> "FixedInDegreeDAGDescription":
        """Build from precomputed arrays instead of deriving them.

        Every argument is a per-graph slice of the batch-wide arrays the compiled
        generator fills, so this does no topology work at all -- the ranks and the
        canonical order arrive already computed, bit-identical to what
        :meth:`compute_node_ranks` and :meth:`compute_canonical_order` produce
        (there is a differential test pinning that).

        The arrays are adopted, not copied. They are slices of batch-wide arrays
        and are never written to after construction.
        """
        self = cls.__new__(cls)
        self._init_layout(
            num_root_nodes,
            num_trunk_nodes,
            num_output_nodes,
            num_trunk_node_types,
            trunk_node_in_degrees,
        )
        # Shadow the derivations; each of these is what a cached_property here
        # would otherwise have to compute.
        self.node_types_array = node_types_array
        self.parent_arrays = (padded_parents, parent_slot_mask)
        self.ranks_array = ranks_array
        self.canonical_order_array = canonical_order_array
        self._build_overlays()
        return self

    def _init_layout(
        self,
        num_root_nodes: int,
        num_trunk_nodes: int,
        num_output_nodes: int,
        num_trunk_node_types: int,
        trunk_node_in_degrees: int | list[int],
    ) -> None:
        if isinstance(trunk_node_in_degrees, int):
            trunk_node_in_degrees = [trunk_node_in_degrees] * num_trunk_node_types

        assert len(trunk_node_in_degrees) == num_trunk_node_types

        self.num_trunk_nodes = num_trunk_nodes
        self.num_root_nodes = num_root_nodes
        self.num_output_nodes = num_output_nodes
        self.num_trunk_node_types = num_trunk_node_types
        self.trunk_node_in_degrees = trunk_node_in_degrees
        self.maximum_indegree = max([1, *self.trunk_node_in_degrees])
        self.num_nodes = num_root_nodes + num_trunk_nodes + num_output_nodes

        # Type-index layout in a single space immediately after the trunk types.
        # Each root gets its own unique type; all output nodes share one class,
        # since individual outputs are identifiable by their fixed slot positions:
        #   [0, num_trunk_node_types)                        -> trunk types
        #   [root_node_types_start, output_node_types_start) -> one type per root slot
        #   [output_node_types_start]                        -> single shared output type
        self.root_node_types_start = num_trunk_node_types
        self.output_node_types_start = num_trunk_node_types + num_root_nodes
        self.num_node_types = num_trunk_node_types + num_root_nodes + 1

    def _validate_against_layout(self) -> None:
        """Check the given lists against the declared geometry.

        Only the list path runs this. :meth:`from_arrays` skips it deliberately:
        its inputs come from the compiled generator, which constructs the same
        invariants by hand, and re-checking them per node would cost more than
        generating the graph did.
        """
        assert len(self.node_types) == self.num_nodes

        for i in range(self.num_root_nodes):
            assert len(self.node_inputs_indices[i]) == 0
            assert self.node_types[i] == self.root_node_types_start + i

        for i in range(self.num_trunk_nodes):
            node_idx = self.num_root_nodes + i
            trunk_type = self.node_types[node_idx]
            assert 0 <= trunk_type < self.num_trunk_node_types
            expected = self.trunk_node_in_degrees[trunk_type]
            assert len(self.node_inputs_indices[node_idx]) == expected

        for i in range(self.num_output_nodes):
            node_idx = self.num_root_nodes + self.num_trunk_nodes + i
            assert len(self.node_inputs_indices[node_idx]) == 1
            assert self.node_types[node_idx] == self.output_node_types_start

    def _build_overlays(self) -> None:
        """Build the eagerly-needed tensors. Shared by both construction paths.

        Everything here reads the ``*_array`` properties rather than the Python
        lists, so which path built the object makes no difference to the result.
        """
        self.node_types_tensor = torch.from_numpy(self.node_types_array)

        # Batch overlay over the flat arrays: longest-path rank per node, and the
        # padded rank partition iterated over during grouped evaluation.
        self.rank_partition = self.build_rank_partition()

        # Canonical sequence overlay: structure-derived topological order with
        # roots pinned first and outputs pinned last, plus padded per-position
        # parent tensors consumed by the sequence compressor/decoder.
        #
        # Node-storage-index -> canonical position: the recursive encoder gathers
        # through it to give every node's context tokens the absolute graph
        # positions the pointer head predicts into.
        self.canonical_positions_tensor = torch.from_numpy(
            self.canonical_positions_array
        )
        (
            self.canonical_order_tensor,
            self.canonical_node_types,
            self.canonical_parent_positions,
            self.canonical_parent_slot_mask,
        ) = self.build_canonical_tensors()

    # ---- array primitives, and the Python views of them ----
    #
    # In each pair the array is the primitive and the list is a lazy view, so
    # that a description built from arrays never materializes a list nothing
    # reads, and one built from lists still exposes every array the hot path
    # wants. Whichever path ran, assigning the primitive in the constructor
    # shadows the property that would otherwise compute it.

    @functools.cached_property
    def node_types_array(self) -> np.ndarray:
        """``node_types`` as ``[N]`` int64, the gather source for type tensors."""
        return np.fromiter(self.node_types, np.int64, self.num_nodes)

    @functools.cached_property
    def node_types(self) -> list[int]:
        return self.node_types_array.tolist()

    @functools.cached_property
    def node_inputs_indices(self) -> list[list[int]]:
        """Per-node parent lists, ragged.

        Read only by things that walk a single graph -- rendering, canonical
        hashing, diagnostics -- never by the batched model paths, which is why it
        is worth deriving on demand rather than always building. One ``tolist``
        for the whole padded block, then a slice per node, is the cheapest route.
        """
        padded_parents, slot_mask = self.parent_arrays
        rows = padded_parents.tolist()
        degrees = slot_mask.sum(axis=1).tolist()
        return [row[:degree] for row, degree in zip(rows, degrees)]

    @functools.cached_property
    def ranks_array(self) -> np.ndarray:
        return np.fromiter(self.compute_node_ranks(), np.int64, self.num_nodes)

    @functools.cached_property
    def node_ranks(self) -> list[int]:
        return self.ranks_array.tolist()

    @functools.cached_property
    def canonical_order_array(self) -> np.ndarray:
        return np.fromiter(self.compute_canonical_order(), np.int64, self.num_nodes)

    @functools.cached_property
    def canonical_order(self) -> list[int]:
        return self.canonical_order_array.tolist()

    @functools.cached_property
    def canonical_positions_array(self) -> np.ndarray:
        """The inverse permutation of :attr:`canonical_order_array`."""
        positions = np.empty(self.num_nodes, dtype=np.int64)
        positions[self.canonical_order_array] = np.arange(self.num_nodes)
        return positions

    @functools.cached_property
    def canonical_positions(self) -> list[int]:
        return self.canonical_positions_array.tolist()

    @functools.cached_property
    def leaf_node_indices_array(self) -> np.ndarray:
        """Nodes nobody references as an input. Outputs are always among them.

        Lazy, along with the two views below: only rendering reads any of them,
        and finding them cost ~8us per graph against a ~11us compiled generator.
        """
        padded_parents, slot_mask = self.parent_arrays
        referenced = np.zeros(self.num_nodes, dtype=bool)
        referenced[padded_parents[slot_mask]] = True
        return np.flatnonzero(~referenced)

    @functools.cached_property
    def leaf_node_indices(self) -> list[int]:
        return self.leaf_node_indices_array.tolist()

    @functools.cached_property
    def leaf_node_indices_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.leaf_node_indices_array)

    @functools.cached_property
    def parent_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(padded_parents, slot_mask)``, both ``[N, maximum_indegree]``.

        The one numpy view of the ragged ``node_inputs_indices``, from which
        every derived overlay below (leaves, rank batches, canonical tensors) is
        a vectorized gather rather than a Python loop over nodes. Marshalling
        that derived data, not the random sampling, is the bulk of the cost of
        generating a graph, and it happens once per graph in the training loop.

        Cached rather than stored: it is pure IPC weight when a description
        crosses a process boundary, so :meth:`__getstate__` drops it and the
        rare consumer that needs it after a round trip rebuilds it here.
        """
        pad = self.maximum_indegree
        in_degrees = np.fromiter(
            map(len, self.node_inputs_indices), np.int64, self.num_nodes
        )
        largest = int(in_degrees.max(initial=0))
        if largest > pad:
            raise ValueError(f"node in-degree {largest} is above maximum {pad}")
        slot_mask = np.arange(pad) < in_degrees[:, None]
        padded = np.zeros((self.num_nodes, pad), dtype=np.int64)
        padded[slot_mask] = np.fromiter(
            itertools.chain.from_iterable(self.node_inputs_indices),
            np.int64,
            int(in_degrees.sum()),
        )
        return padded, slot_mask

    def compute_node_ranks(self) -> list[int]:
        """Longest-path depth of each node (roots are rank 0).

        Every other node's rank is ``1 + max(parent ranks)``. Nodes are stored
        in topological order (roots first, every trunk/output node only
        references earlier indices), so a single forward sweep suffices.
        """
        inputs = self.node_inputs_indices
        ranks = [0] * self.num_nodes
        for node_idx in range(self.num_root_nodes, self.num_nodes):
            # One max() over the slot list rather than one per parent; every
            # non-root node has at least one parent, so the list is never empty.
            ranks[node_idx] = 1 + max([ranks[parent] for parent in inputs[node_idx]])
        return ranks

    @functools.cached_property
    def rank_batches(self) -> list[PreparedRankBatch]:
        """The rank partition split per rank, for one-graph-at-a-time consumers.

        Batched code paths should read :attr:`rank_partition` instead and stay
        flat: materializing these views was ~12% of the cost of generating a
        graph, and both batch collators used to undo them immediately. Kept as a
        lazily built convenience for rendering, tests and diagnostics.
        """
        partition = self.rank_partition
        batches: list[PreparedRankBatch] = []
        offset = 0
        for count, has_valid_parents in zip(
            partition.counts, partition.has_valid_parents
        ):
            start = offset
            offset += count
            end = offset
            batches.append(
                PreparedRankBatch(
                    node_indices=partition.node_indices[start:end],
                    parent_indices=partition.parent_indices[start:end],
                    valid_parent_mask=partition.valid_parent_mask[start:end],
                    subtypes=partition.subtypes[start:end],
                    has_valid_parents=has_valid_parents,
                )
            )
        return batches

    # ---- pickling ----
    #
    # Nothing in the training loop pickles a description any more -- graph
    # batches used to be generated in worker processes and are not now. Kept
    # because it is eight lines and makes descriptions cheap to copy or persist
    # should anything want to.

    # The array primitives are what travel. ``_build_overlays`` reads all six of
    # them, so whichever constructor ran they are present in ``__dict__`` by the
    # time an object exists -- which is what makes it safe to drop every ragged
    # Python view below unconditionally and let each rebuild on demand. Dropping
    # an *array* instead would be a bug: for an object built by
    # :meth:`from_arrays` the ragged views derive from the arrays, so losing an
    # array leaves the pair mutually recursive.
    _PICKLED_ARRAYS = (
        "node_types_array",
        "parent_arrays",
        "ranks_array",
        "canonical_order_array",
        "canonical_positions_array",
    )
    _DROPPED_ON_PICKLE = (
        # Views of rank_partition, and the reason it is stored flat: these are
        # *slices*, and pickling a slice serializes its entire base storage, so
        # shipping them wrote each base once per rank -- 28.5 MB for a batch of
        # 256 against the ~2.5 MB the data occupies, and 418 ms to load. That was
        # worse than regenerating the batch outright.
        "rank_batches",
        # Ragged Python views of the arrays above. node_inputs_indices alone is
        # 152 small lists at the training geometry.
        "node_inputs_indices",
        "node_types",
        "node_ranks",
        "canonical_order",
        "canonical_positions",
        "leaf_node_indices",
        "leaf_node_indices_array",
        "leaf_node_indices_tensor",
    )

    def __getstate__(self) -> dict:
        """Ship the arrays; drop every view derivable from them."""
        state = self.__dict__.copy()
        for cached in self._DROPPED_ON_PICKLE:
            state.pop(cached, None)
        missing = [name for name in self._PICKLED_ARRAYS if name not in state]
        assert not missing, (
            f"array primitives missing from pickled state: {missing}; the views "
            "dropped above derive from them and would recurse"
        )
        return state

    def build_rank_partition(self) -> RankPartition:
        """Precompute the padded CPU rank partition used by the model hot path.

        The training loop needs a node-sorted padded view of each rank every
        encode/decode pass, so build that once when the graph is created instead
        of rebuilding it on the compute device.

        This runs once per generated graph in the training loop, so the whole
        partition is one stable argsort of the node ranks plus one gather per
        tensor through it -- five torch objects for the graph, not four per rank.
        Building the padded rows node by node in Python, or making a tensor per
        rank, each cost tens of microseconds of
        interpreter and allocator overhead and dominated graph generation.
        """
        padded_parents, slot_mask = self.parent_arrays
        ranks = self.ranks_array
        num_ranks = int(ranks.max(initial=-1)) + 1

        # A stable sort by rank groups the nodes by rank while leaving each
        # rank's slice in ascending node order, which is what the hot path
        # indexes by. Empty ranks cannot occur (a node at rank r has a parent at
        # r-1), but bincount tolerates them, unlike a reduceat over boundaries.
        order = np.argsort(ranks, kind="stable")
        counts = np.bincount(ranks, minlength=num_ranks)
        has_valid_parents = np.bincount(
            ranks, weights=slot_mask.any(axis=1), minlength=num_ranks
        ).astype(bool)

        return RankPartition(
            node_indices=torch.from_numpy(order),
            parent_indices=torch.from_numpy(padded_parents[order]),
            valid_parent_mask=torch.from_numpy(slot_mask[order]),
            subtypes=torch.from_numpy(self.node_types_array[order]),
            rank_of_row=torch.from_numpy(ranks[order]),
            counts=tuple(counts.tolist()),
            has_valid_parents=tuple(has_valid_parents.tolist()),
        )

    def compute_canonical_order(self) -> list[int]:
        """Structure-canonical topological order of the graph's nodes.

        Position layout: the ``num_root_nodes`` roots occupy positions
        ``0..R-1`` in root-slot order, the ``num_output_nodes`` outputs occupy
        the final positions in output-slot order, and trunk nodes fill the
        middle. Trunk order comes from Kahn's algorithm with a deterministic
        tie-break: among ready nodes, emit the one with the lexicographically
        smallest key ``(parent canonical positions in slot order, node_type)``.
        Parent positions are compared in slot order -- not sorted -- because
        input-slot order is semantically meaningful.

        The key depends only on graph structure, so isomorphic graphs produce
        identical canonical sequences, with one residual ambiguity: exact
        structural twins (same type, same parents in the same slots) compare
        equal and fall back to original node index. Twins are interchangeable
        as producers, but a consumer referencing both twins in different slots
        can see its parent-position tuple swap under relabeling; this is
        accepted rather than paying for full graph canonization.
        """
        num_root_nodes = self.num_root_nodes
        output_start = num_root_nodes + self.num_trunk_nodes
        # Hoisted to locals: this is the one part of graph construction that
        # stays a sequential interpreted loop (a heap-ordered Kahn sweep has no
        # vectorized form), so attribute lookups in it are a real cost.
        inputs = self.node_inputs_indices
        node_types = self.node_types

        positions: list[int | None] = [None] * self.num_nodes
        for root_idx in range(num_root_nodes):
            positions[root_idx] = root_idx

        children: list[list[int]] = [[] for _ in range(output_start)]
        blocking_parents = [0] * output_start
        for node_idx in range(num_root_nodes, output_start):
            for parent in inputs[node_idx]:
                if parent >= num_root_nodes:
                    blocking_parents[node_idx] += 1
                    children[parent].append(node_idx)

        # The ready key is ``(parent positions in slot order, node type, index)``,
        # built inline at both sites below rather than in a helper: it is
        # evaluated once per trunk node and a call plus a generator expression
        # cost more than the heap operation it feeds. All parents are guaranteed
        # placed by the time a node enters the heap (roots are pre-placed; trunk
        # parents gate readiness), so no position is still None.
        heap = [
            (
                tuple([positions[parent] for parent in inputs[node_idx]]),
                node_types[node_idx],
                node_idx,
            )
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
                    heapq.heappush(
                        heap,
                        (
                            tuple([positions[parent] for parent in inputs[child]]),
                            node_types[child],
                            child,
                        ),
                    )

        assert len(order) == output_start, "canonical sort left unplaced trunk nodes"
        order.extend(range(output_start, self.num_nodes))
        return order

    def build_canonical_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Padded per-position CPU tensors of the canonical sequence overlay.

        Returns ``(order, node_types, parent_positions, parent_slot_mask)``:
        ``order`` is ``[N]`` long (sequence position -> original node index),
        ``node_types`` is ``[N]`` long in sequence order, ``parent_positions``
        is ``[N, maximum_indegree]`` long holding each valid slot's parent
        canonical position (0-padded), and ``parent_slot_mask`` is the matching
        ``[N, maximum_indegree]`` bool validity mask. Every valid parent
        position is strictly less than its consumer's position and strictly
        less than the first output position (outputs are leaves).
        """
        padded_parents, slot_mask = self.parent_arrays
        order = self.canonical_order_array
        positions = self.canonical_positions_array
        # One gather per tensor rather than a padded row built per node: writing
        # slots through Python lists (or worse, tensor ``__setitem__``) was one of
        # the largest single costs in graph generation. Invalid slots are held at
        # 0 explicitly; the mask is what marks them, but consumers still read the
        # value, so it must not be a stale parent position.
        parent_positions = np.where(slot_mask, positions[padded_parents], 0)[order]
        return (
            torch.from_numpy(order),
            torch.from_numpy(self.node_types_array[order]),
            torch.from_numpy(parent_positions),
            torch.from_numpy(slot_mask[order]),
        )


@dataclass
class CollatedRanks:
    """A batch of graphs' rank partitions, regrouped by rank across the batch.

    All row tensors are ``[total_rows]`` (or ``[total_rows, maximum_indegree]``)
    and share one ordering: rank-major, then by position in ``graphs``, then by
    node index. Rank ``r`` is therefore the contiguous slice of length
    ``counts[r]`` starting at ``offsets[r]``, which is what lets a rank-by-rank
    evaluation loop take slices instead of gathering.

    ``batch_indices`` says which graph each row came from, so a consumer can
    index a ``[batch, nodes]`` buffer with ``(batch_indices, node_indices)``.
    """

    batch_indices: torch.Tensor
    node_indices: torch.Tensor
    parent_indices: torch.Tensor
    valid_parent_mask: torch.Tensor
    subtypes: torch.Tensor
    counts: tuple[int, ...]
    offsets: tuple[int, ...]
    has_valid_parents: tuple[bool, ...]

    @property
    def num_ranks(self) -> int:
        return len(self.counts)

    def rank_slice(self, rank: int) -> slice:
        start = self.offsets[rank]
        return slice(start, start + self.counts[rank])


def collate_rank_partitions(
    graphs: Sequence["FixedInDegreeDAGDescription"],
    device: torch.device,
) -> CollatedRanks:
    """Regroup a batch of graphs' rank partitions by rank, on ``device``.

    Every rank-by-rank evaluation over a batch needs, for each rank, the rows of
    that rank from every graph. Each graph already stores its rows rank-major, so
    the batch ordering is one *stable* argsort of the concatenated rank labels:
    stability is what keeps graph order within a rank, and node order within a
    graph, without a second sort key.

    This replaced a nested loop that concatenated one tensor per graph per rank --
    five ``torch.cat`` calls over 256 slices, once per rank, every training step.
    At the training geometry that was 5.7 ms per batch of 256; the work here is a
    fixed handful of ops regardless of rank count.
    """
    if not graphs:
        raise ValueError("cannot collate an empty batch of graphs")

    partitions = [graph.rank_partition for graph in graphs]
    num_ranks = max(partition.num_ranks for partition in partitions)

    rank_of_row = torch.cat([partition.rank_of_row for partition in partitions])
    # Stable, so rows stay ordered by (rank, graph, node) after the permutation.
    order = torch.argsort(rank_of_row, stable=True)

    row_counts = [partition.node_indices.shape[0] for partition in partitions]
    batch_indices = torch.repeat_interleave(
        torch.arange(len(graphs), dtype=torch.long),
        torch.tensor(row_counts, dtype=torch.long),
    )

    counts = torch.bincount(rank_of_row, minlength=num_ranks).tolist()
    offsets = [0]
    for count in counts[:-1]:
        offsets.append(offsets[-1] + count)

    # A rank has valid parents if it does in any graph that reaches that rank.
    has_valid_parents = [False] * num_ranks
    for partition in partitions:
        for rank, valid in enumerate(partition.has_valid_parents):
            has_valid_parents[rank] = has_valid_parents[rank] or valid

    # Async host-to-device copies are only requested on CUDA. On MPS a
    # non-blocking copy of an index tensor can be read by a subsequent gather
    # before it lands, which silently yields out-of-range garbage rather than an
    # error; from unpinned host memory CUDA's non_blocking is synchronous anyway,
    # so nothing is lost by gating it.
    non_blocking = device.type == "cuda"

    def send(rows: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        return rows[permutation].to(device, non_blocking=non_blocking)

    return CollatedRanks(
        batch_indices=send(batch_indices, order),
        node_indices=send(
            torch.cat([partition.node_indices for partition in partitions]), order
        ),
        parent_indices=send(
            torch.cat([partition.parent_indices for partition in partitions]), order
        ),
        valid_parent_mask=send(
            torch.cat([partition.valid_parent_mask for partition in partitions]), order
        ),
        subtypes=send(
            torch.cat([partition.subtypes for partition in partitions]), order
        ),
        counts=tuple(counts),
        offsets=tuple(offsets),
        has_valid_parents=tuple(has_valid_parents),
    )


def make_random_graph_description(
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    trunk_node_in_degrees: int | list[int],
    num_trunk_node_types: int,
) -> FixedInDegreeDAGDescription:
    """One freshly sampled random DAG.

    A convenience wrapper over
    :func:`~dagnabbit.dag.generate.sample_graph_batch`, which is where the
    algorithm and its guarantees are documented. Anything generating more than
    one graph should call that directly: the compiled generator is entered once
    per call, so a batch of 256 costs one crossing rather than 256.
    """
    # Imported here rather than at module scope: dagnabbit.dag.generate needs
    # FixedInDegreeDAGDescription, so a top-level import would be circular.
    from dagnabbit.dag.generate import sample_graph_batch

    return sample_graph_batch(
        1,
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        num_trunk_node_types=num_trunk_node_types,
        trunk_node_in_degrees=trunk_node_in_degrees,
    )[0]


def canonicalize(graph: FixedInDegreeDAGDescription) -> list[tuple]:
    """Return bottom-up structural ids for every node in topological order."""
    canonical_ids: list[tuple] = []
    memo: dict[tuple, tuple] = {}

    def intern(key: tuple) -> tuple:
        existing = memo.get(key)
        if existing is not None:
            return existing
        memo[key] = key
        return key

    output_start = graph.num_root_nodes + graph.num_trunk_nodes
    for node_idx, node_type in enumerate(graph.node_types):
        supertype = subtype_to_supertype(
            node_type,
            num_trunk_node_types=graph.num_trunk_node_types,
            num_root_nodes=graph.num_root_nodes,
        )
        if supertype is NodeSupertype.ROOT:
            root_slot = node_type - graph.root_node_types_start
            canonical_ids.append(intern(("root", root_slot)))
        elif supertype is NodeSupertype.TRUNK:
            parent_ids = tuple(
                canonical_ids[p] for p in graph.node_inputs_indices[node_idx]
            )
            canonical_ids.append(intern(("trunk", node_type, parent_ids)))
        else:
            # Outputs share one type; slot identity comes from fixed position.
            output_slot = node_idx - output_start
            parent_ids = tuple(
                canonical_ids[p] for p in graph.node_inputs_indices[node_idx]
            )
            canonical_ids.append(intern(("output", output_slot, parent_ids)))

    return canonical_ids


def graphs_match(
    a: FixedInDegreeDAGDescription,
    b: FixedInDegreeDAGDescription,
) -> bool:
    """Compare graphs by ordered output canonical ids, ignoring dead nodes."""
    if a.num_output_nodes != b.num_output_nodes:
        return False

    a_ids = canonicalize(a)
    b_ids = canonicalize(b)
    a_output_start = a.num_root_nodes + a.num_trunk_nodes
    b_output_start = b.num_root_nodes + b.num_trunk_nodes
    return tuple(a_ids[a_output_start : a_output_start + a.num_output_nodes]) == tuple(
        b_ids[b_output_start : b_output_start + b.num_output_nodes]
    )
