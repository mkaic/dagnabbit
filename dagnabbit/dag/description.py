import functools
import heapq
import random
from dataclasses import dataclass
from enum import Enum

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
class RankGroup:
    """A batch of nodes that share a topological rank and a batch key.

    All nodes in a group are evaluated together in a single encoder/decoder MLP
    call. Nodes are grouped by :class:`NodeSupertype`, and ``TRUNK`` nodes are
    grouped further by their (trunk-local) subtype, since each trunk type has
    its own module and in-degree. Every node in a group therefore shares the
    same in-degree, so ``parent_buffer_gather_indices`` is a rectangular
    ``[G, in_degree]`` tensor, and all ``subtypes`` are identical within a
    ``TRUNK`` group. Use ``group.subtypes`` to look up the per-node module on
    the model.

    All tensors are stored on CPU at generation time; the model moves them to
    the compute device once.
    """

    supertype: NodeSupertype
    # Global buffer indices of the nodes in this group: LongTensor [G].
    node_buffer_indices: torch.Tensor
    # Buffer positions of each node's ordered parents: LongTensor [G, in_degree].
    parent_buffer_gather_indices: torch.Tensor
    # Raw node_type subtype index of each node: LongTensor [G].
    subtypes: torch.Tensor


@dataclass
class PreparedRankBatch:
    """CPU-side padded tensors for one topological rank of one graph."""

    node_indices: torch.Tensor
    parent_indices: torch.Tensor
    valid_parent_mask: torch.Tensor
    subtypes: torch.Tensor
    has_valid_parents: bool = False


class FixedInDegreeDAGDescription:
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
        if isinstance(trunk_node_in_degrees, int):
            trunk_node_in_degrees = [trunk_node_in_degrees] * num_trunk_node_types

        assert len(trunk_node_in_degrees) == num_trunk_node_types

        self.node_inputs_indices = node_inputs_indices
        self.node_types = node_types
        self.node_types_tensor = torch.tensor(self.node_types, dtype=torch.long)
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

        assert len(self.node_types) == self.num_nodes

        for i in range(num_root_nodes):
            assert len(self.node_inputs_indices[i]) == 0
            assert self.node_types[i] == self.root_node_types_start + i

        for i in range(num_trunk_nodes):
            node_idx = num_root_nodes + i
            trunk_type = self.node_types[node_idx]
            assert 0 <= trunk_type < num_trunk_node_types
            expected = self.trunk_node_in_degrees[trunk_type]
            assert len(self.node_inputs_indices[node_idx]) == expected

        for i in range(num_output_nodes):
            node_idx = num_root_nodes + num_trunk_nodes + i
            assert len(self.node_inputs_indices[node_idx]) == 1
            assert self.node_types[node_idx] == self.output_node_types_start

        self.leaf_node_indices = self.identify_leaf_nodes()
        self.leaf_node_indices_tensor = torch.tensor(
            self.leaf_node_indices,
            dtype=torch.long,
        )

        # Batch overlay over the (unchanged) flat arrays: longest-path rank per
        # node, and the per-rank padded tensors iterated over during grouped
        # evaluation. ``rank_groups`` is the semantic (supertype/subtype) view of
        # the same partition; nothing on the hot path reads it, so it is built
        # lazily on first access rather than for every generated graph.
        self.node_ranks = self.compute_node_ranks()
        self.rank_batches = self.build_rank_batches()

        # Canonical sequence overlay: structure-derived topological order with
        # roots pinned first and outputs pinned last, plus padded per-position
        # parent tensors consumed by the sequence compressor/decoder.
        self.canonical_order = self.compute_canonical_order()
        self.canonical_positions = [0] * self.num_nodes
        for position, node_idx in enumerate(self.canonical_order):
            self.canonical_positions[node_idx] = position
        # Node-storage-index -> canonical position, as a tensor: the recursive
        # encoder gathers through it to give every node's context tokens the
        # absolute graph positions the pointer head predicts into.
        self.canonical_positions_tensor = torch.tensor(
            self.canonical_positions,
            dtype=torch.long,
        )
        (
            self.canonical_order_tensor,
            self.canonical_node_types,
            self.canonical_parent_positions,
            self.canonical_parent_slot_mask,
        ) = self.build_canonical_tensors()

    def identify_leaf_nodes(self) -> list[int]:
        """
        Identify all leaf nodes in the DAG.

        A leaf node is a node whose output is not referenced as an input to any
        other node. Output nodes are guaranteed to be leaves. Returns array
        indices of all leaf nodes as a sorted list of integers.
        """
        referenced = [False] * self.num_nodes
        for inputs in self.node_inputs_indices:
            for parent in inputs:
                referenced[parent] = True
        return [n for n, is_referenced in enumerate(referenced) if not is_referenced]

    def compute_node_ranks(self) -> list[int]:
        """Longest-path depth of each node (roots are rank 0).

        Every other node's rank is ``1 + max(parent ranks)``. Nodes are stored
        in topological order (roots first, every trunk/output node only
        references earlier indices), so a single forward sweep suffices.
        """
        ranks = [0] * self.num_nodes
        for node_idx in range(self.num_root_nodes, self.num_nodes):
            parent_rank = 0
            for parent in self.node_inputs_indices[node_idx]:
                parent_rank = max(parent_rank, ranks[parent])
            ranks[node_idx] = parent_rank + 1
        return ranks

    @functools.cached_property
    def rank_groups(self) -> list[list[RankGroup]]:
        """Semantic per-rank grouping, materialized on first access."""
        return self.build_rank_groups()

    def build_rank_groups(self) -> list[list[RankGroup]]:
        """Group nodes by topological rank and batch key into :class:`RankGroup`s.

        Returns a list indexed by rank (ascending). Each entry is a list of
        groups at that rank, one per batch key, in first-appearance node order.
        The batch key is the node's :class:`NodeSupertype`, refined by
        trunk-local subtype for ``TRUNK`` nodes. Roots (rank 0) form a
        degenerate group with an empty ``[G, 0]`` ``parent_buffer_gather_indices``
        and no MLP.
        """
        max_rank = max(self.node_ranks, default=0)

        # Single pass over all nodes, bucketing into nested dicts indexed by rank
        # then batch key. dict preserves insertion order (Python 3.7+), so both
        # the ranks (visited ascending) and the groups within each rank come out
        # in first-appearance node order without any separate ordering lists.
        nodes_by_rank_and_key: list[dict[tuple, list[int]]] = [
            {} for _ in range(max_rank + 1)
        ]
        output_start = self.num_root_nodes + self.num_trunk_nodes
        for node_idx, rank in enumerate(self.node_ranks):
            if node_idx < self.num_root_nodes:
                supertype = NodeSupertype.ROOT
                trunk_subtype = None
            elif node_idx < output_start:
                supertype = NodeSupertype.TRUNK
                # Trunk types live at the start of the type space, so the raw
                # subtype is already the trunk-local subtype; use it directly to
                # split trunk groups (each trunk type has its own module +
                # in-degree). Roots and outputs collapse into one group per
                # supertype.
                trunk_subtype = self.node_types[node_idx]
            else:
                supertype = NodeSupertype.OUTPUT
                trunk_subtype = None
            key = (supertype, trunk_subtype)
            nodes_by_rank_and_key[rank].setdefault(key, []).append(node_idx)

        rank_groups: list[list[RankGroup]] = []
        for grouped_node_indices in nodes_by_rank_and_key:
            groups: list[RankGroup] = []
            for (supertype, _trunk_subtype), node_list in grouped_node_indices.items():
                parent_lists = [self.node_inputs_indices[n] for n in node_list]

                # Every node in a group shares an in-degree, so the parent gather
                # is rectangular [G, in_degree] (empty second dim for roots).
                in_degree = len(parent_lists[0])
                assert all(len(p) == in_degree for p in parent_lists)

                node_buffer_indices = torch.tensor(node_list, dtype=torch.long)
                parent_buffer_gather_indices = torch.tensor(
                    parent_lists, dtype=torch.long
                ).reshape(len(node_list), in_degree)
                subtypes = torch.tensor(
                    [self.node_types[n] for n in node_list], dtype=torch.long
                )

                groups.append(
                    RankGroup(
                        supertype=supertype,
                        node_buffer_indices=node_buffer_indices,
                        parent_buffer_gather_indices=parent_buffer_gather_indices,
                        subtypes=subtypes,
                    )
                )

            rank_groups.append(groups)

        return rank_groups

    def build_rank_batches(self) -> list[PreparedRankBatch]:
        """Precompute padded CPU rank tensors used by the model hot path.

        The training loop needs a node-sorted padded view of each rank every
        encode/decode pass, so build that once when the graph is created instead
        of rebuilding it on the compute device.

        This runs once per generated graph in the training loop, so it builds
        the padded rows for *every* rank as flat Python lists, converts them in
        four ``torch.tensor`` calls total, and hands each rank a slice. Per-rank
        (or per-:class:`RankGroup`) tensor construction costs tens of
        microseconds of allocator/dispatch overhead each and dominated graph
        generation when done that way.
        """
        max_rank = max(self.node_ranks, default=0)
        nodes_by_rank: list[list[int]] = [[] for _ in range(max_rank + 1)]
        for node_idx, rank in enumerate(self.node_ranks):
            nodes_by_rank[rank].append(node_idx)

        pad = self.maximum_indegree
        flat_nodes: list[int] = []
        flat_subtypes: list[int] = []
        flat_parents: list[list[int]] = []
        flat_mask: list[list[bool]] = []
        rank_has_valid_parents: list[bool] = []

        # Nodes are visited in ascending index order above, so each rank's slice
        # is already node-sorted -- what the old build did with an argsort.
        for node_list in nodes_by_rank:
            has_valid_parents = False
            for node_idx in node_list:
                parents = self.node_inputs_indices[node_idx]
                in_degree = len(parents)
                if in_degree > pad:
                    raise ValueError(
                        f"rank contains in-degree {in_degree}, above maximum {pad}"
                    )
                has_valid_parents = has_valid_parents or in_degree > 0
                flat_nodes.append(node_idx)
                flat_subtypes.append(self.node_types[node_idx])
                flat_parents.append([*parents, *(0,) * (pad - in_degree)])
                flat_mask.append([*(True,) * in_degree, *(False,) * (pad - in_degree)])
            rank_has_valid_parents.append(has_valid_parents)

        total = len(flat_nodes)
        all_nodes = torch.tensor(flat_nodes, dtype=torch.long)
        all_subtypes = torch.tensor(flat_subtypes, dtype=torch.long)
        all_parents = torch.tensor(flat_parents, dtype=torch.long).reshape(total, pad)
        all_mask = torch.tensor(flat_mask, dtype=torch.bool).reshape(total, pad)

        rank_batches: list[PreparedRankBatch] = []
        offset = 0
        for node_list, has_valid_parents in zip(nodes_by_rank, rank_has_valid_parents):
            start = offset
            offset += len(node_list)
            end = offset
            rank_batches.append(
                PreparedRankBatch(
                    node_indices=all_nodes[start:end],
                    parent_indices=all_parents[start:end],
                    valid_parent_mask=all_mask[start:end],
                    subtypes=all_subtypes[start:end],
                    has_valid_parents=has_valid_parents,
                )
            )

        return rank_batches

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
        output_start = self.num_root_nodes + self.num_trunk_nodes
        positions: list[int | None] = [None] * self.num_nodes
        for root_idx in range(self.num_root_nodes):
            positions[root_idx] = root_idx

        children: list[list[int]] = [[] for _ in range(output_start)]
        blocking_parents = [0] * output_start
        for node_idx in range(self.num_root_nodes, output_start):
            for parent in self.node_inputs_indices[node_idx]:
                if parent >= self.num_root_nodes:
                    blocking_parents[node_idx] += 1
                    children[parent].append(node_idx)

        def ready_key(node_idx: int) -> tuple:
            # All parents are guaranteed placed by the time a node enters the
            # heap (roots are pre-placed; trunk parents gate readiness).
            parent_positions = tuple(
                positions[parent] for parent in self.node_inputs_indices[node_idx]
            )
            return (parent_positions, self.node_types[node_idx], node_idx)

        heap = [
            ready_key(node_idx)
            for node_idx in range(self.num_root_nodes, output_start)
            if blocking_parents[node_idx] == 0
        ]
        heapq.heapify(heap)

        order = list(range(self.num_root_nodes))
        while heap:
            node_idx = heapq.heappop(heap)[-1]
            positions[node_idx] = len(order)
            order.append(node_idx)
            for child in children[node_idx]:
                blocking_parents[child] -= 1
                if blocking_parents[child] == 0:
                    heapq.heappush(heap, ready_key(child))

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
        pad = self.maximum_indegree
        canonical_positions = self.canonical_positions
        node_types_list: list[int] = []
        position_rows: list[list[int]] = []
        mask_rows: list[list[bool]] = []
        # Build the padded rows in Python and convert once. Writing each slot
        # through tensor ``__setitem__`` costs a dispatch per element and was one
        # of the largest single costs in graph generation.
        for node_idx in self.canonical_order:
            node_types_list.append(self.node_types[node_idx])
            parents = self.node_inputs_indices[node_idx]
            in_degree = len(parents)
            position_rows.append(
                [
                    *(canonical_positions[parent] for parent in parents),
                    *(0,) * (pad - in_degree),
                ]
            )
            mask_rows.append([*(True,) * in_degree, *(False,) * (pad - in_degree)])

        order = torch.tensor(self.canonical_order, dtype=torch.long)
        node_types = torch.tensor(node_types_list, dtype=torch.long)
        parent_positions = torch.tensor(position_rows, dtype=torch.long).reshape(
            self.num_nodes, pad
        )
        parent_slot_mask = torch.tensor(mask_rows, dtype=torch.bool).reshape(
            self.num_nodes, pad
        )
        return order, node_types, parent_positions, parent_slot_mask


def make_random_graph_description(
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    trunk_node_in_degrees: int | list[int],
    num_trunk_node_types: int,
) -> FixedInDegreeDAGDescription:
    """Generate a random fixed-in-degree DAG via two-pass construction (Algorithm A).

    1. Lay nodes out in topological order: ``num_root_nodes`` roots (in-degree
       0), then ``num_trunk_nodes`` trunk nodes each with a random type (its
       type fixes its in-degree), then ``num_output_nodes`` outputs (in-degree
       1). A consumer's input slots may only point at strictly earlier nodes, so
       the graph is acyclic by construction.
    2. **Coverage pass** (producers walked latest-first): every root and trunk
       must end up with at least one child. Each still-childless producer is
       wired into one open input slot of a randomly chosen *later* consumer.
       Walking latest-first claims the scarce late slots before the plentiful
       early ones.
    3. **Fill pass**: every still-open input slot is filled with a random
       *earlier* producer, distinct from that consumer's existing parents.

    Because the only sinks are outputs, "every trunk is an ancestor of some
    output" is equivalent to "every trunk has a child", which the coverage pass
    enforces locally. Full coverage of *all* producers (roots included) is
    guaranteed for any random type assignment as long as there are enough
    downstream slots::

        num_root_nodes <= num_trunk_nodes * (min_in_degree - 1) + num_output_nodes

    (the worst case is every trunk taking the smallest in-degree). Since a
    producer can only be consumed by strictly-later nodes, the producer->slot
    neighbourhoods are nested, so the latest-first greedy assignment saturates
    whenever that inequality holds; it is asserted up front. The resulting
    distribution is the natural generative model (not uniform over all such
    DAGs), but it is O(edges) and respects every count / in-degree / coverage
    constraint exactly.
    """
    if isinstance(trunk_node_in_degrees, int):
        trunk_node_in_degrees = [trunk_node_in_degrees] * num_trunk_node_types
    else:
        trunk_node_in_degrees = list(trunk_node_in_degrees)

    assert len(trunk_node_in_degrees) == num_trunk_node_types
    assert all(in_degree >= 1 for in_degree in trunk_node_in_degrees)
    assert num_root_nodes >= 1
    assert num_output_nodes >= 1
    assert num_trunk_node_types >= 1
    if num_trunk_nodes > 0:
        # The earliest trunk can draw distinct inputs only from the roots, so
        # there must be at least max-in-degree of them for every gate to be
        # given distinct parents.
        assert num_root_nodes >= max(trunk_node_in_degrees), (
            "num_root_nodes must be >= the largest trunk in-degree so every "
            "gate can be given distinct inputs"
        )

    # Coverage feasibility: with every trunk at the smallest possible in-degree,
    # the downstream input slots must still outnumber the producers enough to
    # give each root (and trunk) a child. See the docstring for the derivation.
    min_in_degree = min(trunk_node_in_degrees)
    max_coverable_roots = num_trunk_nodes * (min_in_degree - 1) + num_output_nodes
    assert num_root_nodes <= max_coverable_roots, (
        f"to guarantee every root is used, need num_root_nodes "
        f"({num_root_nodes}) <= num_trunk_nodes * (min_in_degree - 1) + "
        f"num_output_nodes ({num_trunk_nodes} * {min_in_degree - 1} + "
        f"{num_output_nodes} = {max_coverable_roots}); add trunk nodes "
        "(ideally with in-degree > 1) or output nodes"
    )

    seed = int(torch.randint(0, 2**63 - 1, (1,), dtype=torch.int64).item())
    rng = random.Random(seed)

    num_nodes = num_root_nodes + num_trunk_nodes + num_output_nodes
    output_start = num_root_nodes + num_trunk_nodes

    # Node types and per-node input-slot count (the node's in-degree).
    node_types = [0] * num_nodes
    in_degrees = [0] * num_nodes
    for root_idx in range(num_root_nodes):
        # root_node_types_start (== num_trunk_node_types) + slot.
        node_types[root_idx] = num_trunk_node_types + root_idx
    for trunk_offset in range(num_trunk_nodes):
        node_idx = num_root_nodes + trunk_offset
        trunk_type = rng.randrange(num_trunk_node_types)
        node_types[node_idx] = trunk_type
        in_degrees[node_idx] = trunk_node_in_degrees[trunk_type]
    for output_offset in range(num_output_nodes):
        node_idx = output_start + output_offset
        # All outputs share the single output type (output_node_types_start).
        node_types[node_idx] = num_trunk_node_types + num_root_nodes
        in_degrees[node_idx] = 1

    # Filled input slots per node, appended to as the two passes run. Slot order
    # here is an artifact of the passes and is shuffled away at the end, so both
    # passes can simply append rather than hunt for the first open slot.
    parents: list[list[int]] = [[] for _ in range(num_nodes)]
    # A producer (root or trunk) is "used" once some consumer slot points at it.
    has_child = [False] * num_nodes

    # Pass 1 -- coverage. Producers (roots + trunks) processed latest-first.
    #
    # ``coverable`` holds exactly the consumers eligible for the producer being
    # processed: strictly later than it, and still holding an open slot. It is
    # maintained incrementally instead of rescanning every later node per
    # producer, which made this pass quadratic in the node count. Walking
    # producers latest-first is what makes the incremental maintenance possible:
    # the "strictly later" set only ever grows, one node per step. ``slot_of``
    # tracks each consumer's index in ``coverable`` so a consumer whose slots
    # just filled can be swap-removed in constant time.
    #
    # A producer is chosen at most once here and has never been assigned before
    # its own turn, so a consumer can never already point at it: no duplicate
    # check is needed, and every producer gets covered exactly once.
    coverable: list[int] = []
    slot_of = [-1] * num_nodes

    def mark_coverable(consumer: int) -> None:
        if in_degrees[consumer]:
            slot_of[consumer] = len(coverable)
            coverable.append(consumer)

    def drop_coverable(consumer: int) -> None:
        index = slot_of[consumer]
        moved = coverable.pop()
        if moved != consumer:
            coverable[index] = moved
            slot_of[moved] = index
        slot_of[consumer] = -1

    for consumer in range(output_start, num_nodes):
        mark_coverable(consumer)

    for producer in range(output_start - 1, -1, -1):
        # The feasibility precondition guarantees a later open slot always
        # exists here; this is a defensive safety net.
        assert coverable, f"coverage failed for producer {producer} (internal bug)"
        consumer = coverable[rng.randrange(len(coverable))]
        parents[consumer].append(producer)
        if len(parents[consumer]) == in_degrees[consumer]:
            drop_coverable(consumer)
        has_child[producer] = True
        # The next producer is earlier, so this one becomes a legal consumer.
        mark_coverable(producer)

    # Pass 2 -- fill every remaining slot with a random earlier producer.
    # Every node before ``output_start`` is a producer, and a consumer may only
    # reference strictly earlier nodes, so the eligible range is a contiguous
    # ``[0, candidate_range)``. Rejection-sample from it rather than materializing
    # the filtered pool per slot: ``existing`` holds at most in_degree - 1
    # entries and the range is at least ``num_root_nodes >= max in-degree`` wide,
    # so a draw is accepted with probability > 0 and in practice on the first try.
    for consumer in range(num_root_nodes, num_nodes):
        filled = parents[consumer]
        existing = set(filled)
        candidate_range = min(consumer, output_start)
        while len(filled) < in_degrees[consumer]:
            chosen = rng.randrange(candidate_range)
            while chosen in existing:
                chosen = rng.randrange(candidate_range)
            filled.append(chosen)
            existing.add(chosen)
            has_child[chosen] = True

    assert all(has_child[p] for p in range(output_start)), "uncovered producer remains"

    # Shuffle each consumer's parent order to erase the slot-position artifact
    # left by the two passes, then hand the flat arrays to the representation.
    node_inputs_indices = parents
    for node_idx in range(num_root_nodes, num_nodes):
        rng.shuffle(node_inputs_indices[node_idx])

    return FixedInDegreeDAGDescription(
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        num_trunk_node_types=num_trunk_node_types,
        trunk_node_in_degrees=trunk_node_in_degrees,
        node_inputs_indices=node_inputs_indices,
        node_types=node_types,
    )


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
