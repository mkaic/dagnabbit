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
        # node, and the per-rank groups iterated over during grouped evaluation.
        self.node_ranks = self.compute_node_ranks()
        self.rank_groups = self.build_rank_groups()
        self.rank_batches = self.build_rank_batches()

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

        ``rank_groups`` keeps the semantic grouping for diagnostics and future
        batching strategies. The training loop needs a node-sorted padded view
        of each rank every encode/decode pass, so build that once when the graph
        is created instead of rebuilding it on the compute device.
        """
        rank_batches: list[PreparedRankBatch] = []
        for groups in self.rank_groups:
            if not groups:
                empty = torch.empty(0, dtype=torch.long)
                rank_batches.append(
                    PreparedRankBatch(
                        node_indices=empty,
                        parent_indices=torch.empty(
                            0,
                            self.maximum_indegree,
                            dtype=torch.long,
                        ),
                        valid_parent_mask=torch.empty(
                            0,
                            self.maximum_indegree,
                            dtype=torch.bool,
                        ),
                        subtypes=empty,
                    )
                )
                continue

            node_indices = torch.cat([g.node_buffer_indices for g in groups])
            subtypes = torch.cat([g.subtypes for g in groups])
            parent_indices = torch.zeros(
                node_indices.shape[0],
                self.maximum_indegree,
                dtype=torch.long,
            )
            valid_parent_mask = torch.zeros(
                node_indices.shape[0],
                self.maximum_indegree,
                dtype=torch.bool,
            )

            offset = 0
            has_valid_parents = False
            for group in groups:
                group_parents = group.parent_buffer_gather_indices
                group_size, in_degree = group_parents.shape
                if in_degree > self.maximum_indegree:
                    raise ValueError(
                        f"rank contains in-degree {in_degree}, above maximum "
                        f"{self.maximum_indegree}"
                    )
                if in_degree and group_size:
                    parent_indices[offset : offset + group_size, :in_degree] = (
                        group_parents
                    )
                    valid_parent_mask[offset : offset + group_size, :in_degree] = True
                    has_valid_parents = True
                offset += group_size

            order = node_indices.argsort()
            rank_batches.append(
                PreparedRankBatch(
                    node_indices=node_indices[order],
                    parent_indices=parent_indices[order],
                    valid_parent_mask=valid_parent_mask[order],
                    subtypes=subtypes[order],
                    has_valid_parents=has_valid_parents,
                )
            )

        return rank_batches


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

    # Input slots per node; ``None`` marks an unfilled slot. Roots have none.
    parents: list[list[int | None]] = [
        [None] * in_degrees[node_idx] for node_idx in range(num_nodes)
    ]
    # A producer (root or trunk) is "used" once some consumer slot points at it.
    has_child = [False] * num_nodes

    def open_slot(consumer: int) -> int | None:
        for slot, value in enumerate(parents[consumer]):
            if value is None:
                return slot
        return None

    # Pass 1 -- coverage. Producers (roots + trunks) processed latest-first.
    for producer in range(output_start - 1, -1, -1):
        if has_child[producer]:
            continue
        candidates = [
            consumer
            for consumer in range(producer + 1, num_nodes)
            if open_slot(consumer) is not None and producer not in parents[consumer]
        ]
        # The feasibility precondition guarantees a later open slot always
        # exists here; this is a defensive safety net.
        assert candidates, f"coverage failed for producer {producer} (internal bug)"
        consumer = rng.choice(candidates)
        slot = open_slot(consumer)
        assert slot is not None
        parents[consumer][slot] = producer
        has_child[producer] = True

    # Pass 2 -- fill every remaining slot with a random earlier producer.
    for consumer in range(num_root_nodes, num_nodes):
        for slot in range(len(parents[consumer])):
            if parents[consumer][slot] is not None:
                continue
            existing = {value for value in parents[consumer] if value is not None}
            # Every node before ``output_start`` is a producer, and a consumer
            # may only reference strictly earlier nodes.
            pool = [
                candidate
                for candidate in range(min(consumer, output_start))
                if candidate not in existing
            ]
            chosen = rng.choice(pool)
            parents[consumer][slot] = chosen
            has_child[chosen] = True

    assert all(has_child[p] for p in range(output_start)), "uncovered producer remains"

    # Shuffle each consumer's parent order to erase the slot-position artifact
    # left by the two passes, then hand the flat arrays to the representation.
    node_inputs_indices: list[list[int]] = []
    for node_idx in range(num_nodes):
        slot_values = parents[node_idx]
        assert all(value is not None for value in slot_values)
        ordered_parents: list[int] = [
            value for value in slot_values if value is not None
        ]
        if node_idx >= num_root_nodes:
            rng.shuffle(ordered_parents)
        node_inputs_indices.append(ordered_parents)

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
