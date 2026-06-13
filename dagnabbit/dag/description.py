import random
from collections import deque
from dataclasses import dataclass
from enum import Enum

import torch

try:
    import dagnabbit._native as _native
except ImportError:
    _native = None

NodeInputIndices = list[int]
DEFAULT_MCMC_PASSES_PER_SPLIT_ROUND = 8


class NodeSupertype(Enum):
    """Module-level role of a node, used as a batch key for grouped evaluation."""

    ROOT = "root"
    TRUNK = "trunk"
    OUTPUT = "output"


def subtype_to_supertype(
    node_type: int,
    num_trunk_node_types: int | None = None,
    num_root_nodes: int | None = None,
    num_output_nodes: int | None = None,
) -> NodeSupertype:
    """Map a raw ``node_type`` subtype index to its :class:`NodeSupertype`.

    Uses the unified type-index layout:
        [0, num_trunk_node_types)                        -> TRUNK
        [root_node_types_start, output_node_types_start) -> ROOT
        [output_node_types_start, num_node_types)        -> OUTPUT

    Any layout argument left as ``None`` falls back to the corresponding value
    in ``dagnabbit.scripts.config``, so training callsites can omit them.
    """
    if (
        num_trunk_node_types is None
        or num_root_nodes is None
        or num_output_nodes is None
    ):
        from dagnabbit.scripts import config as cfg

        if num_trunk_node_types is None:
            num_trunk_node_types = cfg.NUM_TRUNK_NODE_TYPES
        if num_root_nodes is None:
            num_root_nodes = cfg.NUM_ROOT_NODES
        if num_output_nodes is None:
            num_output_nodes = cfg.NUM_OUTPUT_NODES

    root_node_types_start = num_trunk_node_types
    output_node_types_start = num_trunk_node_types + num_root_nodes
    num_node_types = num_trunk_node_types + num_root_nodes + num_output_nodes

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
        self.num_trunk_nodes = num_trunk_nodes
        self.num_root_nodes = num_root_nodes
        self.num_output_nodes = num_output_nodes
        self.num_trunk_node_types = num_trunk_node_types
        self.trunk_node_in_degrees = trunk_node_in_degrees
        self.num_nodes = num_root_nodes + num_trunk_nodes + num_output_nodes

        # Each root and output gets its own unique type index, laid out in a
        # single type-index space immediately after the trunk types:
        #   [0, num_trunk_node_types)                                    -> trunk types
        #   [root_node_types_start, output_node_types_start)             -> one type per root slot
        #   [output_node_types_start, num_node_types)                    -> one type per output slot
        self.root_node_types_start = num_trunk_node_types
        self.output_node_types_start = num_trunk_node_types + num_root_nodes
        self.num_node_types = num_trunk_node_types + num_root_nodes + num_output_nodes

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
            assert self.node_types[node_idx] == self.output_node_types_start + i

        self.leaf_node_indices = self.identify_leaf_nodes()

        # Batch overlay over the (unchanged) flat arrays: longest-path rank per
        # node, and the per-rank groups iterated over during grouped evaluation.
        self.node_ranks = self.compute_node_ranks()
        self.rank_groups = self.build_rank_groups()

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


class _EdgeSplitRole(Enum):
    ROOT = "root"
    TRUNK = "trunk"
    LEAF = "leaf"


@dataclass
class _EdgeSplitGraph:
    parents: list[list[int]]
    children: list[list[int]]
    roles: list[_EdgeSplitRole]
    trunk_types: dict[int, int]


def _can_reach(children: list[list[int]], src: int, dst: int) -> bool:
    if src == dst:
        return True

    seen: set[int] = set()
    stack = list(children[src])
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(children[node])
    return False


def _can_reach_with_scratch(
    children: list[list[int]],
    src: int,
    dst: int,
    seen: list[int],
    seen_token: int,
    stack: list[int],
) -> bool:
    if src == dst:
        return True

    stack.clear()
    stack.extend(children[src])
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if seen[node] == seen_token:
            continue
        seen[node] = seen_token
        stack.extend(children[node])
    return False


def _initialize_edge_split_graph(
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    rng: random.Random,
) -> _EdgeSplitGraph:
    num_nodes = num_root_nodes + num_trunk_nodes + num_output_nodes
    parents: list[list[int]] = [[] for _ in range(num_nodes)]
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    roles = [_EdgeSplitRole.TRUNK for _ in range(num_nodes)]

    for root in range(num_root_nodes):
        roles[root] = _EdgeSplitRole.ROOT

    leaf_start = num_root_nodes
    trunk_start = num_root_nodes + num_output_nodes
    for leaf in range(leaf_start, trunk_start):
        roles[leaf] = _EdgeSplitRole.LEAF
        root = rng.randrange(num_root_nodes)
        parents[leaf] = [root]
        children[root].append(leaf)

    return _EdgeSplitGraph(
        parents=parents,
        children=children,
        roles=roles,
        trunk_types={},
    )


def _active_nodes(
    num_root_nodes: int,
    num_output_nodes: int,
    inserted_trunks: int,
) -> list[int]:
    trunk_start = num_root_nodes + num_output_nodes
    return [
        *range(num_root_nodes),
        *range(num_root_nodes, trunk_start),
        *range(trunk_start, trunk_start + inserted_trunks),
    ]


def _active_nonleaf_nodes(
    num_root_nodes: int,
    num_output_nodes: int,
    inserted_trunks: int,
) -> list[int]:
    trunk_start = num_root_nodes + num_output_nodes
    return [
        *range(num_root_nodes),
        *range(trunk_start, trunk_start + inserted_trunks),
    ]


def _active_edges(
    graph: _EdgeSplitGraph,
    active_nodes: list[int],
) -> list[tuple[int, int]]:
    active = set(active_nodes)
    return [
        (parent, child)
        for parent in active_nodes
        for child in graph.children[parent]
        if child in active
    ]


def _sample_valid_parent_by_reachability(
    graph: _EdgeSplitGraph,
    eligible: list[int],
    forbidden_node: int,
    rng: random.Random,
) -> int:
    for _ in range(max(8, len(eligible) * 2)):
        candidate = rng.choice(eligible)
        if not _can_reach(graph.children, forbidden_node, candidate):
            return candidate

    valid = [
        candidate
        for candidate in eligible
        if not _can_reach(graph.children, forbidden_node, candidate)
    ]
    assert valid
    return rng.choice(valid)


def _split_active_edge(
    graph: _EdgeSplitGraph,
    edge: tuple[int, int],
    inserted_trunks: int,
    num_root_nodes: int,
    num_output_nodes: int,
    num_trunk_node_types: int,
    trunk_node_in_degrees: list[int],
    rng: random.Random,
) -> None:
    u, v = edge
    trunk_start = num_root_nodes + num_output_nodes
    w = trunk_start + inserted_trunks

    trunk_type = rng.randrange(num_trunk_node_types)
    graph.trunk_types[w] = trunk_type
    in_degree = trunk_node_in_degrees[trunk_type]
    eligible = _active_nonleaf_nodes(
        num_root_nodes=num_root_nodes,
        num_output_nodes=num_output_nodes,
        inserted_trunks=inserted_trunks,
    )
    extras = [
        _sample_valid_parent_by_reachability(
            graph=graph,
            eligible=eligible,
            forbidden_node=v,
            rng=rng,
        )
        for _ in range(in_degree - 1)
    ]

    graph.children[u].remove(v)
    graph.children[u].append(w)
    graph.parents[v][graph.parents[v].index(u)] = w
    graph.parents[w] = [u, *extras]
    graph.children[w] = [v]
    for extra in extras:
        graph.children[extra].append(w)


def _build_interleaved_edge_split_graph(
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    num_trunk_node_types: int,
    trunk_node_in_degrees: list[int],
    mcmc_passes_per_split_round: int,
    rng: random.Random,
) -> _EdgeSplitGraph:
    raw = _initialize_edge_split_graph(
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        rng=rng,
    )

    inserted_trunks = 0
    while inserted_trunks < num_trunk_nodes:
        active = _active_nodes(
            num_root_nodes=num_root_nodes,
            num_output_nodes=num_output_nodes,
            inserted_trunks=inserted_trunks,
        )
        edges = _active_edges(raw, active)
        assert edges
        rng.shuffle(edges)

        for edge in edges:
            if inserted_trunks >= num_trunk_nodes:
                break
            _split_active_edge(
                graph=raw,
                edge=edge,
                inserted_trunks=inserted_trunks,
                num_root_nodes=num_root_nodes,
                num_output_nodes=num_output_nodes,
                num_trunk_node_types=num_trunk_node_types,
                trunk_node_in_degrees=trunk_node_in_degrees,
                rng=rng,
            )
            inserted_trunks += 1

        active = _active_nodes(
            num_root_nodes=num_root_nodes,
            num_output_nodes=num_output_nodes,
            inserted_trunks=inserted_trunks,
        )
        _mcmc_rewire(
            raw,
            num_steps=mcmc_passes_per_split_round * len(active),
            rng=rng,
            active_nodes=active,
        )

    if num_trunk_nodes == 0:
        active = _active_nodes(
            num_root_nodes=num_root_nodes,
            num_output_nodes=num_output_nodes,
            inserted_trunks=0,
        )
        _mcmc_rewire(
            raw,
            num_steps=mcmc_passes_per_split_round * len(active),
            rng=rng,
            active_nodes=active,
        )

    return raw


def _mcmc_rewire(
    graph: _EdgeSplitGraph,
    num_steps: int,
    rng: random.Random,
    active_nodes: list[int] | None = None,
) -> None:
    if num_steps <= 0:
        return

    if active_nodes is None:
        active_nodes = list(range(len(graph.parents)))

    nonleaf = [
        node for node in active_nodes if graph.roles[node] is not _EdgeSplitRole.LEAF
    ]
    if not nonleaf:
        return

    children = graph.children
    parents = graph.parents
    can_reach = _can_reach_with_scratch
    seen = [0] * len(children)
    seen_token = 0
    stack: list[int] = []
    randrange = rng.randrange
    node_order = list(active_nodes)
    rng.shuffle(node_order)
    node_order_idx = 0

    for _ in range(num_steps):
        if node_order_idx == len(node_order):
            rng.shuffle(node_order)
            node_order_idx = 0
        node = node_order[node_order_idx]
        node_order_idx += 1

        parent_list = parents[node]
        if not parent_list:
            continue

        parent_slot = randrange(len(parent_list))
        parent = parent_list[parent_slot]
        new_parent = nonleaf[randrange(len(nonleaf))]

        if new_parent == node or new_parent == parent:
            continue
        if len(children[parent]) == 1:
            continue

        seen_token += 1
        if can_reach(children, node, new_parent, seen, seen_token, stack):
            continue

        parents[node][parent_slot] = new_parent
        children[parent].remove(node)
        children[new_parent].append(node)


def _assemble_edge_split_description(
    graph: _EdgeSplitGraph,
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    num_trunk_node_types: int,
    trunk_node_in_degrees: list[int],
    rng: random.Random,
) -> FixedInDegreeDAGDescription:
    in_degrees = [len(parent_list) for parent_list in graph.parents]
    ready = deque(node for node, in_degree in enumerate(in_degrees) if in_degree == 0)
    topo_order: list[int] = []

    while ready:
        node = ready.popleft()
        topo_order.append(node)
        for child in graph.children[node]:
            in_degrees[child] -= 1
            if in_degrees[child] == 0:
                ready.append(child)

    assert len(topo_order) == len(graph.parents)

    roots: list[int] = []
    trunks: list[int] = []
    leaves: list[int] = []
    for node in topo_order:
        role = graph.roles[node]
        if role is _EdgeSplitRole.ROOT:
            roots.append(node)
        elif role is _EdgeSplitRole.TRUNK:
            trunks.append(node)
        else:
            leaves.append(node)

    final_order = roots
    final_order.extend(trunks)
    final_order.extend(leaves)

    old_to_new = [-1] * len(graph.parents)
    for new_node, old_node in enumerate(final_order):
        old_to_new[old_node] = new_node

    node_inputs_indices: list[list[int]] = [[] for _ in graph.parents]
    node_types = [0] * len(graph.parents)
    root_node_types_start = num_trunk_node_types
    output_node_types_start = num_trunk_node_types + num_root_nodes
    root_slot = 0
    output_slot = 0

    for new_node, old_node in enumerate(final_order):
        role = graph.roles[old_node]
        parent_list = list(graph.parents[old_node])
        if role is _EdgeSplitRole.TRUNK:
            rng.shuffle(parent_list)
        node_inputs_indices[new_node] = [old_to_new[parent] for parent in parent_list]

        if role is _EdgeSplitRole.ROOT:
            node_types[new_node] = root_node_types_start + root_slot
            root_slot += 1
        elif role is _EdgeSplitRole.TRUNK:
            node_types[new_node] = graph.trunk_types[old_node]
        else:
            node_types[new_node] = output_node_types_start + output_slot
            output_slot += 1

    assert root_slot == num_root_nodes
    assert output_slot == num_output_nodes

    return FixedInDegreeDAGDescription(
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        num_trunk_node_types=num_trunk_node_types,
        trunk_node_in_degrees=trunk_node_in_degrees,
        node_inputs_indices=node_inputs_indices,
        node_types=node_types,
    )


def _description_from_flat_parent_arrays(
    *,
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    num_trunk_node_types: int,
    trunk_node_in_degrees: list[int],
    parent_offsets: list[int],
    parent_indices: list[int],
    node_types: list[int],
) -> FixedInDegreeDAGDescription:
    num_nodes = num_root_nodes + num_trunk_nodes + num_output_nodes
    assert len(parent_offsets) == num_nodes + 1
    assert parent_offsets[0] == 0
    assert parent_offsets[-1] == len(parent_indices)

    node_inputs_indices: list[list[int]] = []
    for start, end in zip(parent_offsets, parent_offsets[1:]):
        assert start <= end
        node_inputs_indices.append(list(parent_indices[start:end]))

    return FixedInDegreeDAGDescription(
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        num_trunk_node_types=num_trunk_node_types,
        trunk_node_in_degrees=trunk_node_in_degrees,
        node_inputs_indices=node_inputs_indices,
        node_types=list(node_types),
    )


def _make_random_graph_description_python(
    *,
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    trunk_node_in_degrees: list[int],
    num_trunk_node_types: int,
    mcmc_passes_per_split_round: int,
    seed: int,
) -> FixedInDegreeDAGDescription:
    rng = random.Random(seed)
    raw_graph = _build_interleaved_edge_split_graph(
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        num_trunk_node_types=num_trunk_node_types,
        trunk_node_in_degrees=trunk_node_in_degrees,
        mcmc_passes_per_split_round=mcmc_passes_per_split_round,
        rng=rng,
    )
    return _assemble_edge_split_description(
        graph=raw_graph,
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        num_trunk_node_types=num_trunk_node_types,
        trunk_node_in_degrees=trunk_node_in_degrees,
        rng=rng,
    )


def make_random_graph_description(
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    trunk_node_in_degrees: int | list[int],
    num_trunk_node_types: int,
    mcmc_passes_per_split_round: int = DEFAULT_MCMC_PASSES_PER_SPLIT_ROUND,
) -> FixedInDegreeDAGDescription:
    """Generate a random DAG with exact root, trunk, and output-node counts."""
    if isinstance(trunk_node_in_degrees, int):
        trunk_node_in_degrees = [trunk_node_in_degrees] * num_trunk_node_types
    else:
        trunk_node_in_degrees = list(trunk_node_in_degrees)

    assert len(trunk_node_in_degrees) == num_trunk_node_types
    assert all(in_degree >= 1 for in_degree in trunk_node_in_degrees)
    assert num_root_nodes >= 1
    assert num_output_nodes >= 1
    assert num_trunk_node_types >= 1
    assert mcmc_passes_per_split_round >= 0

    seed = int(torch.randint(0, 2**63 - 1, (1,), dtype=torch.int64).item())

    if _native is not None:
        parent_offsets, parent_indices, node_types = _native.generate_random_graph_arrays(
            num_root_nodes=num_root_nodes,
            num_trunk_nodes=num_trunk_nodes,
            num_output_nodes=num_output_nodes,
            trunk_node_in_degrees=trunk_node_in_degrees,
            num_trunk_node_types=num_trunk_node_types,
            seed=seed,
            mcmc_passes_per_split_round=mcmc_passes_per_split_round,
        )
        return _description_from_flat_parent_arrays(
            num_root_nodes=num_root_nodes,
            num_trunk_nodes=num_trunk_nodes,
            num_output_nodes=num_output_nodes,
            num_trunk_node_types=num_trunk_node_types,
            trunk_node_in_degrees=trunk_node_in_degrees,
            parent_offsets=parent_offsets,
            parent_indices=parent_indices,
            node_types=node_types,
        )

    return _make_random_graph_description_python(
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        trunk_node_in_degrees=trunk_node_in_degrees,
        num_trunk_node_types=num_trunk_node_types,
        mcmc_passes_per_split_round=mcmc_passes_per_split_round,
        seed=seed,
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

    for node_idx, node_type in enumerate(graph.node_types):
        supertype = subtype_to_supertype(
            node_type,
            num_trunk_node_types=graph.num_trunk_node_types,
            num_root_nodes=graph.num_root_nodes,
            num_output_nodes=graph.num_output_nodes,
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
            output_slot = node_type - graph.output_node_types_start
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
