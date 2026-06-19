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


def make_random_graph_description(
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    trunk_node_in_degrees: int | list[int],
    num_trunk_node_types: int,
) -> FixedInDegreeDAGDescription:
    """Generate a random DAG with exact root, trunk, and output-node counts.

    The previous edge-splitting + MCMC generator (and its Rust port) has been
    removed; this is the clean seam where the new sampler will live. It must
    return a :class:`FixedInDegreeDAGDescription` whose node arrays satisfy the
    representation's invariants:

    - ``num_root_nodes`` roots first (in-degree 0), then ``num_trunk_nodes``
      trunk nodes, then ``num_output_nodes`` outputs (in-degree 1), all in
      topological order (every node references only earlier indices).
    - Each trunk node's in-degree matches its type's entry in
      ``trunk_node_in_degrees``; trunk types are drawn from
      ``[0, num_trunk_node_types)``.

    See the conversation notes for the two candidate algorithms to implement
    here: (A) forward straight-line construction with coverage repair, and
    (B) exactly-uniform sampling via the recursive method / Boltzmann samplers.
    """
    raise NotImplementedError(
        "random DAG generation has been removed; implement the new sampler here"
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
