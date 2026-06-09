from dataclasses import dataclass
from enum import Enum

import torch

NodeInputIndices = list[int]


class NodeSupertype(Enum):
    """Module-level role of a node, used as a batch key for grouped evaluation."""

    ROOT = "root"
    TRUNK = "trunk"
    OUTPUT = "output"
    CONDENSER = "condenser"


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

    Note that the subtype alone cannot distinguish ``CONDENSER`` from ``TRUNK``
    (condenser trunk nodes reuse the trunk type space); that distinction is made
    at the graph level by the caller.
    """
    if num_trunk_node_types is None or num_root_nodes is None or num_output_nodes is None:
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
        referenced: set[int] = set()
        for inputs in self.node_inputs_indices:
            referenced.update(inputs)
        return sorted(n for n in range(self.num_nodes) if n not in referenced)

    def compute_node_ranks(self) -> list[int]:
        """Longest-path depth of each node (roots are rank 0).

        Every other node's rank is ``1 + max(parent ranks)``. Nodes are stored
        in topological order (roots first, every trunk/output node only
        references earlier indices), so a single forward sweep suffices.
        """
        ranks = [0] * self.num_nodes
        for node_idx in range(self.num_root_nodes, self.num_nodes):
            parents = self.node_inputs_indices[node_idx]
            ranks[node_idx] = 1 + max((ranks[p] for p in parents), default=0)
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
        for node_idx, (subtype, rank) in enumerate(
            zip(self.node_types, self.node_ranks)
        ):
            supertype = subtype_to_supertype(
                subtype,
                num_trunk_node_types=self.num_trunk_node_types,
                num_root_nodes=self.num_root_nodes,
                num_output_nodes=self.num_output_nodes,
            )
            # Trunk types live at the start of the type space, so the raw subtype
            # is already the trunk-local subtype; use it directly to split trunk
            # groups (each trunk type has its own module + in-degree). Roots and
            # outputs collapse into one group per supertype.
            trunk_subtype = subtype if supertype is NodeSupertype.TRUNK else None
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
    """
    Generate a random DAG where each trunk node can reference any previous
    node (root or trunk). Each output node is guaranteed to be a leaf and
    has exactly one input sampled from any previously emitted node (root or
    trunk).
    """
    if isinstance(trunk_node_in_degrees, int):
        trunk_node_in_degrees = [trunk_node_in_degrees] * num_trunk_node_types

    trunk_node_types = torch.randint(
        0, num_trunk_node_types, (num_trunk_nodes,), dtype=torch.uint8
    ).tolist()

    max_in_degree = max(trunk_node_in_degrees)
    max_allowed_node_indices = (
        torch.arange(num_trunk_nodes, dtype=torch.int32) + num_root_nodes
    )

    noise = torch.rand(max_in_degree, num_trunk_nodes, dtype=torch.float32)
    trunk_all_indices = (noise * max_allowed_node_indices.float()).floor().int()

    node_inputs_indices: list[list[int]] = []

    for _ in range(num_root_nodes):
        node_inputs_indices.append([])

    for i in range(num_trunk_nodes):
        node_in_degree = trunk_node_in_degrees[trunk_node_types[i]]
        node_inputs_indices.append(trunk_all_indices[:node_in_degree, i].tolist())

    num_candidate_inputs = num_root_nodes + num_trunk_nodes
    if num_output_nodes > 0:
        assert num_candidate_inputs > 0
        output_inputs = torch.randint(
            0, num_candidate_inputs, (num_output_nodes,), dtype=torch.int32
        ).tolist()
        for idx in output_inputs:
            node_inputs_indices.append([idx])

    root_node_types_start = num_trunk_node_types
    output_node_types_start = num_trunk_node_types + num_root_nodes
    node_types: list[int] = (
        [root_node_types_start + i for i in range(num_root_nodes)]
        + trunk_node_types
        + [output_node_types_start + i for i in range(num_output_nodes)]
    )

    return FixedInDegreeDAGDescription(
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        num_trunk_node_types=num_trunk_node_types,
        trunk_node_in_degrees=trunk_node_in_degrees,
        node_inputs_indices=node_inputs_indices,
        node_types=node_types,
    )


def make_condenser_graph_description(
    primary_graph: FixedInDegreeDAGDescription,
) -> FixedInDegreeDAGDescription:
    n_roots = len(primary_graph.leaf_node_indices)
    assert n_roots > 1, (
        "condenser graph requires more than one leaf in the primary graph"
    )

    trunk_node_input_indices: list[NodeInputIndices] = []
    leaf_node_indices: set[int] = set(range(n_roots))

    while len(leaf_node_indices) > 1:
        leaf_list = sorted(leaf_node_indices)
        perm = torch.randperm(len(leaf_list)).tolist()
        shuffled = [leaf_list[i] for i in perm]

        if len(shuffled) % 2 != 0:
            shuffled.pop()

        for i in range(0, len(shuffled), 2):
            input_a = shuffled[i]
            input_b = shuffled[i + 1]
            trunk_node_input_indices.append([input_a, input_b])
            leaf_node_indices.remove(input_a)
            leaf_node_indices.remove(input_b)
            leaf_node_indices.add(len(trunk_node_input_indices) - 1 + n_roots)

    num_trunk_nodes = len(trunk_node_input_indices)

    node_inputs_indices: list[NodeInputIndices] = []
    for _ in range(n_roots):
        node_inputs_indices.append([])
    node_inputs_indices.extend(trunk_node_input_indices)

    num_trunk_node_types = 1
    root_node_types_start = num_trunk_node_types
    node_types: list[int] = [root_node_types_start + i for i in range(n_roots)] + [
        0
    ] * num_trunk_nodes

    return FixedInDegreeDAGDescription(
        num_root_nodes=n_roots,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=0,
        num_trunk_node_types=num_trunk_node_types,
        node_inputs_indices=node_inputs_indices,
        node_types=node_types,
        trunk_node_in_degrees=2,
    )
