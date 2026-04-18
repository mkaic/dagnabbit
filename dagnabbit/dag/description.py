import torch

NodeInputIndices = list[int]


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
