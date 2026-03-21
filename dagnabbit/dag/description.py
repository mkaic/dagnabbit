import torch
from jaxtyping import UInt8
from torch import Tensor


class FixedInDegreeDAGDescription:
    def __init__(
        self,
        num_root_nodes: int,
        num_trunk_nodes: int,
        num_trunk_node_types: int,
        trunk_node_in_degrees: int | list[int],
        trunk_node_inputs_indices: list[list[int]],
        trunk_node_types: list[int],
    ):
        if isinstance(trunk_node_in_degrees, int):
            trunk_node_in_degrees = [trunk_node_in_degrees] * num_trunk_node_types

        assert len(trunk_node_in_degrees) == num_trunk_node_types

        self.trunk_node_inputs_indices = trunk_node_inputs_indices
        self.trunk_node_types = trunk_node_types
        self.num_trunk_nodes = num_trunk_nodes
        self.num_root_nodes = num_root_nodes
        self.num_trunk_node_types = num_trunk_node_types
        self.trunk_node_in_degrees = trunk_node_in_degrees

        assert all(d > 0 for d in self.trunk_node_in_degrees)
        assert len(self.trunk_node_inputs_indices) == num_trunk_nodes
        for i, inputs in enumerate[list[int]](self.trunk_node_inputs_indices):
            expected = self.trunk_node_in_degrees[self.trunk_node_types[i]]
            assert len(inputs) == expected

        self.leaf_node_indices = self.identify_leaf_nodes()

    def identify_leaf_nodes(self) -> list[int]:
        """
        Identify all leaf nodes in the DAG.

        A leaf node is a node whose output is not referenced as an input to any other node.
        Returns array indices of all leaf nodes as a list of integers.
        """
        num_nodes = self.num_root_nodes + self.num_trunk_nodes
        referenced: set[int] = set()
        for inputs in self.trunk_node_inputs_indices:
            referenced.update(inputs)
        return sorted(n for n in range(num_nodes) if n not in referenced)


def make_random_graph_description(
    num_root_nodes: int,
    num_trunk_nodes: int,
    trunk_node_in_degrees: list[int],
    num_trunk_node_types: int,
) -> FixedInDegreeDAGDescription:
    """
    Generate a random DAG where each node can reference any previous node.
    Each node can reference up to its type's in_degree previous nodes,
    including the same node more than once.
    """
    if isinstance(trunk_node_in_degrees, int):
        trunk_node_in_degrees = [trunk_node_in_degrees] * num_trunk_node_types
    else:
        trunk_node_in_degrees = trunk_node_in_degrees

    trunk_node_types = torch.randint(
        0, num_trunk_node_types, (num_trunk_nodes,), dtype=torch.uint8
    ).tolist()

    max_in_degree = max(trunk_node_in_degrees)
    max_allowed_node_indices = (
        torch.arange(num_trunk_nodes, dtype=torch.int32) + num_root_nodes
    )

    noise = torch.rand(max_in_degree, num_trunk_nodes, dtype=torch.float32)
    all_indices = (noise * max_allowed_node_indices.float()).floor().int()

    trunk_node_inputs_indices: list[list[int]] = []
    for i in range(num_trunk_nodes):
        node_in_degree = trunk_node_in_degrees[trunk_node_types[i]]
        trunk_node_inputs_indices.append(all_indices[:node_in_degree, i].tolist())

    return FixedInDegreeDAGDescription(
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_trunk_node_types=num_trunk_node_types,
        trunk_node_in_degrees=trunk_node_in_degrees,
        trunk_node_inputs_indices=trunk_node_inputs_indices,
        trunk_node_types=trunk_node_types,
    )


def make_condenser_graph_description(
    primary_graph: FixedInDegreeDAGDescription,
) -> FixedInDegreeDAGDescription | None:
    n_roots = len(primary_graph.leaf_node_indices)
    if n_roots == 1:
        return None

    trunk_node_input_indices: list[list[int]] = []
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

    return FixedInDegreeDAGDescription(
        num_root_nodes=n_roots,
        num_trunk_nodes=len(trunk_node_input_indices),
        num_trunk_node_types=1,
        trunk_node_inputs_indices=trunk_node_input_indices,
        trunk_node_types=[0] * len(trunk_node_input_indices),
        trunk_node_in_degrees=2,
    )


def graft_condenser_graph_onto_primary_graph(
    primary_graph: FixedInDegreeDAGDescription,
    condenser_graph: FixedInDegreeDAGDescription,
) -> FixedInDegreeDAGDescription:
    primary_total = primary_graph.num_root_nodes + primary_graph.num_trunk_nodes
    leaf_indices = primary_graph.leaf_node_indices

    remapped_condenser_indices: list[list[int]] = []
    for inputs in condenser_graph.trunk_node_inputs_indices:
        remapped: list[int] = []
        for idx in inputs:
            if idx < condenser_graph.num_root_nodes:
                remapped.append(leaf_indices[idx])
            else:
                remapped.append(idx - condenser_graph.num_root_nodes + primary_total)
        remapped_condenser_indices.append(remapped)

    combined_indices = (
        primary_graph.trunk_node_inputs_indices + remapped_condenser_indices
    )

    max_primary_type = primary_graph.trunk_node_types.max().item()
    condenser_types = torch.full(
        (condenser_graph.num_trunk_nodes,),
        max_primary_type + 1,
        dtype=torch.uint8,
    )
    combined_types = torch.cat([primary_graph.trunk_node_types, condenser_types])
    combined_in_degrees = (
        primary_graph.trunk_node_in_degrees + condenser_graph.trunk_node_in_degrees
    )

    return FixedInDegreeDAGDescription(
        num_root_nodes=primary_graph.num_root_nodes,
        num_trunk_nodes=primary_graph.num_trunk_nodes + condenser_graph.num_trunk_nodes,
        num_trunk_node_types=primary_graph.num_trunk_node_types + 1,
        trunk_node_in_degrees=combined_in_degrees,
        trunk_node_inputs_indices=combined_indices,
        trunk_node_types=combined_types,
    )
