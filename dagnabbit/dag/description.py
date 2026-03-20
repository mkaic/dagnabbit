import torch
from jaxtyping import UInt16, UInt8
from torch import Tensor


class FixedInDegreeDAGDescription:
    def __init__(
        self,
        num_root_nodes: int,
        num_trunk_nodes: int,
        num_trunk_node_types: int,
        trunk_node_in_degree: int,
        trunk_node_inputs_indices: UInt16[Tensor, "trunk_node_in_degree num_trunk_nodes"],
        trunk_node_types: UInt8[Tensor, "num_trunk_nodes"],
    ):
        assert trunk_node_inputs_indices.shape[0] == trunk_node_in_degree

        self.trunk_node_inputs_indices = trunk_node_inputs_indices.to(torch.uint16)
        self.trunk_node_types = trunk_node_types.to(torch.uint8)
        self.num_trunk_nodes = num_trunk_nodes
        self.num_root_nodes = num_root_nodes
        self.trunk_node_in_degree = trunk_node_in_degree

        assert self.trunk_node_inputs_indices.is_contiguous()
        assert self.trunk_node_types.is_contiguous()
        assert self.trunk_node_in_degree > 0

        self.leaf_node_indices = self.identify_leaf_nodes()

    def identify_leaf_nodes(self) -> list[int]:
        """
        Identify all leaf nodes in the DAG.

        A leaf node is a node whose output is not referenced as an input to any other node.
        Returns array indices of all leaf nodes as a list of integers.
        """
        num_nodes = self.num_root_nodes + self.num_trunk_nodes
        referenced = set(
            self.trunk_node_inputs_indices.to(torch.int32).flatten().tolist()
        )
        return sorted(n for n in range(num_nodes) if n not in referenced)

    def to(self, device: torch.device) -> "FixedInDegreeDAGDescription":
        self.trunk_node_inputs_indices = self.trunk_node_inputs_indices.to(device)
        self.trunk_node_types = self.trunk_node_types.to(device)
        return self

def make_random_graph_description(
    num_root_nodes: int,
    num_trunk_nodes: int,
    in_degree: int,
    num_node_types: int,
) -> FixedInDegreeDAGDescription:
    """
    Generate a random DAG where each node can reference any previous node.
    Each node can reference up to `in_degree` previous nodes, including the same node more than once.
    """

    max_allowed_node_indices = (
        torch.arange(num_trunk_nodes, dtype=torch.int32) + num_root_nodes
    )

    # Sample uniformly from [0, num_valid_references - 1] for each node.
    # Since noise is in [0, 1), (noise * n).int() gives integers in [0, n-1].
    noise = torch.rand(in_degree, num_trunk_nodes, dtype=torch.float32)
    node_inputs_indices = (noise * max_allowed_node_indices.float()).floor().int()

    node_types = torch.randint(
        0, num_node_types, (num_trunk_nodes,), dtype=torch.uint8
    )

    return FixedInDegreeDAGDescription(
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_trunk_node_types=num_node_types,
        trunk_node_in_degree=in_degree,
        trunk_node_inputs_indices=node_inputs_indices,
        trunk_node_types=node_types,
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
        perm = torch.randperm(len(leaf_node_indices)).tolist()
        shuffled = [leaf_node_indices[i] for i in perm]

        if len(shuffled) % 2 == 0:
            odd_node_out = shuffled.pop()
        else:
            odd_node_out = None
        
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
        trunk_node_inputs_indices=torch.tensor(trunk_node_input_indices, dtype=torch.int),
        trunk_node_types=torch.zeros(len(trunk_node_input_indices), dtype=torch.uint8),
        trunk_node_in_degree=2,
    )