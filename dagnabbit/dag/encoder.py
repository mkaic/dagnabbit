import torch
import torch.nn as nn
import torch.nn.functional as F


class DAGEncoder(nn.Module):
    def __init__(self, num_root_nodes, embedding_dim):
        super().__init__()
        self.num_root_nodes = num_root_nodes
        self.operators = nn.ModuleDict(
            {
                "NAND": None,
                "NOR": None,
                "LEAF": None,
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class NeuralOperator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # takes two vectors as inputs, outputs a single vector
        self.linear_1 = nn.Linear(input_dim * 2, input_dim * 4)
        self.linear_2 = nn.Linear(input_dim * 4, input_dim * 1)

        self.activation = nn.GELU()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Takes two vectors as inputs, outputs a single vector
        a: [batch_size, input_dim]
        b: [batch_size, input_dim]
        output: [batch_size, input_dim]
        """
        x = torch.cat([a, b], dim=1)

        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.activation(x)

        x = F.normalize(x, dim=1)

        return x







import torch
from torch import Tensor
from jaxtyping import Float32
from typing import Callable

from dagnabbit.dag.description import BinaryLogicGateDAGDescription

InputEmbeddingsTensor = Float32[Tensor, "num_root_nodes embedding_dim"]
OutputEmbeddingsTensor = Float32[Tensor, "num_outputs embedding_dim"]


def lookback_optimized_kernel(
    dag: BinaryLogicGateDAGDescription,
    operators: list[Callable[[Tensor, Tensor], Tensor]],
    root_node_values: InputEmbeddingsTensor,
    num_outputs: int,
) -> OutputEmbeddingsTensor:

    dag = dag.to(root_node_values.device)
    root_node_values = root_node_values.to(root_node_values.device)

    num_root_nodes, embedding_dim = root_node_values.shape

    buffer_size = dag.lookback + address_bitcount
    buffer = torch.zeros(buffer_size, num_packed_bytes, dtype=torch.uint8)
    buffer[:address_bitcount] = root_node_values

    if reduce_leaf_nodes:
        leaf_node_map = {}
        for leaf_node_index in dag.leaf_node_indices:
            if leaf_node_index < dag.num_root_nodes:
                leaf_node_map[leaf_node_index] = root_node_values[leaf_node_index]
            else:
                leaf_node_map[leaf_node_index] = None

    for gate_idx in range(dag.num_gates):
        # leaf_node_map_shapes = {k: v.shape if v is not None else None for k, v in leaf_node_map.items()}
        # pprint(leaf_node_map_shapes)
        # sleep(1)

        gate_input_values = []
        for input_slot in range(2):
            input_array_index = dag.gate_inputs_array_indices[
                input_slot, gate_idx
            ].item()
            if input_array_index < dag.num_root_nodes:
                retrieved_bitarray = buffer[input_array_index]
            else:
                input_gate_index = input_array_index - dag.num_root_nodes
                input_buffer_index = input_gate_index % dag.lookback + address_bitcount
                retrieved_bitarray = buffer[input_buffer_index]

            gate_input_values.append(retrieved_bitarray)
            # print(retrieved_bitarray.numpy(), input_array_index)

        operator_index = dag.gate_types[gate_idx].item()
        operator_key = list(operators.keys())[operator_index]
        operator_function = operators[operator_key]
        output_value = operator_function(*gate_input_values)

        output_buffer_slot = gate_idx % dag.lookback + address_bitcount
        buffer[output_buffer_slot] = output_value

        if reduce_leaf_nodes:
            leaf_node_map[gate_idx + dag.num_root_nodes] = output_value

    if reduce_leaf_nodes:
        return recursively_apply_operator_to_random_pairs(
            leaf_node_map.values(), reduce_leaf_nodes_operator
        )

    else:
        final_output_buffer_indices = (
            torch.arange(dag.num_gates - num_outputs, dag.num_gates) % dag.lookback
            + address_bitcount
        )
        final_output_values = buffer[final_output_buffer_indices]

        return final_output_values


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    shape = (128, 128, 3)
    num_outputs = 8

    # Get address bitarrays for the image shape
    address_bitarrays = get_address_bitarrays(
        shape
    )  # [num_root_nodes, num_packed_bytes]
    num_root_nodes = address_bitarrays.shape[0]

    # Convert to torch tensor
    root_node_values = torch.from_numpy(address_bitarrays)

    num_gates = 4096
    lookback = 256

    print("Warming up...")
    # warmup
    for _ in range(10):
        dag = BinaryLogicGateDAGDescription.random(num_root_nodes, num_gates, lookback)
        output = lookback_optimized_kernel(
            dag=dag,
            operators=VALID_OPERATORS,
            root_node_values=root_node_values,
            num_outputs=num_outputs,
            reduce_leaf_nodes_operator=lambda a, b: torch.bitwise_xor(a, b),
        )  # [num_outputs, num_packed_bytes]

    print("Warmup complete")
    # print(dag)

    num_iterations = 100

    print(f"Timing {num_iterations} iterations...")
    start_time = perf_counter()
    for _ in range(num_iterations):
        # Generate a random DAG
        dag = BinaryLogicGateDAGDescription.random(num_root_nodes, num_gates, lookback)
        # print(dag)

        # Evaluate the DAG
        output = lookback_optimized_kernel(
            dag=dag,
            operators=VALID_OPERATORS,
            root_node_values=root_node_values,
            num_outputs=num_outputs,
            reduce_leaf_nodes_operator=lambda a, b: torch.bitwise_xor(a, b),
        )  # [num_outputs, num_packed_bytes]

    elapsed = perf_counter() - start_time
    print(
        f"{num_iterations} iterations in {elapsed:.3f}s ({elapsed / num_iterations * 1000:.3f}ms per iteration)"
    )
    print(f"{num_iterations / elapsed:.0f} iterations/second")

    # Convert to numpy and add batch dimension for output_to_image_array
    output_np = output.numpy()[None, ...]  # [1, num_outputs, num_packed_bytes]

    # Convert to image array
    image_array = output_to_image_array(output_np, shape)  # [16, 16, 3]

    # Save the image
    image = Image.fromarray(image_array, mode="RGB")
    image.save("dag_output.png")
    print(f"Saved dag_output.png")
