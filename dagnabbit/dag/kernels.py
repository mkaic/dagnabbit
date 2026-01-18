import torch
from torch import Tensor
from jaxtyping import UInt16, UInt8, Float32
from time import sleep, perf_counter
from typing import Callable
import random
from pprint import pprint

from dagnabbit.bitarrays import output_to_image_array, get_address_bitarrays
from dagnabbit.dag.description import BinaryLogicGateDAGDescription
from dagnabbit.dag.operators import VALID_OPERATORS

BitpackedInputTensor = UInt8[Tensor, "address_bitcount num_packed_bytes"]
BitpackedOutputTensor = UInt8[Tensor, "num_outputs num_packed_bytes"]

InputEmbeddingsTensor = Float32[Tensor, "address_bitcount embedding_dim"]
OutputEmbeddingsTensor = Float32[Tensor, "num_outputs embedding_dim"]


def recursively_apply_operator_to_random_pairs(
    inputs: list[Tensor],
    operator: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """
    Reduces a list of tensors to a single tensor by repeatedly:
    1. Shuffling the inputs
    2. Pairing adjacent items and applying the operator (batched)
    3. If odd count, holding out the last item and re-appending after pairing

    Uses an iterative while-loop approach instead of recursion.
    All pairs are processed as a batch for efficiency.
    """
    # Stack inputs into a single tensor with batch dimension
    current = torch.stack(list(inputs), dim=0)  # [N, ...]

    while current.shape[0] > 1:
        n = current.shape[0]

        # Shuffle along batch dimension
        perm = torch.randperm(n, device=current.device)
        current = current[perm]

        # Hold out last element if odd count
        held_out = None
        if n % 2 == 1:
            held_out = current[-1:]  # Keep as [1, ...] for easy concatenation
            current = current[:-1]
            n = n - 1

        # Reshape to pair adjacent elements: [N, ...] -> [N//2, 2, ...]
        pair_shape = (n // 2, 2) + current.shape[1:]
        paired = current.view(pair_shape)

        # Apply operator to all pairs at once
        result = operator(paired[:, 0], paired[:, 1])  # [N//2, ...]

        # Re-append held-out item if there was one
        if held_out is not None:
            result = torch.cat([result, held_out], dim=0)

        current = result

    return current[0]


def lookback_optimized_kernel(
    dag: BinaryLogicGateDAGDescription,
    operators: list[Callable[[Tensor, Tensor], Tensor]],
    root_node_values: BitpackedInputTensor | InputEmbeddingsTensor,
    num_outputs: int,
    reduce_leaf_nodes_operator: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> BitpackedOutputTensor | OutputEmbeddingsTensor:
    reduce_leaf_nodes = reduce_leaf_nodes_operator is not None

    dag = dag.to(root_node_values.device)
    root_node_values = root_node_values.to(root_node_values.device)

    address_bitcount, num_packed_bytes = root_node_values.shape

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
