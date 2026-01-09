import torch
from torch import Tensor
from jaxtyping import UInt16, UInt8
from time import sleep, perf_counter

from dagnabbit.bitarrays import output_to_image_array, get_address_bitarrays
from dagnabbit.dag.description import BinaryLogicGateDAGDescription
from dagnabbit.dag.gates import AVAILABLE_GATE_TYPES

BitpackedInputTensor = UInt8[Tensor, "address_bitcount num_packed_bytes"]
BitpackedOutputTensor = UInt8[Tensor, "num_outputs num_packed_bytes"]


def evaluate_dag_triton_kernel(
) -> BitpackedOutputTensor:

    return final_output_values


def evaluate_dag_triton(
    dag: BinaryLogicGateDAGDescription,
    root_node_values: BitpackedInputTensor,
    num_outputs: int,
) -> BitpackedOutputTensor:

    dag = dag.to(root_node_values.device)
    gate_inputs_array_indices = dag.gate_inputs_array_indices
    gate_types = dag.gate_types

    root_node_values = root_node_values.to(root_node_values.device)

    return evaluate_dag_triton_kernel(gate_inputs_array_indices, gate_types, root_node_values, num_outputs)

if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    shape = (76, 45, 3)
    num_outputs = 8

    # Get address bitarrays for the image shape
    address_bitarrays = get_address_bitarrays(
        shape
    )  # [num_root_nodes, num_packed_bytes]
    num_root_nodes = address_bitarrays.shape[0]

    # Convert to torch tensor
    root_node_values = torch.from_numpy(address_bitarrays)

    
    num_iterations = 1000
    start_time = perf_counter()
    for _ in range(num_iterations):
        # Generate a random DAG
        num_gates = 1024
        lookback = 64
        dag = BinaryLogicGateDAGDescription.random(num_root_nodes, num_gates, lookback)
        # print(dag)

        # Evaluate the DAG
        output = evaluate_dag_torch(
            dag, root_node_values, num_outputs
        )  # [num_outputs, num_packed_bytes]
    elapsed = perf_counter() - start_time
    print(f"{num_iterations} iterations in {elapsed:.3f}s ({elapsed / num_iterations * 1000:.3f}ms per iteration)")
    print(f"{num_iterations / elapsed:.0f} iterations/second")

    # Convert to numpy and add batch dimension for output_to_image_array
    output_np = output.numpy()[None, ...]  # [1, num_outputs, num_packed_bytes]

    # Convert to image array
    image_array = output_to_image_array(output_np, shape)  # [16, 16, 3]

    # Save the image
    image = Image.fromarray(image_array, mode="RGB")
    image.save("dag_output.png")
    print(f"Saved dag_output.png")
