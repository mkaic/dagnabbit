import torch
from torch import Tensor
from jaxtyping import UInt16, UInt8
from time import sleep

from dagnabbit.bitarrays import output_to_image_array, get_address_bitarrays
from dagnabbit.dag.description import BinaryLogicGateDAGDescription
from dagnabbit.dag.gates import AVAILABLE_GATE_TYPES

BitpackedTensor = UInt8[Tensor, "address_bitcount num_packed_bytes"]

gate_functions = {
    "NAND": lambda x, y: torch.bitwise_not(torch.bitwise_and(x, y)),
    "NOR": lambda x, y: torch.bitwise_not(torch.bitwise_or(x, y)),
}


def evaluate_dag_torch(
    dag: BinaryLogicGateDAGDescription,
    root_node_values: BitpackedTensor,
    num_outputs: int,
) -> BitpackedTensor:

    address_bitcount, num_packed_bytes = root_node_values.shape

    buffer_size = dag.lookback + address_bitcount
    buffer = torch.zeros(buffer_size, num_packed_bytes, dtype=torch.uint8)
    buffer[:address_bitcount] = root_node_values

    for gate_idx in range(dag.num_gates):
        # print("\n\n")
        # sleep(5)
        # print(buffer[:address_bitcount].numpy())
        # print(buffer[address_bitcount:].numpy())
        # print("\n")

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

        gate_type = dag.gate_types[gate_idx].item()
        gate_function = gate_functions[AVAILABLE_GATE_TYPES[gate_type]]
        output_value = gate_function(*gate_input_values)

        output_buffer_slot = gate_idx % dag.lookback + address_bitcount
        buffer[output_buffer_slot] = output_value

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

    # Generate a random DAG
    num_gates = 4096
    lookback = 256
    dag = BinaryLogicGateDAGDescription.random(num_root_nodes, num_gates, lookback)
    # print(dag)

    # Evaluate the DAG
    output = evaluate_dag_torch(
        dag, root_node_values, num_outputs
    )  # [num_outputs, num_packed_bytes]

    # Convert to numpy and add batch dimension for output_to_image_array
    output_np = output.numpy()[None, ...]  # [1, num_outputs, num_packed_bytes]

    # Convert to image array
    image_array = output_to_image_array(output_np, shape)  # [16, 16, 3]

    # Save the image
    image = Image.fromarray(image_array, mode="RGB")
    image.save("dag_output.png")
    print(f"Saved dag_output.png")
