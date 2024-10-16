from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from . import gates as GF


class Node:
    def __init__(self, id: str):
        self.id = id
        self.input_nodes: List["Node"] = []
        self.logical_function = GF.NP_NAND
        self.value = None

    def __repr__(self):
        return self.id


def connect(transmitting: Node, receiving: Node):
    receiving.input_nodes.append(transmitting)


def disconnect(transmitting: Node, receiving: Node):
    receiving.input_nodes.remove(transmitting)


class ComputationGraph:
    def __init__(self, num_gates: int, num_inputs: int, num_outputs: int = 8):
        self.num_gates = num_gates
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.total_nodes = num_gates + num_inputs + num_outputs

        self.input_nodes: List[Node] = []
        for i in range(self.num_inputs):
            self.input_nodes.append(Node(id=f"INPUT_{i}"))

        self.output_nodes: List[Node] = []
        for i in range(self.num_outputs):
            node = Node(id=f"OUTPUT_{i}")
            available_inputs = self.input_nodes  # + self.gate_nodes
            node.input_nodes = list(
                np.random.choice(available_inputs, 1, replace=False)
            )
            self.output_nodes.append(node)

        self.gate_nodes: List[Node] = []
        print("Initializing gates...")
        for i in tqdm(range(self.num_gates)):
            new_node = Node(id=f"GATE_{i}")
            new_node.logical_function = GF.NP_NAND

            random_node_idx = np.random.randint(0, self.num_outputs + i)
            random_node: Node = (self.gate_nodes + self.output_nodes)[random_node_idx]
            random_input: Node = np.random.choice(random_node.input_nodes)

            disconnect(transmitting=random_input, receiving=random_node)
            connect(transmitting=new_node, receiving=random_node)

            connect(transmitting=random_input, receiving=new_node)

            insertion_idx = min(random_node_idx, len(self.gate_nodes))
            available_inputs = self.input_nodes + self.gate_nodes[:insertion_idx]
            connect(transmitting=np.random.choice(available_inputs), receiving=new_node)

            self.gate_nodes.insert(insertion_idx, new_node)

    def evaluate(self, inputs: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
        assert inputs.shape[0] == self.num_inputs

        output_values = np.zeros((self.num_outputs, inputs.shape[-1]), dtype=np.uint8)

        for i, input_node in enumerate(self.input_nodes):
            input_node.value = inputs[i]

        for gate_node in self.gate_nodes:
            function_inputs = [n.value for n in gate_node.input_nodes]
            try:
                gate_node.value = gate_node.logical_function(*function_inputs)
            except Exception as e:
                print("gate_node.logical_function", gate_node.logical_function)
                print("function_inputs", function_inputs)
                print("gate_node.id", gate_node)
                print("gate_node.input_nodes", gate_node.input_nodes)
                raise e

        for i, output_node in enumerate(self.output_nodes):
            output_node.value = output_node.input_nodes[0].value
            output_values[i] = output_node.value

        return output_values
