import numpy as np
from typing import Tuple, List

from . import gates as GF


class Node:
    def __init__(self):
        self.input_nodes: List["Node"] = []
        self.logical_function = GF.NP_NAND
        self.value = None


class ComputationGraph:
    def __init__(self, num_gates: int, num_inputs: int, num_outputs: int = 8):
        self.num_gates = num_gates
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.total_nodes = num_gates + num_inputs + num_outputs

        self.input_nodes: List[Node] = []
        for _ in range(self.num_inputs):
            self.input_nodes.append(Node())

        self.gate_nodes: List[Node] = []
        for i in range(self.num_gates):
            node = Node()
            available_inputs = self.input_nodes + self.gate_nodes[:i]
            node.input_nodes = np.random.choice(available_inputs, 2, replace=False)
            node.logical_function = np.random.choice(GF.AVAILABLE_FUNCTIONS)
            self.gate_nodes.append(node)

        self.output_nodes: List[Node] = []
        for _ in range(self.num_outputs):
            node = Node()
            available_inputs = self.input_nodes + self.gate_nodes
            node.input_nodes = np.random.choice(available_inputs, 1, replace=False)
            self.output_nodes.append(node)

    def evaluate(self, inputs: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
        assert inputs.shape[0] == self.num_inputs

        output_values = np.zeros((self.num_outputs, inputs.shape[-1]), dtype=np.uint8)

        for i, input_node in enumerate(self.input_nodes):
            input_node.value = inputs[i]

        for gate_node in self.gate_nodes:
            function_inputs = [n.value for n in gate_node.input_nodes]
            gate_node.value = gate_node.logical_function(*function_inputs)

        for i, output_node in enumerate(self.output_nodes):
            output_node.value = output_node.input_nodes[0].value
            output_values[i] = output_node.value

        return output_values
