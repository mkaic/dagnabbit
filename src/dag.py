from typing import List, Tuple, Set

import numpy as np
from tqdm import tqdm

from . import gates as GF


class Node:
    def __init__(self, id: str):
        self.id = id
        self.input_nodes: List["Node"] = []
        self.output_nodes: List["Node"] = []
        self.logical_function = GF.NP_NAND
        self.value = None
        self.hash = hash(self.id)
        self.descendants: Set[str] = set([self])

    def __repr__(self):
        return self.id

    def __hash__(self):
        return self.hash


class ComputationGraph:
    def __init__(self, num_gates: int, num_inputs: int, num_outputs: int = 8):
        self.num_gates = num_gates
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.total_nodes = num_gates + num_inputs + num_outputs

        self.input_nodes: List[Node] = []
        for i in range(self.num_inputs):
            self.input_nodes.append(Node(id=f"INPUT_{i}"))

        self.gate_nodes: List[Node] = []
        print("Initializing gates...")
        for i in tqdm(range(self.num_gates)):
            new_node = Node(id=f"GATE_{i}")
            new_node.logical_function = GF.NP_NAND

            for _ in range(2):
                random_input = np.random.choice(self.input_nodes + self.gate_nodes)
                self.connect(transmitting=random_input, receiving=new_node)

            self.gate_nodes.append(new_node)

        self.output_nodes: List[Node] = []
        for i in range(self.num_outputs):
            node = Node(id=f"OUTPUT_{i}")
            self.connect(transmitting=np.random.choice(self.gate_nodes), receiving=node)
            self.output_nodes.append(node)

    def topological_sort(self):
        all_nodes = self.input_nodes + self.gate_nodes + self.output_nodes
        reference_counts = {n: len(n.output_nodes) for n in all_nodes}
        gates_with_zero_references = [n for n in all_nodes if reference_counts[n] == 0]
        sorted_gates = []

        while len(gates_with_zero_references) > 0:
            node = gates_with_zero_references.pop()
            for input_node in node.input_nodes:
                for output_node in input_node.output_nodes:
                    if output_node == node:
                        reference_counts[input_node] -= 1
                if reference_counts[input_node] == 0:
                    gates_with_zero_references.append(input_node)

            sorted_gates.append(node)

        sorted_gates.reverse()
        sorted_gates = [
            n for n in sorted_gates if not (("INPUT" in n.id) or ("OUTPUT" in n.id))
        ]
        self.gate_nodes = sorted_gates

    def connect(self, transmitting: Node, receiving: Node):
        receiving.input_nodes.append(transmitting)
        transmitting.output_nodes.append(receiving)
        transmitting.descendants = transmitting.descendants.union(receiving.descendants)

    def disconnect(self, transmitting: Node, receiving: Node):
        receiving.input_nodes.remove(transmitting)
        transmitting.output_nodes.remove(receiving)

        self.topological_sort()

        for node in self.gate_nodes:
            node.descendants = set([node.id])
        for node in reversed(self.gate_nodes):
            node: Node
            for input_node in node.input_nodes:
                input_node.descendants = input_node.descendants | node.descendants

    def stage_node_input_mutation(self, node: Node) -> Tuple[Node, Node, Node]:
        old_input = np.random.choice(node.input_nodes)
        options: set = set(self.input_nodes) | set(self.gate_nodes) - node.descendants
        options.remove(old_input)
        new_input = np.random.choice(list(options))

        self.disconnect(transmitting=old_input, receiving=node)
        self.connect(transmitting=new_input, receiving=node)

        return node, old_input, new_input

    def undo_node_input_mutation(self, node: Node, old_input: Node, new_input: Node):
        self.disconnect(transmitting=new_input, receiving=node)
        self.connect(transmitting=old_input, receiving=node)

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
