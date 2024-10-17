from typing import List, Tuple, Callable

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
        self.descendants: set[str] = set([self])

    def __repr__(self):
        return self.id


class ComputationGraph:
    def __init__(self, num_gates: int, num_inputs: int, num_outputs: int = 255):
        self.num_gates = num_gates
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.input_nodes: List[Node] = []
        for i in range(self.num_inputs):
            self.input_nodes.append(Node(id=f"I{i}"))

        self.gate_nodes: List[Node] = []
        print("Initializing gates...")
        for i in tqdm(range(self.num_gates)):
            new_node = Node(id=f"G{i}")
            new_node.logical_function = GF.NP_NAND

            for _ in range(2):
                random_input = np.random.choice(self.input_nodes + self.gate_nodes)
                self.connect(transmitting=random_input, receiving=new_node)

            self.gate_nodes.append(new_node)

        self.output_nodes: List[Node] = []
        for i in range(self.num_outputs):
            node = Node(id=f"?{i}")
            self.connect(transmitting=np.random.choice(self.gate_nodes), receiving=node)
            self.output_nodes.append(node)

        self.refresh_descendants()

    def topological_sort(self):
        all_nodes = self.input_nodes + self.gate_nodes + self.output_nodes

        refcounts = {}
        gates_with_zero_references = []
        sorted_gates: List[Node] = []

        for node in all_nodes:
            refcounts[node.id] = len(set(node.output_nodes))
            if refcounts[node.id] == 0:
                gates_with_zero_references.append(node)

        while len(gates_with_zero_references) > 0:

            node: Node = gates_with_zero_references.pop()
            sorted_gates.append(node)

            for input_node in set(node.input_nodes):
                refcounts[input_node.id] -= 1

                if refcounts[input_node.id] == 0:
                    gates_with_zero_references.append(input_node)

        sorted_gates.reverse()
        sorted_gates = [n for n in sorted_gates if not (("I" in n.id) or ("?" in n.id))]
        self.gate_nodes = sorted_gates

        # cyclic = [
        #     f"{n.input_nodes} --> {n.id} --> {n.output_nodes} | {n.descendants}" for n in all_nodes if refcounts[n.id] != 0
        # ]
        # print("\n")
        # print("Cyclic gates:")
        # for c in cyclic:
        #     print(c)
        # print("\n")

    def refresh_descendants(self):
        for n in reversed(self.input_nodes + self.gate_nodes + self.output_nodes):
            n: Node
            n.descendants = set([n])
            for output_node in n.output_nodes:
                n.descendants = n.descendants | output_node.descendants

    def connect(self, transmitting: Node, receiving: Node):
        receiving.input_nodes.append(transmitting)
        transmitting.output_nodes.append(receiving)
        transmitting.descendants = transmitting.descendants | receiving.descendants

    def disconnect(self, transmitting: Node, receiving: Node):
        receiving.input_nodes.remove(transmitting)
        transmitting.output_nodes.remove(receiving)

    def stage_node_input_mutation(self, node: Node) -> Tuple[Node, Node, Node]:
        old_input = np.random.choice(node.input_nodes)

        options: set = set(self.input_nodes) | set(self.gate_nodes) - node.descendants
        options.remove(old_input)
        new_input = np.random.choice(list(options))

        self.disconnect(transmitting=old_input, receiving=node)
        self.connect(transmitting=new_input, receiving=node)

        self.topological_sort()
        self.refresh_descendants()

        return node, old_input, new_input

    def undo_node_input_mutation(self, node: Node, old_input: Node, new_input: Node):
        self.disconnect(transmitting=new_input, receiving=node)
        self.connect(transmitting=old_input, receiving=node)

        self.topological_sort()
        self.refresh_descendants()

    def stage_node_function_mutation(
        self, node: Node
    ) -> Tuple[Node, Callable, Callable]:

        old_function = node.logical_function

        options = [f for f in GF.AVAILABLE_FUNCTIONS if f != old_function]
        new_function = np.random.choice(options)

        node.logical_function = new_function

        return node, old_function, new_function

    def undo_node_function_mutation(
        self, node: Node, old_function: Callable, new_function: Callable
    ):
        node.logical_function = old_function

    def evaluate(self, inputs: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
        assert inputs.shape[0] == self.num_inputs

        output_values = np.zeros((self.num_outputs, inputs.shape[-1]), dtype=np.uint8)

        for i, input_node in enumerate(self.input_nodes):
            input_node.value = inputs[i]

        for gate_node in self.gate_nodes:
            gate_node.value = None

        for gate_node in self.gate_nodes:
            function_inputs = [n.value for n in gate_node.input_nodes]
            gate_node.value = gate_node.logical_function(*function_inputs)

        for i, output_node in enumerate(self.output_nodes):
            output_node.value = output_node.input_nodes[0].value
            output_values[i] = output_node.value

        return output_values
