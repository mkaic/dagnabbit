from typing import List, Tuple, Callable

import numpy as np
from tqdm import tqdm

from . import gates as GF
from pprint import pprint

DEBUG = False


def random_dag(
    num_gates: int, num_inputs: int, num_outputs: int
) -> List[Tuple[int, ...]]:
    edges = []

    # Add input nodes (empty tuples)
    for _ in range(num_inputs):
        edges.append(())

    # Add gate nodes
    for i in range(num_gates):
        # Available nodes are inputs and previous gates
        available = list(range(num_inputs + i))
        inputs = tuple(np.random.choice(available, size=2, replace=False))
        edges.append(inputs)

    # Add output nodes (single input tuples)
    available = list(range(num_inputs + num_gates))
    for _ in range(num_outputs):
        output_source = (np.random.choice(available),)
        edges.append(output_source)

    return edges


class ComputationGraph:
    def __init__(self):

        self.node_inputs: dict[str, list[str]] = {}
        self.node_outputs: dict[str, list[str]] = {}
        self.node_functions: dict[str, Callable] = {}
        self.node_values: dict[str, np.ndarray[np.byte]] = {}
        self.evaluation_order: list[str] = []

        self.input_node_ids: set[str] = set()
        self.gate_node_ids: set[str] = set()
        self.output_node_ids: set[str] = set()

    @classmethod
    def from_valid_edges(
        cls, edges: List[Tuple[int, ...]], num_inputs: int, num_outputs: int
    ) -> "ComputationGraph":

        graph = cls()

        graph.num_inputs = num_inputs
        graph.num_outputs = num_outputs
        graph.num_gates = len(edges) - num_inputs - num_outputs

        # Set up input nodes
        for i in range(num_inputs):
            id = f"I{i}"
            graph.input_node_ids.add(id)
            graph.evaluation_order.append(id)
            graph.node_inputs[id] = []
            graph.node_outputs[id] = []
            graph.node_functions[id] = None
            graph.node_values[id] = None

        # Set up gate nodes and their connections
        for i, inputs in enumerate(edges[num_inputs:-num_outputs]):
            id = f"G{i}"
            graph.gate_node_ids.add(id)
            graph.evaluation_order.append(id)
            graph.node_inputs[id] = []
            graph.node_outputs[id] = []
            graph.node_functions[id] = np.random.choice(GF.AVAILABLE_FUNCTIONS)
            graph.node_values[id] = None

            # Connect inputs based on the edge tuple
            for input_idx in inputs:
                if input_idx < num_inputs:
                    input_id = f"I{input_idx}"
                else:
                    input_id = f"G{input_idx - num_inputs}"
                graph.connect(transmitting_id=input_id, receiving_id=id)

        # Set up output nodes
        for i in range(num_outputs):
            id = f"@{i}"
            graph.output_node_ids.add(id)
            graph.evaluation_order.append(id)
            graph.node_inputs[id] = []
            graph.node_outputs[id] = []
            graph.node_functions[id] = None
            graph.node_values[id] = None

            # Connect output to last gate in the edges list
            output_source_idx = edges[-num_outputs + i][0]
            if output_source_idx < num_inputs:
                source_id = f"I{output_source_idx}"
            else:
                source_id = f"G{output_source_idx - num_inputs}"
            graph.connect(transmitting_id=source_id, receiving_id=id)

        return graph

    def __repr__(self):
        to_return = []
        for id in self.evaluation_order:
            to_return.append(
                f"{self.node_inputs[id]} --> {id} --> {self.node_outputs[id]}"
            )
        return "\n".join(to_return)

    def evaluate(self, inputs: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
        assert inputs.shape[0] == self.num_inputs

        output_values = np.zeros((self.num_outputs, inputs.shape[-1]), dtype=np.uint8)

        for i, input_node_id in enumerate(sorted(list(self.input_node_ids))):
            self.node_values[input_node_id] = inputs[i]

        for gate_node_id in self.gate_node_ids:
            self.node_values[gate_node_id] = None

        for gate_node_id in [
            n for n in self.evaluation_order if n in self.gate_node_ids
        ]:
            function_inputs = [
                self.node_values[input_id]
                for input_id in self.node_inputs[gate_node_id]
            ]
            self.node_values[gate_node_id] = self.node_functions[gate_node_id](
                *function_inputs
            )

        for i, output_node_id in enumerate(sorted(list(self.output_node_ids))):
            output_source_node_id = self.node_inputs[output_node_id][0]
            self.node_values[output_node_id] = self.node_values[output_source_node_id]
            output_values[i] = self.node_values[output_node_id]

        return output_values

    def connect(self, transmitting_id: str, receiving_id: str):
        self.node_inputs[receiving_id].append(transmitting_id)
        self.node_outputs[transmitting_id].append(receiving_id)

    def disconnect(self, transmitting_id: str, receiving_id: str):
        self.node_inputs[receiving_id].remove(transmitting_id)
        self.node_outputs[transmitting_id].remove(receiving_id)

    def topological_sort(self):

        refcounts: dict[str, int] = {}
        ids_with_zero_references: list[str] = []
        sorted_ids: list[str] = []

        for id in self.evaluation_order:
            refcounts[id] = len(set(self.node_inputs[id]))
            if refcounts[id] == 0:
                ids_with_zero_references.append(id)

        if DEBUG:
            print("refcounts")
            pprint(refcounts)
            print("ids_with_zero_references")
            pprint(ids_with_zero_references)

        while len(ids_with_zero_references) > 0:

            node_id = ids_with_zero_references.pop()
            sorted_ids.append(node_id)

            for output_node_id in set(self.node_outputs[node_id]):
                refcounts[output_node_id] -= 1

                if refcounts[output_node_id] == 0:
                    ids_with_zero_references.append(output_node_id)

        self.evaluation_order = sorted_ids

        if DEBUG:
            cyclic = [
                f"{self.node_inputs[id]} --> "
                "{id} --> {self.node_outputs[id]} | "
                "{self.node_descendants[id]}"
                for id in self.evaluation_order
                if refcounts[id] != 0
            ]
            print("\n")
            print("Cyclic gates:")
            for c in cyclic:
                print(c)
            print("\n")

    def find_descendants(self, node_id: str) -> set[str]:
        descendants = set()
        stack = [node_id]
        seen = set()

        while len(stack) > 0:
            current_node_id = stack.pop()
            if current_node_id in seen:
                continue
            else:
                seen.add(current_node_id)
                descendants.add(current_node_id)

                for output_node_id in self.node_outputs[current_node_id]:
                    stack.append(output_node_id)

        return descendants

    def stage_node_input_mutation(self, node_id: str) -> Tuple[str, str, str]:
        old_input_id = str(np.random.choice(self.node_inputs[node_id]))

        descendants = self.find_descendants(node_id)

        options: list[str] = [
            i
            for i in self.evaluation_order
            if i not in descendants and i != old_input_id
        ]

        new_input_id = str(np.random.choice(options))

        self.disconnect(transmitting_id=old_input_id, receiving_id=node_id)
        self.connect(transmitting_id=new_input_id, receiving_id=node_id)

        self.topological_sort()

        return old_input_id, new_input_id

    def undo_node_input_mutation(
        self, node_id: str, old_input_id: str, new_input_id: str
    ):
        self.disconnect(transmitting_id=new_input_id, receiving_id=node_id)
        self.connect(transmitting_id=old_input_id, receiving_id=node_id)

        self.topological_sort()

    def stage_node_function_mutation(
        self, node_id: str
    ) -> Tuple[str, Callable, Callable]:

        old_function = self.node_functions[node_id]

        options = [f for f in GF.AVAILABLE_FUNCTIONS if f != old_function]
        new_function = np.random.choice(options)

        self.node_functions[node_id] = new_function

        return old_function, new_function

    def undo_node_function_mutation(
        self, node_id: str, old_function: Callable, new_function: Callable
    ):
        self.node_functions[node_id] = old_function
