from typing import Callable

import numpy as np

from pprint import pprint

DEBUG = False


class ComputationGraph:
    def __init__(self):

        self.node_inputs: dict[str, list[str]] = {}
        self.node_outputs: dict[str, list[str]] = {}
        self.node_functions: dict[str, Callable] = {}
        self.node_values: dict[str, np.ndarray[np.byte]] = {}
        self.evaluation_order: list[str] = []

        self.input_node_ids: set[str] = set[str]()
        self.gate_node_ids: set[str] = set[str]()
        self.output_node_ids: set[str] = set[str]()

        self.num_inputs = None
        self.num_gates = None
        self.num_outputs = None

    @classmethod
    def from_description(cls, description: dict) -> "ComputationGraph":
        graph = cls()

        compute_node_input_edge_pairs = description["compute_node_input_edge_pairs"]
        compute_node_functions = description["compute_node_functions"]

        graph.num_inputs = description["num_inputs"]
        graph.num_gates = len(compute_node_input_edge_pairs)
        graph.num_outputs = description["num_outputs"]

        # Set up input nodes
        for i in range(graph.num_inputs):
            id = f"I{i}"
            graph.input_node_ids.add(id)
            graph.evaluation_order.append(id)
            graph.node_inputs[id] = []
            graph.node_outputs[id] = []
            graph.node_functions[id] = None
            graph.node_values[id] = None

        # Set up gate nodes and their connections
        for i, inputs in enumerate[tuple[int, int]](compute_node_input_edge_pairs):
            id = f"G{i}"
            graph.gate_node_ids.add(id)
            graph.evaluation_order.append(id)
            graph.node_inputs[id] = []
            graph.node_outputs[id] = []
            graph.node_functions[id] = compute_node_functions[i]
            graph.node_values[id] = None

            # Connect inputs based on the edge tuple
            for input_idx in inputs:
                if input_idx < graph.num_inputs:
                    input_id = f"I{input_idx}"
                else:
                    input_id = f"G{input_idx - graph.num_inputs}"
                graph.connect(transmitting_id=input_id, receiving_id=id)

        # Set up output nodes - connecting to the last num_outputs gates
        for i in range(graph.num_outputs):
            id = f"@{i}"
            graph.output_node_ids.add(id)
            graph.evaluation_order.append(id)
            graph.node_inputs[id] = []
            graph.node_outputs[id] = []
            graph.node_functions[id] = None
            graph.node_values[id] = None

            # Connect to one of the last num_outputs gates
            source_id = f"G{graph.num_gates - graph.num_outputs + i}"
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
        assert inputs.shape[0] == len(self.input_node_ids)

        output_values = np.zeros((self.num_outputs, inputs.shape[-1]), dtype=np.uint8)

        for i, input_node_id in enumerate[str](sorted(list[str](self.input_node_ids))):
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

        for i, output_node_id in enumerate[str](
            sorted(list[str](self.output_node_ids))
        ):
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
                f"{self.node_inputs[id]} --> " "{id} --> {self.node_outputs[id]}"
                for id in self.evaluation_order
                if refcounts[id] != 0
            ]
            print("\n")
            print("Cyclic gates:")
            for c in cyclic:
                print(c)
            print("\n")
