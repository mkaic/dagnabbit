from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor

class BipartiteComputationGraphLayer:
    def __init__(self, num_inputs: int, num_outputs: int, gate_functions: list[Callable]):
        self.num_inputs: int = num_inputs
        self.num_outputs: int = num_outputs
        self.gate_functions: list[Callable] = gate_functions
        # [num_outputs, 2]
        self.output_node_input_indices: Tensor = torch.randint(0, num_inputs, (num_outputs, 2))

class LayeredComputationGraph:
    """Heavily restricted subcase of ComputationGraph where the adjacency matrix is nearly block-diagonal. Basically an MLP, but neurons are 2-sparse binary logic gates."""
    def __init__(self):
        self.num_inputs: int = None
        self.num_outputs: int = None
        self.num_layers: int = None
        self.num_neurons_per_layer: int = None

