from turtle import position
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from dagnabbit.src.dag import ComputationGraph
from dagnabbit.src import gate_functions
from dagnabbit.src.bitarrays import output_to_image_array


def get_sinusoidal_position_encodings(
    length: int, dim: int, base: int = 10000
) -> Tensor:
    """Generate sinusoidal position encodings of shape (length, dim)."""
    position = torch.arange(length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(base) / dim))
    encodings = torch.zeros(length, dim)
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)
    return encodings


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        activation=nn.ReLU,
    ):
        """Multi-layer perceptron with configurable sizes and activation function."""
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )
        self.activation = activation()

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        logits = self.layers[-1](x)

        return logits


class NeuralImplicitComputationGraph(nn.Module):

    def __init__(
        self,
        num_input_nodes,
        num_compute_nodes,
        num_output_nodes,
        num_layers,
        input_dim,
        hidden_dim,
    ):

        self.num_input_nodes = num_input_nodes
        self.num_compute_nodes = num_compute_nodes
        self.num_output_nodes = num_output_nodes
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mlp = MLP(
            layer_sizes=[
                self.input_dim,
                *[self.hidden_dim for _ in range(6)],
                self.num_input_nodes + self.num_compute_nodes,
            ],
            activation=torch.nn.GELU,
        )

        self.graph: ComputationGraph = None

    @torch.compile()
    def mlp_forward(self, position_encodings: Tensor) -> torch.LongTensor:
        # [num_compute_nodes, num_input_nodes + num_compute_nodes]
        logits = self.mlp(position_encodings)

        # Each compute node has two inputs, therefore the neural network outputs two consecutive
        # decisions about which two *previous* nodes will be those inputs. Valid options
        # include previous compute nodes and the input nodes.
        # As a concrete example, compute node 3 will be able to be connect
        # to the input nodes and computen nodes 0, 1, and 2.

        # We set invalid indices to -inf so that they end up as 0 after softmax
        # We add address_bitdepth here to allow
        for i in range(len(logits)):

            # the i // 2 is because the neural net is evaluated twice for each node, so there are
            # two output logits — one for each input to the node.
            mask_after = self.num_input_nodes + (i // 2)
            logits[i, mask_after:] = -torch.inf

        probabilities = torch.softmax(logits, dim=-1)

        samples = torch.multinomial(probabilities, 1)
        samples = samples.flatten()

        return samples

    def sample_new_graph(self, position_encodings: Tensor) -> ComputationGraph:

        model_decisions = self.mlp_forward(position_encodings)
        # Convert samples into edges list
        compute_node_input_edge_pairs = []
        compute_node_functions = [
            gate_functions.NP_NAND for _ in range(self.num_compute_nodes)
        ]

        # Group samples into pairs to form edges pointing into each compute node
        sample_pairs = zip[tuple](model_decisions[::2], model_decisions[1::2])
        for a, b in sample_pairs:
            compute_node_input_edge_pairs.append((a.item(), b.item()))

        description = {
            "num_inputs": self.num_input_nodes,
            "num_outputs": self.num_output_nodes,
            "compute_node_input_edge_pairs": compute_node_input_edge_pairs,
            "compute_node_functions": compute_node_functions,
        }

        return ComputationGraph.from_description(description=description)

    def forward(
        self,
        position_encodings: Tensor,
        address_bitarrays: np.ndarray[np.uint8],
        output_shape: tuple,
    ):

        graph = self.sample_new_graph(pos_enc=position_encodings)

        output = graph.evaluate(address_bitarrays)
        output = output_to_image_array(output, output_shape)

        return output
