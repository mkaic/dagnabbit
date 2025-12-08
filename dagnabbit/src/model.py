import numpy as np
import torch
import torch.nn as nn
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


class NeuralImplicitComputationGraph(nn.Module):

    def __init__(
        self,
        num_input_nodes,
        num_compute_nodes,
        num_output_nodes,
        num_layers,
        position_encoding_dim,
        hidden_dim,
        recurrent_dim,
        num_prior_gates_connectable,
    ):

        super().__init__()

        self.num_input_nodes = num_input_nodes
        self.num_compute_nodes = num_compute_nodes
        self.num_output_nodes = num_output_nodes
        self.num_layers = num_layers
        self.position_encoding_dim = position_encoding_dim
        self.hidden_dim = hidden_dim
        self.recurrent_dim = recurrent_dim
        self.num_prior_gates_connectable = num_prior_gates_connectable

        # one position encoding for the gate being chose
        # one to represent the last edge choice
        # one to represent the last function choice
        #
        self.input_projection = nn.Linear(
            (self.position_encoding_dim * 3) + self.recurrent_dim, self.hidden_dim
        )

        self.mlp = nn.Sequential()
        for _ in range(num_layers):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.GELU())

        self.edge_choice_projection = nn.Linear(
            self.hidden_dim, self.num_input_nodes + self.num_compute_nodes
        )

        self.function_choice_projection = nn.Linear(
            self.hidden_dim, len(gate_functions.AVAILABLE_FUNCTIONS)
        )

        self.recurrent_projection = nn.Linear(self.hidden_dim, self.recurrent_dim)

        self.graph: ComputationGraph = None

        self.initial_recurrent_logits = nn.Parameter(torch.randn(self.recurrent_dim))

    # @torch.compile()
    def mlp_forward(self, position_encodings: Tensor) -> torch.LongTensor:
        # [num_compute_nodes * 2, num_input_nodes + num_compute_nodes]

        edge_choices = []
        function_choices = []

        last_edge_idx_chosen = 0
        last_function_idx_chosen = 0

        recurrent_logits = self.initial_recurrent_logits
        for compute_gate_idx in range(len(position_encodings)):

            inputs = torch.cat(
                [
                    position_encodings[compute_gate_idx],
                    position_encodings[last_edge_idx_chosen],
                    position_encodings[last_function_idx_chosen],
                    recurrent_logits,
                ]
            )
            inputs = self.input_projection(inputs)

            hidden_logits = self.mlp(inputs)

            # Each compute node has two inputs, therefore the neural network outputs two consecutive
            # decisions about which two *previous* nodes will be those inputs. Valid options
            # include previous compute nodes and the input nodes.
            # As a concrete example, compute node 3 will be able to be connect
            # to the input nodes and computen nodes 0, 1, and 2.

            # We set invalid indices to -inf so that they end up as 0 after softmax
            # We add address_bitdepth here to allow

            # the idx // 2 is because the neural net is evaluated twice for each node, so there are
            # two output logits — one for each input to the node.
            edge_choice_logits = self.edge_choice_projection(hidden_logits)

            mask_after = self.num_input_nodes + (compute_gate_idx // 2)
            edge_choice_logits[mask_after:] = -torch.inf
            mask_before = int(
                max(0, (compute_gate_idx // 2) - self.num_prior_gates_connectable)
            )
            edge_choice_logits[:mask_before] = -torch.inf

            edge_choice = torch.multinomial(
                torch.softmax(edge_choice_logits, dim=-1), 1
            ).squeeze(-1)
            edge_choices.append(edge_choice)
            last_edge_idx_chosen = edge_choice

            # When we choose the first input for a gate, we should also choose its function.
            if (compute_gate_idx % 2) == 0:
                function_choice_logits = self.function_choice_projection(hidden_logits)
                function_choice = torch.multinomial(
                    torch.softmax(function_choice_logits, dim=-1), 1
                ).squeeze(-1)
                function_choices.append(function_choice)
                last_function_idx_chosen = function_choice

            #
            recurrent_logits = torch.nn.functional.normalize(
                recurrent_logits + self.recurrent_projection(hidden_logits), dim=0
            )

        return edge_choices, function_choices

    def sample_new_graph(self, position_encodings: Tensor) -> ComputationGraph:

        edge_choices, function_choices = self.mlp_forward(position_encodings)

        # Convert samples into edges list
        compute_node_input_edge_pairs = []
        compute_node_functions = [
            gate_functions.AVAILABLE_FUNCTIONS[fn_idx] for fn_idx in function_choices
        ]

        # Group samples into pairs to form edges pointing into each compute node
        sample_pairs = zip(edge_choices[::2], edge_choices[1::2])
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

        graph = self.sample_new_graph(position_encodings=position_encodings)

        output = graph.evaluate(address_bitarrays)
        output = output_to_image_array(output, output_shape)

        return output
