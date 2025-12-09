from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch import Tensor, LongTensor, BoolTensor

from dagnabbit.bitarrays import output_to_image_array

BitTensor = Tensor


class BipartiteNANDGraphLayer(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.num_inputs: int = num_inputs
        self.num_outputs: int = num_outputs
        # [num_outputs, 2]
        self.output_node_input_indices: LongTensor = nn.Parameter(
            torch.randint(
                0,
                num_inputs,
                (num_outputs, 2),
                dtype=torch.long,
            ),
            requires_grad=False,
        )
        self.nor_mask: BoolTensor = nn.Parameter(
            torch.zeros(num_outputs, dtype=torch.bool), requires_grad=False
        )

    def forward(self, input_bitarrays: BitTensor) -> BitTensor:
        # input bitarrays: [num_inputs, num_bytes]
        # output bitarrays: [num_outputs, num_bytes]

        # [num_outputs, 2, num_bytes]
        function_inputs = input_bitarrays[self.output_node_input_indices]

        # [num_outputs, num_bytes]
        and_outputs = torch.bitwise_and(
            function_inputs[:, 0, :], function_inputs[:, 1, :]
        )

        or_outputs = torch.bitwise_or(
            function_inputs[:, 0, :], function_inputs[:, 1, :]
        )

        # invert_mask: [num_outputs] -> [num_outputs, 1]
        # function_outputs: [num_outputs, num_bytes] -> [num_outputs, num_bytes]
        function_outputs = torch.bitwise_not(
            torch.where(
                self.nor_mask.unsqueeze(-1),
                or_outputs,
                and_outputs,
            )
        )

        return function_outputs


class LayeredNANDGraph(nn.Module):
    """Heavily restricted subcase of ComputationGraph where the adjacency matrix is nearly block-diagonal. Basically an MLP, but neurons are 2-sparse NAND gates."""

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_layers: int,
        num_neurons_per_layer: int,
    ):
        super().__init__()
        self.num_inputs: int = num_inputs
        self.num_outputs: int = num_outputs
        self.num_layers: int = num_layers
        self.num_neurons_per_layer: int = num_neurons_per_layer

        self.layers: list[BipartiteNANDGraphLayer] = [
            BipartiteNANDGraphLayer(num_inputs, num_neurons_per_layer)
        ]
        self.layer_logits: ModuleList[BipartiteNANDGraphLayerLogits] = ModuleList(
            [BipartiteNANDGraphLayerLogits(num_inputs, num_neurons_per_layer)]
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                BipartiteNANDGraphLayer(num_neurons_per_layer, num_neurons_per_layer)
            )
            self.layer_logits.append(
                BipartiteNANDGraphLayerLogits(
                    num_neurons_per_layer, num_neurons_per_layer
                )
            )

        self.layers.append(BipartiteNANDGraphLayer(num_neurons_per_layer, num_outputs))
        self.layer_logits.append(
            BipartiteNANDGraphLayerLogits(num_neurons_per_layer, num_outputs)
        )

    @torch.compile()
    def _forward_compilable(self, input_bitarrays: BitTensor) -> BitTensor:
        for layer in self.layers:
            input_bitarrays = layer(input_bitarrays)
        return input_bitarrays

    def forward(
        self, input_bitarrays: BitTensor, output_shape: Tuple[int]
    ) -> BitTensor:

        self.resample_layers_stochastic()

        output = self._forward_compilable(input_bitarrays)

        output = output.cpu().numpy()
        output = output_to_image_array(output, output_shape)
        return output

    def resample_layers_stochastic(self) -> None:
        for i, (layer, layer_logits) in enumerate(zip(self.layers, self.layer_logits)):
            indices, nor_mask = layer_logits.sample_stochastic()
            layer.output_node_input_indices.data = indices
            layer.nor_mask.data = nor_mask.bool()


class BipartiteNANDGraphLayerLogits(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
    ):
        super().__init__()
        self.num_inputs: int = num_inputs
        self.num_outputs: int = num_outputs

        self.adjacency_probability_matrix: nn.Parameter = nn.Parameter(
            torch.randn(num_outputs, num_inputs, dtype=torch.float32)
        )
        
        self.adjacency_temperature: float = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.nor_probability: nn.Parameter = nn.Parameter(
            torch.randn((num_outputs,), dtype=torch.float32)
        )

        self.nor_temperature: float = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def sample_stochastic(self) -> LongTensor:
        # [num_outputs, 2]
        return (
            torch.multinomial(
                torch.softmax(self.adjacency_probability_matrix * self.adjacency_temperature, dim=-1),
                num_samples=2,
                replacement=True,
            ),
            torch.bernoulli(torch.sigmoid(self.nor_probability * self.nor_temperature)),
        )
