from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import BoolTensor, LongTensor, Tensor
from torch.nn import ModuleList
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits

from dagnabbit.bitarrays import output_to_image_array

BitTensor = Tensor
IndicesTensor = LongTensor
MaskTensor = BoolTensor

def advanced_index(tensor: Tensor, indices: IndicesTensor) -> Tensor:
    return tensor[indices]

batched_advanced_index = torch.vmap(advanced_index)


class BipartiteNANDGraphLayer(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
    ):
        super().__init__()
        self.num_inputs: int = num_inputs
        self.num_outputs: int = num_outputs

        self.adjacency_matrix_logits: nn.Parameter = nn.Parameter(
            torch.zeros(2, num_outputs, num_inputs, dtype=torch.float32)
        )

        self.invert_logits: nn.Parameter = nn.Parameter(
            torch.zeros((num_outputs,), dtype=torch.float32)
        )

    def sample_graph_parameters(
        self, stochastic: bool = True, batch_size: int = 1
    ) -> Tuple[IndicesTensor, MaskTensor]:
        # [num_outputs, 2]

        if stochastic:
            connection_probabilities = torch.softmax(
                self.adjacency_matrix_logits, dim=2
            )

            # [2*num_outputs, num_inputs] -> [2*num_outputs, batch_size]
            connection_indices = torch.multinomial(
                connection_probabilities.view(2 * self.num_outputs, self.num_inputs),
                num_samples=batch_size,
                replacement=True,
            )

            # [2*num_outputs, batch_size]
            # -> [batch_size, 2*num_outputs]
            # -> [batch_size, 2, num_outputs]
            # -> [batch_size, num_outputs, 2]
            connection_indices = (
                connection_indices.transpose(0, 1)
                .reshape(batch_size, 2, self.num_outputs)
                .moveaxis(1, 2)
            )

            invert_probabilities = torch.sigmoid(self.invert_logits)
            invert_probabilities = invert_probabilities.unsqueeze(0).expand(batch_size, -1)
            invert_mask = torch.bernoulli(invert_probabilities).bool()

        else:
            # [2, num_outputs]
            connection_indices = torch.argmax(self.adjacency_matrix_logits, dim=2)

            # [2, num_outputs] -> [num_outputs, 2] -> [1, num_outputs, 2]
            connection_indices = connection_indices.transpose(0, 1).unsqueeze(0)

            # [num_outputs] -> [1, num_outputs]
            invert_mask = (torch.sigmoid(self.invert_logits) > 0.5).unsqueeze(0)

        return connection_indices, invert_mask

    def forward(
        self,
        input_bitarrays: BitTensor,
        batch_size: int,
        stochastic: bool = True,
    ) -> Tuple[BitTensor, IndicesTensor, MaskTensor]:
        # input bitarrays: [num_inputs, num_bytes]
        # output bitarrays: [num_outputs, num_bytes]

        if len(input_bitarrays.shape) == 2:
            input_bitarrays = input_bitarrays.unsqueeze(0).expand(batch_size, -1, -1)
  
        # [batch_size, num_outputs, 2], [batch_size, num_outputs]
        connection_indices, invert_mask = self.sample_graph_parameters(
            batch_size=batch_size, stochastic=stochastic
        )

        # [batch_size, num_outputs, 2, num_bytes]
        function_inputs = batched_advanced_index(input_bitarrays, connection_indices)


        # [batch_size, num_outputs, num_bytes]
        and_outputs = torch.bitwise_and(
            function_inputs[:, :, 0, :], function_inputs[:, :, 1, :]
        )

        # [batch_size, num_outputs, num_bytes]
        or_outputs = torch.bitwise_or(
            function_inputs[:, :, 0, :], function_inputs[:, :, 1, :]
        )

        # invert_mask: [batch_size, num_outputs] -> [batch_size, num_outputs, 1]

        function_outputs = torch.bitwise_not(torch.where(
            invert_mask.unsqueeze(-1),
            or_outputs,
            and_outputs,
        ))

        # [batch_size, num_outputs, num_bytes]
        return function_outputs, connection_indices, invert_mask

    def calculate_gradients(
        self,
        connection_indices: IndicesTensor,
        invert_mask: MaskTensor,
        rewards: Tensor,
        batch_size: int,
    ) -> None:

        # connection_indices: [batch_size, num_outputs, 2]
        # invert_mask: [batch_size, num_outputs]
        # rewards: [batch_size]
    
        # [2, num_outputs, num_inputs] -> [num_inputs, num_outputs, 2]
        reshaped_adjacency_matrix_logits: Tensor = self.adjacency_matrix_logits.permute(2, 1, 0)
        # [num_inputs, num_outputs, 2] -> [batch_size, num_inputs, num_outputs, 2]
        reshaped_adjacency_matrix_logits = reshaped_adjacency_matrix_logits.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # input shape expected [batch_size, num_classes, ...] (unnormalized logits)
        # target shape expected [batch_size, ...] (integer class indices)

        # in this case, that means:
        # reshaped_adjacency_matrix_logits: [batch_size, num_inputs, num_outputs, 2]
        # connection_indices: [batch_size, num_outputs, 2]
        # cross_entropy_with_actual_connections: [batch_size, num_outputs, 2]
        cross_entropy_with_actual_connections = cross_entropy(
            input=reshaped_adjacency_matrix_logits,
            target=connection_indices,
            reduction='none',
        )

        rewards = torch.from_numpy(rewards).to(self.adjacency_matrix_logits.device)
        rewards = rewards.view(batch_size, 1, 1)
        weighted_connections_loss = (cross_entropy_with_actual_connections * rewards).mean()

        reshaped_inversion_logits: Tensor = self.invert_logits.unsqueeze(0).expand(batch_size, -1)
        binary_cross_entropy_with_actual_inversion_masks = binary_cross_entropy_with_logits(
            input=reshaped_inversion_logits,
            target=invert_mask.float(),
            reduction='none',
        )
        weighted_inversion_loss = (binary_cross_entropy_with_actual_inversion_masks * rewards).mean()

        total_loss = weighted_connections_loss + weighted_inversion_loss

        total_loss.backward()
        

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

        self.layers: list[BipartiteNANDGraphLayer] = ModuleList(
            [BipartiteNANDGraphLayer(num_inputs, num_neurons_per_layer)]
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                BipartiteNANDGraphLayer(num_neurons_per_layer, num_neurons_per_layer)
            )

        self.layers.append(BipartiteNANDGraphLayer(num_neurons_per_layer, num_outputs))

    # @torch.compile()
    def _forward_compilable(
        self, input_bitarrays: BitTensor, stochastic: bool = True, batch_size: int = 1
    ) -> Tuple[BitTensor, List[IndicesTensor], List[MaskTensor]]:
        connection_indices_list = []
        invert_masks_list = []

        for layer in self.layers:
            input_bitarrays, connection_indices, invert_masks = layer.forward(
                input_bitarrays=input_bitarrays,
                stochastic=stochastic,
                batch_size=batch_size,
            )
            connection_indices_list.append(connection_indices)
            invert_masks_list.append(invert_masks)

        return input_bitarrays, connection_indices_list, invert_masks_list

    def forward(
        self,
        input_bitarrays: BitTensor,
        output_shape: Tuple[int],
        stochastic: bool = True,
        batch_size: int = 1,
    ) -> np.ndarray[np.uint8]:

        if not stochastic and batch_size > 1:
            print(
                "Deterministically sampling graph parameters. Batch size overridden to 1."
            )
            batch_size = 1

        output, connection_indices, invert_mask = self._forward_compilable(
            input_bitarrays=input_bitarrays,
            stochastic=stochastic,
            batch_size=batch_size,
        )

        output_shape = (batch_size, *output_shape)

        output = output.cpu().numpy()
        output = output_to_image_array(output, output_shape)
        return output, connection_indices, invert_mask

    def calculate_gradients(
        self,
        connection_indices: List[IndicesTensor],
        invert_mask: List[MaskTensor],
        rewards: Tensor,
        batch_size: int,
    ) -> None:
        for layer, connection_indices, invert_mask in zip(self.layers, connection_indices, invert_mask):
            layer.calculate_gradients(connection_indices, invert_mask, rewards, batch_size)