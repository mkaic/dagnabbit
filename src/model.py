import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from typing import List, Tuple


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


def binary_to_integer(binary_vectors: Tensor) -> Tensor:
    """Convert batch of binary vectors to integers using big-endian encoding.

    Args:
        binary_vectors: Tensor of shape (batch_size, num_bits) containing 0s and 1s

    Returns:
        Tensor of shape (batch_size,) containing integer values
    """
    powers = 2 ** torch.arange(
        binary_vectors.shape[-1] - 1, -1, -1, device=binary_vectors.device
    )
    return (binary_vectors * powers).sum(dim=-1)
