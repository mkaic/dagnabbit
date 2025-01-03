from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor


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


def get_binary_position_encodings(length: int, dim: int) -> Tensor:
    """Generate binary position encodings of shape (length, dim).

    Each position index is converted to binary and scaled to [-1, +1].
    If dim is larger than needed for binary representation, left-pads with -1.

    Args:
        length: Number of positions to encode
        dim: Dimension of encoding vectors

    Returns:
        Tensor of shape (length, dim) containing -1s and +1s
    """
    # Convert each position to binary
    indices = np.arange(length)
    bin_strings = [format(i, "b").zfill(dim) for i in indices]

    # Convert to tensor of 1s and 0s
    encodings = torch.tensor([[int(b) for b in s[-dim:]] for s in bin_strings])

    # Scale from {0,1} to {-1,+1}
    encodings = encodings * 2 - 1

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


def save_if_best_rmse(rmse, best_rmse, output, last_updated_at, update_counter, step):
    if rmse < best_rmse:
        best_rmse = rmse
        print(f"RMSE: {best_rmse:.5f} | Step: {step:04} | Saved: {last_updated_at:.5f}")

        if best_rmse < last_updated_at * 0.995:

            output_pil = Image.fromarray(np.moveaxis(output, 0, -1))
            output_pil.save(
                "dagnabbit/outputs/output.jpg",
                format="JPEG",
                subsampling=0,
                quality=100,
            )
            output_pil.save(
                f"dagnabbit/outputs/timelapse/{update_counter:06}.jpg",
                format="JPEG",
                subsampling=0,
                quality=100,
            )

            last_updated_at = best_rmse
            update_counter += 1

    return best_rmse, last_updated_at, update_counter
