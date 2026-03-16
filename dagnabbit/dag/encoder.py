import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable


class MLP(nn.Module):
    def __init__(self, vector_dims: Iterable[int], activation: nn.Module = nn.GELU()):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Linear(vector_dims[i], vector_dims[i + 1])
                for i in range(len(vector_dims) - 1)
            ]
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


class FixedInDegreeNodeAutoEncoder(nn.Module):
    def __init__(self, node_embedding_dim: int, in_degree: int):
        super().__init__()

        self.node_embedding_dim = node_embedding_dim
        input_dim = node_embedding_dim * in_degree
        self.encoder = MLP([input_dim, input_dim * 2, node_embedding_dim])
        self.decoder = MLP([node_embedding_dim, input_dim * 2, input_dim])

    def encode(self, input_node_embeddings: Tensor, ) -> Tensor:
        return self.encoder(input_node_embeddings)

    def decode(self, node_embedding: Tensor) -> Tensor:
        return self.decoder(node_embedding)


class NodeAECollection(nn.Module):
    def __init__(self, node_embedding_dim: int, in_degree: int, num_node_types: int):
        super().__init__()

        self.node_aes = nn.ModuleList([
            FixedInDegreeNodeAutoEncoder(node_embedding_dim, in_degree)
            for _ in range(num_node_types)
        ])