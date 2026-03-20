import torch
from torch import Tensor
import torch.nn as nn
from typing import Iterable

from dagnabbit.dag.description import FixedInDegreeDAGDescription


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
        for i, layer in enumerate[nn.Module](self.layers):
            x = layer(x)
            if i + 1 < len(self.layers):
                x = self.activation(x)

        return x


class NodeEncoder(nn.Module):
    def __init__(self, node_embedding_dim: int, in_degree: int):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.in_degree = in_degree
        self.encoder = MLP(
            [node_embedding_dim * in_degree, node_embedding_dim * 2, node_embedding_dim]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class NodeDecoder(nn.Module):
    def __init__(self, node_embedding_dim: int, in_degree: int):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.in_degree = in_degree
        self.decoder = MLP(
            [node_embedding_dim, node_embedding_dim * 2, node_embedding_dim * in_degree]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


class FixedInDegreeNodeAutoEncoder(nn.Module):
    def __init__(self, node_embedding_dim: int, in_degree):
        super().__init__()

        self.node_embedding_dim = node_embedding_dim

        self.encoder = NodeEncoder(node_embedding_dim, in_degree)
        self.decoder = NodeDecoder(node_embedding_dim, in_degree)

    def encode(
        self,
        input_node_embeddings: Tensor,
    ) -> Tensor:
        return self.encoder(input_node_embeddings)

    def decode(self, node_embedding: Tensor) -> Tensor:
        return self.decoder(node_embedding)


class DagnabbitAutoEncoder(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        trunk_node_in_degrees: int | list[int],
        num_trunk_node_types: int,
        condenser_node_in_degree: int,
        num_root_nodes: int,
    ):
        super().__init__()

        self.node_embedding_dim = node_embedding_dim

        if isinstance(trunk_node_in_degrees, int):
            self.trunk_node_in_degrees = [trunk_node_in_degrees] * num_trunk_node_types
        else:
            self.trunk_node_in_degrees = trunk_node_in_degrees

        assert len(self.trunk_node_in_degrees) == num_trunk_node_types

        self.num_trunk_node_types = num_trunk_node_types
        self.condenser_node_in_degree = condenser_node_in_degree
        self.num_root_nodes = num_root_nodes

        self.node_autoencoders = nn.ModuleList()

        for _ in range(num_trunk_node_types + 1):  # +1 for condenser nodes
            pass

        self.node_type_predictor = MLP(
            [
                self.node_embedding_dim,
                self.node_embedding_dim * 2,
                self.num_trunk_node_types,
            ]
        )
        self.node_is_roots_predictor = MLP(
            [
                self.node_embedding_dim,
                self.node_embedding_dim * 2,
                self.num_root_nodes,
            ]
        )

    def encode(
        self,
        primary_graph: FixedInDegreeDAGDescription,
        condenser_graph: FixedInDegreeDAGDescription,
    ) -> Tensor:
        pass
