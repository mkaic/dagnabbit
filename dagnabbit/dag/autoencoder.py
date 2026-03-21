from torch._tensor import Tensor
from torch._tensor import Tensor
import torch
from torch import Tensor
import torch.nn as nn
from typing import Iterable

from dagnabbit.dag.description import FixedInDegreeDAGDescription, make_condenser_graph_description, graft_condenser_graph_onto_primary_graph


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
            trunk_node_in_degrees = [trunk_node_in_degrees] * num_trunk_node_types
        else:
            trunk_node_in_degrees = trunk_node_in_degrees

        assert len(trunk_node_in_degrees) == num_trunk_node_types

        self.trunk_node_in_degrees = trunk_node_in_degrees + [condenser_node_in_degree]

        self.num_trunk_node_types = num_trunk_node_types + 1

        self.num_root_nodes = num_root_nodes

        self.node_autoencoders: list[FixedInDegreeNodeAutoEncoder] = nn.ModuleList()

        for node_type_idx, in_degree in zip[tuple[int, int]](
            range(num_trunk_node_types), self.trunk_node_in_degrees
        ):  # +1 for condenser nodes
            self.node_autoencoders.append(
                FixedInDegreeNodeAutoEncoder(node_embedding_dim, in_degree)
            )

        # For the purposes of this aux loss, here each input node has it's own unique "type"
        # Meaning we need to add num_root_nodes neurons to the output
        self.node_type_predictor = MLP(
            [
                self.node_embedding_dim,
                self.node_embedding_dim * 2,
                self.num_trunk_node_types + self.num_root_nodes,
            ]
        )

        self.root_node_embeddings = torch.randn(self.num_root_nodes, self.node_embedding_dim, dtype=torch.float32)

    def encode(
        self,
        primary_graph: FixedInDegreeDAGDescription,
    ) -> tuple[Tensor, FixedInDegreeDAGDescription]:

        if len(primary_graph.leaf_node_indices) !- 1:
            condenser_graph = make_condenser_graph_description(primary_graph)
            graph = graft_condenser_graph_onto_primary_graph(primary_graph=primary_graph, condenser_graph=condenser_graph)
        else:
            graph = primary_graph

        trunk_node_embeddings = torch.empty((graph.num_trunk_nodes, self.node_embedding_dim), dtype=torch.float32)

        embeddings_buffer = torch.cat([self.root_node_embeddings, trunk_node_embeddings], dim=0)

        for i in range(graph.num_trunk_nodes):
           
            buffer_read_indices = graph.trunk_node_inputs_indices[i]

            node_type = graph.trunk_node_types[i]
            node_autoencoder = self.node_autoencoders[node_type]

            inputs = embeddings_buffer[buffer_read_indices, :]

            node_embedding = node_autoencoder.encode(inputs.flatten())

            buffer_write_index = i + self.num_root_nodes
            embeddings_buffer[buffer_write_index] = node_embedding

        return embeddings_buffer, graph

    def decode_autoregressive(self, graph_embedding, graph):
        pass