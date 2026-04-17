import torch
from torch import Tensor
import torch.nn as nn
from typing import Iterable

from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_condenser_graph_description,
)


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
        x = x.flatten()
        x = self.encoder(x)

        return x


class NodeDecoder(nn.Module):
    def __init__(self, node_embedding_dim: int, in_degree: int):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.in_degree = in_degree
        self.decoder = MLP(
            [node_embedding_dim, node_embedding_dim * 2, node_embedding_dim * in_degree]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.decoder(x)
        x = torch.chunk(x, self.in_degree, dim=0)

        return x


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
        num_output_nodes: int,
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
        self.num_output_nodes = num_output_nodes

        self.node_autoencoders: list[FixedInDegreeNodeAutoEncoder] = nn.ModuleList()

        for node_type_idx, in_degree in zip[tuple[int, int]](
            range(num_trunk_node_types), self.trunk_node_in_degrees
        ):  # +1 for condenser nodes
            self.node_autoencoders.append(
                FixedInDegreeNodeAutoEncoder(node_embedding_dim, in_degree)
            )

        # Single shared autoencoder for all output nodes. Its in_degree is 2:
        # one slot for the output node's single graph-parent embedding, and
        # one slot for a learnable per-output-slot embedding.
        self.output_autoencoder = FixedInDegreeNodeAutoEncoder(
            node_embedding_dim, in_degree=2
        )
        self.output_node_embeddings = nn.Embedding(
            self.num_output_nodes, node_embedding_dim
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

        self.root_node_embeddings = nn.Embedding(self.num_root_nodes)

    def evaluate_graph(
        self,
        graph: FixedInDegreeDAGDescription,
        root_node_embeddings: Tensor,
    ) -> Tensor:
        assert graph.num_output_nodes <= self.num_output_nodes

        embeddings_buffer = torch.empty(
            (graph.num_nodes, self.node_embedding_dim),
            dtype=torch.float32,
        )
        embeddings_buffer[: graph.num_root_nodes] = root_node_embeddings

        for node_idx in range(graph.num_root_nodes, graph.num_nodes):
            buffer_read_indices = graph.node_inputs_indices[node_idx]
            parent_embeddings = embeddings_buffer[buffer_read_indices, :]

            node_type = graph.node_types[node_idx]

            if node_type >= graph.output_node_types_start:
                # Output nodes are guaranteed leaves with a single graph-parent.
                # Every output slot has its own unique type index, so the slot
                # identity is derived from the node type. All output nodes
                # share a single in-degree-2 autoencoder that combines the
                # parent embedding with a learnable per-slot embedding.
                output_slot_idx = node_type - graph.output_node_types_start
                slot_embedding = self.output_node_embeddings.weight[
                    output_slot_idx
                ].unsqueeze(0)
                node_autoencoder = self.output_autoencoder
                encode_input = torch.cat([parent_embeddings, slot_embedding], dim=0)
            else:
                node_autoencoder = self.node_autoencoders[node_type]
                encode_input = parent_embeddings

            embeddings_buffer[node_idx] = node_autoencoder.encode(encode_input)

        return embeddings_buffer

    def training_forward(
        self,
        primary_graph: FixedInDegreeDAGDescription,
    ):

        # Encode

        primary_buffer = self.evaluate_graph(primary_graph, self.root_node_embeddings.weight)

        if len(primary_graph.leaf_node_indices) > 1:
            condenser_graph = make_condenser_graph_description(primary_graph)
            leaf_embeddings = primary_buffer[primary_graph.leaf_node_indices]
            condenser_buffer = self.evaluate_graph(condenser_graph, leaf_embeddings)
            graph_embedding = condenser_buffer[-1]
        else:
            graph_embedding = primary_buffer[primary_graph.leaf_node_indices]

        # Guided Autoregressive Decode

        condenser_decode_buffer = [{'predicted_node_type': None, 'predicted_parent_embeddings': [], } for _ in range(condenser_graph.num_nodes)]

        condenser_decode_buffer[-1] = graph_embedding

        for node_idx in reversed(range(condenser_graph.num_nodes)):
            pass



    def inference_embed_graph(
        self,
        primary_graph: FixedInDegreeDAGDescription,
    ) -> tuple[Tensor, Tensor | None, FixedInDegreeDAGDescription | None]:
        primary_buffer = self.evaluate_graph(
            primary_graph, self.root_node_embeddings.weight
        )

        if len(primary_graph.leaf_node_indices) > 1:
            condenser_graph = make_condenser_graph_description(primary_graph)
            leaf_embeddings = primary_buffer[primary_graph.leaf_node_indices]
            condenser_buffer = self.evaluate_graph(condenser_graph, leaf_embeddings)
            graph_embedding = condenser_buffer[-1]
        else:
            graph_embedding = primary_buffer[primary_graph.leaf_node_indices]

        return graph_embedding


    def inference_decode_blind_autoregressive(
        self,
        graph_embedding: Tensor,
    ):
        pass