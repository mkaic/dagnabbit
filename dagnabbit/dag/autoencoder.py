import torch
from torch import Tensor
import torch.nn as nn
from typing import Iterable, Callable
from functools import partial
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


class OutputNodeAutoEncoder(nn.Module):
    def __init__(self, node_embedding_dim: int, num_output_nodes: int):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.encoder = NodeEncoder(node_embedding_dim, in_degree=2)
        self.decoder = NodeDecoder(node_embedding_dim, in_degree=1)
        self.output_node_embeddings = nn.Embedding(num_output_nodes, node_embedding_dim)

    def encode(self, input_node_embeddings: Tensor, output_slot_idx: int) -> Tensor:

        output_slot_embedding = self.output_node_embeddings.weight[output_slot_idx]
        x = torch.cat([input_node_embeddings, output_slot_embedding], dim=0)

        return self.encoder(x)

    def decode(self, node_embedding: Tensor) -> Tensor:

        return self.decoder(node_embedding)


class DagnabbitAutoEncoder(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        trunk_node_type_in_degrees: int | list[int],
        num_trunk_node_types: int,
        condenser_node_type_in_degree: int,
        num_root_nodes: int,
        num_output_nodes: int,
    ):
        super().__init__()

        if isinstance(trunk_node_type_in_degrees, int):
            trunk_node_type_in_degrees = [
                trunk_node_type_in_degrees
            ] * num_trunk_node_types
        assert len(trunk_node_type_in_degrees) == num_trunk_node_types

        self.node_embedding_dim = node_embedding_dim
        self.num_trunk_node_types = num_trunk_node_types
        self.num_root_nodes = num_root_nodes
        self.num_output_nodes = num_output_nodes
        self.trunk_node_in_degrees = trunk_node_type_in_degrees
        self.condenser_node_in_degree = condenser_node_type_in_degree

        # ---- Trunk node autoencoders (one per node type) ----
        self.node_autoencoders: dict[int, FixedInDegreeNodeAutoEncoder] = (
            nn.ModuleList()
        )
        for node_type_idx, in_degree in (
            range(num_trunk_node_types),
            self.trunk_node_in_degrees,
        ):
            self.node_autoencoders[node_type_idx] = FixedInDegreeNodeAutoEncoder(
                node_embedding_dim, in_degree
            )

        # ---- Output autoencoder (shared across all output slots) ----
        # in_degree is 2: one slot for the output node's single graph-parent
        # embedding, and one slot for a learnable per-output-slot embedding.
        self.output_autoencoder = OutputNodeAutoEncoder(node_embedding_dim, in_degree=2)

        for output_slot_idx in range(num_output_nodes):
            type_idx = output_slot_idx + num_trunk_node_types + num_root_nodes
            self.node_autoencoders[type_idx] = partial(
                self.output_autoencoder.encode, output_slot_idx=output_slot_idx
            )

        # ---- Condenser node autoencoder (shared across all condenser nodes) ----
        self.condenser_node_autoencoder = FixedInDegreeNodeAutoEncoder(
            node_embedding_dim, in_degree=condenser_node_type_in_degree
        )

        # ---- Root node embeddings ----
        self.root_node_embeddings = nn.Embedding(self.num_root_nodes)

        # ---- Auxiliary node-type prediction head ----
        # For the purposes of this aux loss, each root node has its own unique
        # "type", so num_root_nodes extra output neurons are appended.
        self.node_type_predictor = MLP(
            [
                self.node_embedding_dim,
                self.node_embedding_dim * 2,
                self.num_node_types,
            ]
        )

    @staticmethod
    def evaluate_graph(
        graph: FixedInDegreeDAGDescription,
        root_node_embeddings: Tensor,
        node_autoencoders: dict[int, Callable[[Tensor], Tensor]],
        node_embedding_dim: int,
    ) -> Tensor:
        assert graph.num_output_nodes <= graph.num_output_nodes

        embeddings_buffer = torch.empty(
            (graph.num_nodes, node_embedding_dim),
            dtype=torch.float32,
        )
        embeddings_buffer[: graph.num_root_nodes] = root_node_embeddings

        for node_idx in range(graph.num_root_nodes, graph.num_nodes):
            buffer_read_indices = graph.node_inputs_indices[node_idx]
            parent_embeddings = embeddings_buffer[buffer_read_indices, :]

            node_type = graph.node_types[node_idx]

            node_autoencoder = node_autoencoders[node_type]

            embeddings_buffer[node_idx] = node_autoencoder.encode(parent_embeddings)

        return embeddings_buffer

    def encode_graph_with_condenser(
        self,
        primary_graph: FixedInDegreeDAGDescription,
    ) -> tuple[FixedInDegreeDAGDescription, Tensor]:
        primary_buffer = self.evaluate_graph(
            graph=primary_graph,
            root_node_embeddings=self.root_node_embeddings.weight,
            node_autoencoders=self.node_autoencoders,
            node_embedding_dim=self.node_embedding_dim,
        )

        condenser_graph = make_condenser_graph_description(primary_graph)
        leaf_embeddings = primary_buffer[primary_graph.leaf_node_indices]
        condenser_buffer = self.evaluate_graph(
            graph=condenser_graph,
            root_node_embeddings=leaf_embeddings,
            node_autoencoders={0: self.condenser_node_autoencoder},
            node_embedding_dim=self.node_embedding_dim,
        )
        graph_embedding = condenser_buffer[condenser_graph.leaf_node_indices]
        return condenser_graph, graph_embedding

    def training_forward(
        self,
        primary_graph: FixedInDegreeDAGDescription,
    ):

        # Encode
        condenser_graph, graph_embedding = (
            self.encode_graph_with_condenser(primary_graph)
        )

        # Guided Autoregressive Decode

        condenser_decode_buffer = [
            {
                "predicted_node_type": None,
                "predicted_parent_embeddings": [],
            }
            for _ in range(condenser_graph.num_nodes)
        ]

        condenser_decode_buffer[-1] = {
            "predicted_node_type": None,
            "predicted_parent_embeddings": [graph_embedding],
        }

        for node_idx in reversed(range(condenser_graph.num_nodes)):
            this_node_predicted_embeddings = condenser_decode_buffer[node_idx][
                "predicted_parent_embeddings"
            ]

    def inference_decode_blind_autoregressive(
        self,
        graph_embedding: Tensor,
    ):
        pass
