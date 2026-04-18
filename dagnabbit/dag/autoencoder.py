from abc import ABC, abstractmethod
import torch
from torch import Tensor
import torch.nn as nn
from typing import Iterable, Callable
from functools import partial
from dataclasses import dataclass
from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_condenser_graph_description,
)
import torch.nn.functional as F


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


class NodeAutoEncoder(nn.Module):
    def encode(self, input_node_embeddings: Tensor) -> Tensor:
        raise NotImplementedError

    def decode(self, node_embedding: Tensor) -> Tensor:
        raise NotImplementedError


class FixedInDegreeNodeAutoEncoder(NodeAutoEncoder):
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


class OutputNodeAutoEncoder(NodeAutoEncoder):
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


@dataclass
class TrainingDecodeBufferEntry:
    embeddings_predicted_by_children: list[Tensor]
    classification_loss: Tensor | None
    child_predicted_embeddings_similarity_loss: Tensor | None


@dataclass
class TrainingStepLossReturnType:
    condenser_node_classification_losses: list[Tensor]
    condenser_node_predicted_embeddings_similarity_losses: list[Tensor]
    primary_node_classification_losses: list[Tensor]
    primary_node_predicted_embeddings_similarity_losses: list[Tensor]


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
        node_autoencoders: dict[int, NodeAutoEncoder],
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
    ) -> TrainingStepLossReturnType:
        """Returns losses for the training step."""

        # Encode
        condenser_graph, graph_embedding = self.encode_graph_with_condenser(
            primary_graph
        )

        # Guided Autoregressive Decode

        condenser_decode_buffer = [
            TrainingDecodeBufferEntry(
                embeddings_predicted_by_children=[],
                child_predicted_embeddings_similarity_loss=None,
                classification_loss=None,
            )
            for _ in range(condenser_graph.num_nodes)
        ]

        condenser_decode_buffer[-1] = TrainingDecodeBufferEntry(
            embeddings_predicted_by_children=[graph_embedding],
            child_predicted_embeddings_similarity_loss=torch.std(
                graph_embedding, dim=0
            ),
            classification_loss=None,
        )

        for node_idx in reversed(
            range(condenser_graph.num_root_nodes, condenser_graph.num_nodes)
        ):
            self._decode_step(
                node_idx=node_idx,
                graph=condenser_graph,
                decode_buffer=condenser_decode_buffer,
                node_autoencoders={0: self.condenser_node_autoencoder},
            )

        # Transplant the decoder buffer entries for the condenser graph inputs
        # into their corresponding primary graph leaf node spots. Condenser
        # root `i` is the same semantic node as primary node
        # `primary_graph.leaf_node_indices[i]` (see `encode_graph_with_condenser`),
        # so predictions accumulated at condenser-root entries are predictions
        # of those primary leaves.
        primary_decode_buffer = [
            TrainingDecodeBufferEntry(
                embeddings_predicted_by_children=[],
                child_predicted_embeddings_similarity_loss=None,
                classification_loss=None,
            )
            for _ in range(primary_graph.num_nodes)
        ]

        for condenser_root_idx in range(condenser_graph.num_root_nodes):
            primary_leaf_idx = primary_graph.leaf_node_indices[condenser_root_idx]
            primary_decode_buffer[
                primary_leaf_idx
            ].embeddings_predicted_by_children = condenser_decode_buffer[
                condenser_root_idx
            ].embeddings_predicted_by_children

        for node_idx in reversed(range(primary_graph.num_nodes)):
            self._decode_step(
                node_idx=node_idx,
                graph=primary_graph,
                decode_buffer=primary_decode_buffer,
                node_autoencoders=self.node_autoencoders,
            )

        condenser_trunk_entries = condenser_decode_buffer[
            condenser_graph.num_root_nodes :
        ]

        return TrainingStepLossReturnType(
            condenser_node_classification_losses=[
                e.classification_loss for e in condenser_trunk_entries
            ],
            condenser_node_predicted_embeddings_similarity_losses=[
                e.child_predicted_embeddings_similarity_loss
                for e in condenser_trunk_entries
            ],
            primary_node_classification_losses=[
                e.classification_loss for e in primary_decode_buffer
            ],
            primary_node_predicted_embeddings_similarity_losses=[
                e.child_predicted_embeddings_similarity_loss
                for e in primary_decode_buffer
            ],
        )

    def _decode_step(
        self,
        node_idx: int,
        graph: FixedInDegreeDAGDescription,
        decode_buffer: list[TrainingDecodeBufferEntry],
        node_autoencoders: dict[int, Callable[[Tensor], Tensor]],
    ) -> None:
        """Populate `decode_buffer[node_idx]`'s losses and push this node's
        predicted parent embeddings into its parents' buffer entries."""

        decode_buffer_entry = decode_buffer[node_idx]

        embeddings_predicted_by_children = torch.stack(
            decode_buffer_entry.embeddings_predicted_by_children
        )

        average_predicted_embeddings = torch.mean(
            embeddings_predicted_by_children,
            dim=0,
        )
        predicted_type_logits: Tensor = self.node_type_predictor(
            average_predicted_embeddings
        )

        decode_buffer_entry.classification_loss = F.cross_entropy(
            predicted_type_logits,
            torch.tensor(
                graph.node_types[node_idx],
                dtype=torch.long,
                device=predicted_type_logits.device,
            ),
        )
        decode_buffer_entry.child_predicted_embeddings_similarity_loss = torch.std(
            embeddings_predicted_by_children
        )

        node_parent_indices = graph.node_inputs_indices[node_idx]
        if len(node_parent_indices) == 0:
            return

        node_autoencoder = node_autoencoders[graph.node_types[node_idx]]
        predicted_parent_embeddings = node_autoencoder.decode(
            average_predicted_embeddings
        )

        for parent_idx, parent_embedding in zip(
            node_parent_indices, predicted_parent_embeddings
        ):
            parent_decode_buffer_entry = decode_buffer[parent_idx]
            parent_decode_buffer_entry.embeddings_predicted_by_children.append(
                parent_embedding
            )

    def inference_decode_blind_autoregressive(
        self,
        graph_embedding: Tensor,
    ):
        pass
