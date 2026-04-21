import math

import torch
from torch import Tensor
import torch.nn as nn
from typing import Iterable
from dataclasses import dataclass
from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_condenser_graph_description,
)
import torch.nn.functional as F


def _normalize_to_sqrt_n(x: Tensor) -> Tensor:
    """Rescale each D-dim vector along the last axis to have L2 norm sqrt(D)."""
    dim = x.shape[-1]
    return F.normalize(x, dim=-1) * math.sqrt(dim)


def _class_balance_weights(node_types: list[int]) -> list[float]:
    """Return per-node weights of ``1 / count_of_that_type``.

    Scaling each node's classification loss by this weight makes every type
    present in ``node_types`` contribute equally to the summed loss, regardless
    of how many nodes of that type appear in the graph.
    """
    counts: dict[int, int] = {}
    for t in node_types:
        counts[t] = counts.get(t, 0) + 1
    return [1.0 / counts[t] for t in node_types]


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
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers):
                x = self.activation(x)

        return x


def _mlp_dims(
    in_dim: int,
    out_dim: int,
    mlp_depth: int,
    mlp_expansion_factor: float,
) -> list[int]:
    """Build an MLP layer-size list with ``mlp_depth`` hidden layers between
    ``in_dim`` and ``out_dim``. Each hidden layer's width is
    ``round(in_dim * mlp_expansion_factor)`` (and at least 1)."""
    assert mlp_depth >= 0
    assert mlp_expansion_factor > 0
    hidden_width = max(1, round(in_dim * mlp_expansion_factor))
    return [in_dim] + [hidden_width] * mlp_depth + [out_dim]


class NodeEncoder(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        in_degree: int,
        mlp_depth: int,
        mlp_expansion_factor: float,
    ):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.in_degree = in_degree
        self.encoder = MLP(
            _mlp_dims(
                node_embedding_dim * in_degree,
                node_embedding_dim,
                mlp_depth,
                mlp_expansion_factor,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten()
        x = self.encoder(x)
        x = _normalize_to_sqrt_n(x)

        return x


class NodeDecoder(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        in_degree: int,
        mlp_depth: int,
        mlp_expansion_factor: float,
    ):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.in_degree = in_degree
        self.decoder = MLP(
            _mlp_dims(
                node_embedding_dim,
                node_embedding_dim * in_degree,
                mlp_depth,
                mlp_expansion_factor,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.decoder(x)
        x = x.view(self.in_degree, self.node_embedding_dim)
        x = _normalize_to_sqrt_n(x)
        x = torch.unbind(x, dim=0)

        return x


class NodeAutoEncoder(nn.Module):
    """Uniform node autoencoder interface.

    Both ``encode`` and ``decode`` receive the full ``node_type`` index from
    the graph. Subclasses are free to use or ignore it: the trunk / condenser
    autoencoder ignores it, while ``OutputNodeAutoEncoder`` uses it to look up
    the per-output-slot embedding.
    """

    def encode(self, input_node_embeddings: Tensor, node_type: int) -> Tensor:
        raise NotImplementedError

    def decode(self, node_embedding: Tensor, node_type: int) -> Tensor:
        raise NotImplementedError


class FixedInDegreeNodeAutoEncoder(NodeAutoEncoder):
    def __init__(
        self,
        node_embedding_dim: int,
        in_degree,
        mlp_depth: int,
        mlp_expansion_factor: float,
    ):
        super().__init__()

        self.node_embedding_dim = node_embedding_dim

        self.encoder = NodeEncoder(
            node_embedding_dim, in_degree, mlp_depth, mlp_expansion_factor
        )
        self.decoder = NodeDecoder(
            node_embedding_dim, in_degree, mlp_depth, mlp_expansion_factor
        )

    def encode(
        self,
        input_node_embeddings: Tensor,
        node_type: int,
    ) -> Tensor:
        del node_type
        return self.encoder(input_node_embeddings)

    def decode(self, node_embedding: Tensor, node_type: int) -> Tensor:
        del node_type
        return self.decoder(node_embedding)


class OutputNodeAutoEncoder(NodeAutoEncoder):
    def __init__(
        self,
        node_embedding_dim: int,
        num_output_nodes: int,
        output_node_types_start: int,
        mlp_depth: int,
        mlp_expansion_factor: float,
    ):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.output_node_types_start = output_node_types_start
        self.encoder = NodeEncoder(
            node_embedding_dim,
            in_degree=2,
            mlp_depth=mlp_depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )
        self.decoder = NodeDecoder(
            node_embedding_dim,
            in_degree=1,
            mlp_depth=mlp_depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )
        self.output_node_embeddings = nn.Embedding(num_output_nodes, node_embedding_dim)

    def encode(self, input_node_embeddings: Tensor, node_type: int) -> Tensor:
        output_slot_idx = node_type - self.output_node_types_start
        # `input_node_embeddings` is [1, D] (an output node has exactly one
        # graph-parent). Slice instead of index-scalar so the slot embedding
        # keeps the same rank ([1, D]) and the concat gives [2, D].
        output_slot_embedding = self.output_node_embeddings.weight[
            output_slot_idx : output_slot_idx + 1
        ]
        x = torch.cat([input_node_embeddings, output_slot_embedding], dim=0)

        return self.encoder(x)

    def decode(self, node_embedding: Tensor, node_type: int) -> Tensor:
        del node_type
        return self.decoder(node_embedding)


@dataclass
class TrainingDecodeBufferEntry:
    embeddings_predicted_by_children: list[Tensor]
    classification_loss: Tensor | None
    child_predicted_embeddings_similarity_loss: Tensor | None
    predicted_type_logits: Tensor | None


@dataclass
class TrainingStepLossReturnType:
    condenser_node_classification_losses: list[Tensor]
    condenser_node_predicted_embeddings_similarity_losses: list[Tensor]
    primary_node_classification_losses: list[Tensor]
    primary_node_predicted_embeddings_similarity_losses: list[Tensor]
    # Raw per-node logits and true type labels for downstream diagnostics
    # (e.g. per-node-type accuracy per class).
    primary_node_predicted_type_logits: list[Tensor]
    primary_node_true_types: list[int]


class DagnabbitAutoEncoder(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        trunk_node_type_in_degrees: int | list[int],
        num_trunk_node_types: int,
        condenser_node_type_in_degree: int,
        num_root_nodes: int,
        num_output_nodes: int,
        mlp_depth: int,
        mlp_expansion_factor: float,
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
        self.num_node_types = num_trunk_node_types + num_root_nodes + num_output_nodes
        self.mlp_depth = mlp_depth
        self.mlp_expansion_factor = mlp_expansion_factor

        # ---- Trunk node autoencoders (one per node type) ----
        self.trunk_node_autoencoders = nn.ModuleList(
            [
                FixedInDegreeNodeAutoEncoder(
                    node_embedding_dim,
                    in_degree,
                    mlp_depth=mlp_depth,
                    mlp_expansion_factor=mlp_expansion_factor,
                )
                for in_degree in self.trunk_node_in_degrees
            ]
        )

        # ---- Output autoencoder (shared across all output slots) ----
        # Each output node has exactly one graph-parent; OutputNodeAutoEncoder
        # concatenates that parent embedding with a learnable per-output-slot
        # embedding (looked up from the node_type it is called with), so its
        # internal encoder has in_degree=2.
        output_node_types_start = num_trunk_node_types + num_root_nodes
        self.output_autoencoder = OutputNodeAutoEncoder(
            node_embedding_dim,
            num_output_nodes=num_output_nodes,
            output_node_types_start=output_node_types_start,
            mlp_depth=mlp_depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )

        # ---- Condenser node autoencoder (shared across all condenser nodes) ----
        self.condenser_node_autoencoder = FixedInDegreeNodeAutoEncoder(
            node_embedding_dim,
            in_degree=condenser_node_type_in_degree,
            mlp_depth=mlp_depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )

        # ---- Root node embeddings ----
        self.root_node_embeddings = nn.Embedding(
            self.num_root_nodes, self.node_embedding_dim
        )

        # ---- Unified node-autoencoder lookup by node type ----
        # Trunk types live at [0, num_trunk_node_types); output types live at
        # [output_node_types_start, num_node_types). All output types share the
        # same OutputNodeAutoEncoder instance, which reads the node_type passed
        # to .encode / .decode to pick the right per-slot embedding.
        self.node_autoencoders: dict[int, NodeAutoEncoder] = {}
        for node_type_idx, ae in enumerate(self.trunk_node_autoencoders):
            self.node_autoencoders[node_type_idx] = ae
        for output_slot_idx in range(num_output_nodes):
            type_idx = output_node_types_start + output_slot_idx
            self.node_autoencoders[type_idx] = self.output_autoencoder

        # ---- Auxiliary node-type prediction head ----
        self.node_type_predictor = MLP(
            _mlp_dims(
                in_dim=self.node_embedding_dim,
                out_dim=self.num_node_types,
                mlp_depth=mlp_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )
        )

    @staticmethod
    def evaluate_graph(
        graph: FixedInDegreeDAGDescription,
        root_node_embeddings: Tensor,
        node_autoencoders: dict[int, NodeAutoEncoder],
        node_embedding_dim: int,
    ) -> Tensor:
        assert root_node_embeddings.shape[0] == graph.num_root_nodes

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

            embeddings_buffer[node_idx] = node_autoencoder.encode(
                parent_embeddings, node_type
            )

        return embeddings_buffer

    def encode_graph_with_condenser(
        self,
        primary_graph: FixedInDegreeDAGDescription,
        return_buffers: bool = False,
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
        # The condenser reduces to exactly one leaf by construction (see
        # `make_condenser_graph_description`); index with a scalar so the
        # graph embedding is shape [D] to match the per-node entries the
        # decode buffer accumulates elsewhere.
        assert len(condenser_graph.leaf_node_indices) == 1
        graph_embedding = condenser_buffer[condenser_graph.leaf_node_indices[0]]
        if return_buffers:
            return condenser_graph, graph_embedding, primary_buffer, condenser_buffer
        else:
            return condenser_graph, graph_embedding

    def training_forward(
        self,
        primary_graph: FixedInDegreeDAGDescription,
    ) -> TrainingStepLossReturnType:
        """Returns losses for the training step."""

        # Encode
        condenser_graph, graph_embedding, primary_buffer, condenser_buffer = (
            self.encode_graph_with_condenser(
                primary_graph,
                return_buffers=True,
            )
        )

        # Guided Autoregressive Decode

        condenser_decode_buffer = [
            TrainingDecodeBufferEntry(
                embeddings_predicted_by_children=[],
                child_predicted_embeddings_similarity_loss=None,
                classification_loss=None,
                predicted_type_logits=None,
            )
            for _ in range(condenser_graph.num_nodes)
        ]

        condenser_decode_buffer[-1] = TrainingDecodeBufferEntry(
            embeddings_predicted_by_children=[graph_embedding],
            child_predicted_embeddings_similarity_loss=torch.std(
                graph_embedding, dim=0
            ),
            classification_loss=None,
            predicted_type_logits=None,
        )

        condenser_classification_loss_weights = _class_balance_weights(
            list(condenser_graph.node_types)
        )
        primary_classification_loss_weights = _class_balance_weights(
            list(primary_graph.node_types)
        )

        for node_idx in reversed(
            range(condenser_graph.num_root_nodes, condenser_graph.num_nodes)
        ):
            self._decode_step(
                node_idx=node_idx,
                graph=condenser_graph,
                decode_buffer=condenser_decode_buffer,
                node_autoencoders={0: self.condenser_node_autoencoder},
                classification_loss_weight=condenser_classification_loss_weights[
                    node_idx
                ],
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
                predicted_type_logits=None,
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
                classification_loss_weight=primary_classification_loss_weights[
                    node_idx
                ],
            )

        condenser_trunk_entries = condenser_decode_buffer[
            condenser_graph.num_root_nodes :
        ]

        condenser_trunk_true_types = condenser_graph.node_types[
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
            primary_node_predicted_type_logits=[
                e.predicted_type_logits for e in primary_decode_buffer
            ],
            primary_node_true_types=list(primary_graph.node_types),
        )

    def _decode_step(
        self,
        node_idx: int,
        graph: FixedInDegreeDAGDescription,
        decode_buffer: list[TrainingDecodeBufferEntry],
        node_autoencoders: dict[int, NodeAutoEncoder],
        classification_loss_weight: float = 1.0,
    ) -> None:
        """Populate `decode_buffer[node_idx]`'s losses and push this node's
        predicted parent embeddings into its parents' buffer entries.

        ``classification_loss_weight`` multiplies this node's classification
        loss; callers use it for class-balanced weighting across node types.
        """

        decode_buffer_entry = decode_buffer[node_idx]

        embeddings_predicted_by_children = torch.stack(
            decode_buffer_entry.embeddings_predicted_by_children
        )

        average_predicted_embeddings = torch.mean(
            embeddings_predicted_by_children,
            dim=0,
        )
        average_predicted_embeddings = _normalize_to_sqrt_n(average_predicted_embeddings)

        predicted_type_logits: Tensor = self.node_type_predictor(
            average_predicted_embeddings
        )
        decode_buffer_entry.predicted_type_logits = predicted_type_logits

        classification_loss = F.cross_entropy(
            predicted_type_logits,
            torch.tensor(
                graph.node_types[node_idx],
                dtype=torch.long,
                device=predicted_type_logits.device,
            ),
        )
        decode_buffer_entry.classification_loss = (
            classification_loss * classification_loss_weight
        )
        decode_buffer_entry.child_predicted_embeddings_similarity_loss = torch.std(
            embeddings_predicted_by_children
        )

        node_parent_indices = graph.node_inputs_indices[node_idx]
        if len(node_parent_indices) == 0:
            return

        node_type = graph.node_types[node_idx]
        node_autoencoder = node_autoencoders[node_type]
        predicted_parent_embeddings = node_autoencoder.decode(
            average_predicted_embeddings, node_type
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
