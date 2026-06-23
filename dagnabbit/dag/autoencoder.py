import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    NodeSupertype,
    PreparedRankBatch,
    subtype_to_supertype,
)


def _feed_forward_layers(
    vector_dims: Iterable[int],
    dropout: float,
) -> list[nn.Module]:
    vector_dims = list(vector_dims)
    layers: list[nn.Module] = []
    for i in range(len(vector_dims) - 1):
        layers.append(nn.Linear(vector_dims[i], vector_dims[i + 1]))
        if i + 1 < len(vector_dims) - 1:
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return layers


def _normalize_to_unit_sphere(x: Tensor) -> Tensor:
    """Project each embedding vector onto the unit L2 hypersphere."""
    return F.normalize(x, dim=-1)


def _normalize_to_model_sphere(x: Tensor) -> Tensor:
    """Project embeddings to the fixed radius used at recursive boundaries."""
    return _normalize_to_unit_sphere(x) * math.sqrt(x.shape[-1])


class TransformerBlock(nn.Module):
    """nGPT-style attention and MLP updates on the unit hypersphere."""

    def __init__(
        self,
        node_embedding_dim: int,
        num_heads: int,
        transformer_mlp_depth: int,
        mlp_expansion_factor: float,
        dropout: float,
        residual_step_init: float,
    ):
        super().__init__()
        if transformer_mlp_depth < 0:
            raise ValueError("transformer_mlp_depth must be non-negative")
        if not 0.0 < residual_step_init < 1.0:
            raise ValueError("residual_step_init must be strictly between 0 and 1")
        hidden_dim = max(1, round(node_embedding_dim * mlp_expansion_factor))
        self.attn = nn.MultiheadAttention(
            node_embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            *_feed_forward_layers(
                [node_embedding_dim]
                + [hidden_dim] * transformer_mlp_depth
                + [node_embedding_dim],
                dropout,
            )
        )
        self.ff_dropout = nn.Dropout(dropout)

        # Sigmoid-parameterized, per-channel interpolation steps. Initializing
        # them near zero makes each block begin as a small directional update
        # instead of an unrestricted residual addition.
        step_logit = math.log(residual_step_init) - math.log1p(-residual_step_init)
        self.attn_step_logits = nn.Parameter(
            torch.full((node_embedding_dim,), step_logit)
        )
        self.ff_step_logits = nn.Parameter(
            torch.full((node_embedding_dim,), step_logit)
        )

    @staticmethod
    def _interpolate_on_sphere(
        current: Tensor,
        proposal: Tensor,
        step_logits: Tensor,
    ) -> Tensor:
        step = step_logits.sigmoid().to(dtype=current.dtype)
        return _normalize_to_unit_sphere(
            current + step * (proposal - current)
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        x = _normalize_to_unit_sphere(x)
        y = x
        y, _ = self.attn(
            y,
            y,
            y,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        y = _normalize_to_unit_sphere(self.attn_dropout(y))
        x = self._interpolate_on_sphere(x, y, self.attn_step_logits)

        y = _normalize_to_unit_sphere(self.ff_dropout(self.ff(x)))
        return self._interpolate_on_sphere(x, y, self.ff_step_logits)


class TypeConditionedSequenceTransformer(nn.Module):
    """Shared sequence processor for node parent slots.

    The public context length is ``max_context_length``. The module appends
    learnable register tokens and one node-type token internally, then returns
    only the transformed public context slots.
    """

    def __init__(
        self,
        node_embedding_dim: int,
        num_node_types: int,
        max_context_length: int,
        num_layers: int,
        num_register_tokens: int,
        num_heads: int,
        transformer_mlp_depth: int,
        mlp_expansion_factor: float,
        dropout: float = 0.0,
        residual_step_init: float = 0.05,
    ):
        super().__init__()
        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_register_tokens < 0:
            raise ValueError("num_register_tokens must be non-negative")
        if transformer_mlp_depth < 0:
            raise ValueError("transformer_mlp_depth must be non-negative")
        if node_embedding_dim % num_heads != 0:
            raise ValueError("node_embedding_dim must be divisible by num_heads")

        self.node_embedding_dim = node_embedding_dim
        self.max_context_length = max_context_length
        self.num_register_tokens = num_register_tokens
        self.transformer_mlp_depth = transformer_mlp_depth
        self.position_embeddings = nn.Parameter(
            torch.empty(max_context_length, node_embedding_dim)
        )
        self.register_tokens = nn.Parameter(
            torch.empty(num_register_tokens, node_embedding_dim)
        )
        self.node_type_embeddings = nn.Embedding(num_node_types, node_embedding_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    node_embedding_dim=node_embedding_dim,
                    num_heads=num_heads,
                    transformer_mlp_depth=transformer_mlp_depth,
                    mlp_expansion_factor=mlp_expansion_factor,
                    dropout=dropout,
                    residual_step_init=residual_step_init,
                )
                for _ in range(num_layers)
            ]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position_embeddings, std=0.02)
        nn.init.normal_(self.register_tokens, std=0.02)

    def forward(
        self,
        x: Tensor,
        node_types: Tensor,
        valid_context_mask: Tensor | None = None,
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, K, D]")
        if x.shape[1] != self.max_context_length:
            raise ValueError(
                f"x context length must be {self.max_context_length}; got {x.shape[1]}"
            )
        if x.shape[2] != self.node_embedding_dim:
            raise ValueError(
                f"x embedding dim must be {self.node_embedding_dim}; got {x.shape[2]}"
            )
        if node_types.shape != (x.shape[0],):
            raise ValueError("node_types must have shape [B]")

        batch_size = x.shape[0]
        x = x + self.position_embeddings.unsqueeze(0).to(dtype=x.dtype, device=x.device)

        register_tokens = self.register_tokens.to(dtype=x.dtype, device=x.device)
        registers = register_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        type_token = (
            self.node_type_embeddings(node_types).to(dtype=x.dtype).unsqueeze(1)
        )
        x = torch.cat([x, registers, type_token], dim=1)

        key_padding_mask = None
        if valid_context_mask is not None:
            if valid_context_mask.shape != (batch_size, self.max_context_length):
                raise ValueError("valid_context_mask must have shape [B, K]")
            valid_extra = torch.ones(
                batch_size,
                self.num_register_tokens + 1,
                dtype=torch.bool,
                device=valid_context_mask.device,
            )
            valid_tokens = torch.cat([valid_context_mask, valid_extra], dim=1)
            key_padding_mask = ~valid_tokens

        for block in self.blocks:
            x = block(x, key_padding_mask)

        # Recursive encoder/decoder calls exchange vectors at the conventional
        # Transformer radius sqrt(D), but without learned affine norm terms.
        return _normalize_to_model_sphere(x[:, : self.max_context_length])


class TransformerNodeEncoder(nn.Module):
    def __init__(self, sequence_transformer: TypeConditionedSequenceTransformer):
        super().__init__()
        self.sequence_transformer = sequence_transformer

    def forward_batch(
        self,
        parent_embeddings: Tensor,
        subtypes: Tensor,
        valid_parent_mask: Tensor,
    ) -> Tensor:
        transformed = self.sequence_transformer(
            parent_embeddings,
            subtypes,
            valid_parent_mask,
        )
        weights = valid_parent_mask.to(dtype=transformed.dtype).unsqueeze(-1)
        counts = weights.sum(dim=1).clamp(min=1.0)
        pooled = (transformed * weights).sum(dim=1) / counts.sqrt()
        return _normalize_to_model_sphere(pooled)


class TransformerNodeDecoder(nn.Module):
    def __init__(
        self,
        sequence_transformer: TypeConditionedSequenceTransformer,
        num_parent_slots: int,
        num_leaf_context_slots: int,
    ):
        super().__init__()
        self.sequence_transformer = sequence_transformer
        self.num_parent_slots = num_parent_slots
        self.num_leaf_context_slots = num_leaf_context_slots

        if num_parent_slots <= 0:
            raise ValueError("num_parent_slots must be positive")
        if num_leaf_context_slots < 0:
            raise ValueError("num_leaf_context_slots must be non-negative")

        expected_context_length = num_parent_slots + num_leaf_context_slots
        if sequence_transformer.max_context_length != expected_context_length:
            raise ValueError(
                "decoder transformer context length must equal parent slots plus "
                "leaf context slots; got "
                f"{sequence_transformer.max_context_length} vs "
                f"{expected_context_length}"
            )

    def forward_batch(
        self,
        node_embeddings: Tensor,
        subtypes: Tensor,
        leaf_embeddings: Tensor,
    ) -> Tensor:
        if node_embeddings.ndim != 2:
            raise ValueError("node_embeddings must have shape [B, D]")
        batch_size, embedding_dim = node_embeddings.shape
        if embedding_dim != self.sequence_transformer.node_embedding_dim:
            raise ValueError(
                "node_embeddings embedding dim must be "
                f"{self.sequence_transformer.node_embedding_dim}; got {embedding_dim}"
            )

        local_context = node_embeddings.unsqueeze(1).expand(
            -1,
            self.num_parent_slots,
            -1,
        )
        leaf_context = self._prepare_leaf_context(leaf_embeddings, node_embeddings)
        context = torch.cat([local_context, leaf_context], dim=1)
        transformed = self.sequence_transformer(context, subtypes)
        return transformed[:, : self.num_parent_slots]

    def _prepare_leaf_context(
        self,
        leaf_embeddings: Tensor,
        node_embeddings: Tensor,
    ) -> Tensor:
        return leaf_embeddings.to(
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        )


@dataclass
class TrainingStepLossReturnType:
    # Node-aligned tensor over the primary graph batch: [B, N].
    primary_node_classification_losses: Tensor
    # Raw per-node logits and true type labels for downstream diagnostics.
    # ``primary_node_predicted_type_logits[b, i]`` corresponds to
    # ``primary_node_true_types[b, i]``.
    primary_node_predicted_type_logits: Tensor
    primary_node_true_types: Tensor
    # Teacher-forced counterparts (same node alignment, same true labels). These
    # come from a second decode pass that feeds each node its true encode
    # embedding instead of its own prediction; see
    # ``DagnabbitAutoEncoder._decode_pipeline``.
    teacher_forced_primary_node_classification_losses: Tensor
    teacher_forced_primary_node_predicted_type_logits: Tensor
    # Per-edge reconstruction: 1 - cos(predicted_parent, encoder_buffer[parent]).
    # Edge-flat 1-D tensor (empty when no edges exist in the graph).
    primary_node_parent_reconstruction_losses: Tensor
    teacher_forced_primary_node_parent_reconstruction_losses: Tensor
    # Per-node consistency: variance of predictions landing on the same parent,
    # averaged over D. Node-indexed [B, N] tensor; zero for count <= 1 nodes.
    primary_node_parent_consistency_losses: Tensor
    teacher_forced_primary_node_parent_consistency_losses: Tensor


@dataclass
class _DecodePipelineResult:
    """Dense per-node outputs of a single decode pass over the primary graph."""

    primary_combined: Tensor
    primary_logits: Tensor
    primary_class_losses: Tensor
    # Edge-flat 1-D tensor of per-edge reconstruction losses.
    primary_recon: Tensor
    # Node-indexed 1-D tensor of per-parent consistency losses (zero for count<=1).
    primary_consistency: Tensor


@dataclass
class _BatchedRank:
    batch_indices: Tensor
    node_indices: Tensor
    parent_indices: Tensor
    valid_parent_mask: Tensor
    subtypes: Tensor
    has_valid_parents: bool = False


@dataclass
class _DecodeNode:
    id: int
    embedding: Tensor
    type_id: int | None
    is_root: bool
    is_output: bool
    parents: list[int | None]
    expanded: bool = False


@dataclass
class _ParentPrediction:
    child_id: int
    slot: int
    embedding: Tensor


class DagnabbitAutoEncoder(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        trunk_node_type_in_degrees: int | list[int],
        num_trunk_node_types: int,
        num_root_nodes: int,
        num_output_nodes: int,
        mlp_expansion_factor: float,
        reconstruction_detach_target: bool = True,
        compute_reconstruction_loss: bool = False,
        transformer_num_layers: int = 1,
        transformer_mlp_depth: int = 1,
        transformer_num_register_tokens: int = 2,
        transformer_num_heads: int = 4,
        transformer_dropout: float = 0.0,
        transformer_residual_step_init: float = 0.05,
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
        self.num_node_types = num_trunk_node_types + num_root_nodes + num_output_nodes
        self.mlp_expansion_factor = mlp_expansion_factor
        self.reconstruction_detach_target = reconstruction_detach_target
        self.compute_reconstruction_loss = compute_reconstruction_loss
        self.maximum_indegree = max([1, *self.trunk_node_in_degrees])
        self.transformer_num_layers = transformer_num_layers
        self.transformer_mlp_depth = transformer_mlp_depth
        self.transformer_num_register_tokens = transformer_num_register_tokens
        self.transformer_num_heads = transformer_num_heads
        self.transformer_dropout = transformer_dropout
        self.transformer_residual_step_init = transformer_residual_step_init

        # ---- Shared, type-conditioned node transformers ----
        # The encoder public context is the ordered parent-slot sequence padded
        # to maximum_indegree. The decoder public context starts with those same
        # parent-output slots, then appends one auxiliary slot per graph leaf
        # (the fixed output nodes in generated graphs). Each transformer appends
        # its own registers and one node-type token internally, so all trunk and
        # output types share weights.
        self.node_encoder = TransformerNodeEncoder(
            TypeConditionedSequenceTransformer(
                node_embedding_dim=node_embedding_dim,
                num_node_types=self.num_node_types,
                max_context_length=self.maximum_indegree,
                num_layers=self.transformer_num_layers,
                num_register_tokens=transformer_num_register_tokens,
                num_heads=transformer_num_heads,
                transformer_mlp_depth=transformer_mlp_depth,
                mlp_expansion_factor=mlp_expansion_factor,
                dropout=transformer_dropout,
                residual_step_init=transformer_residual_step_init,
            )
        )
        self.node_decoder = TransformerNodeDecoder(
            TypeConditionedSequenceTransformer(
                node_embedding_dim=node_embedding_dim,
                num_node_types=self.num_node_types,
                max_context_length=self.maximum_indegree + self.num_output_nodes,
                num_layers=self.transformer_num_layers,
                num_register_tokens=transformer_num_register_tokens,
                num_heads=transformer_num_heads,
                transformer_mlp_depth=transformer_mlp_depth,
                mlp_expansion_factor=mlp_expansion_factor,
                dropout=transformer_dropout,
                residual_step_init=transformer_residual_step_init,
            ),
            num_parent_slots=self.maximum_indegree,
            num_leaf_context_slots=self.num_output_nodes,
        )

        # ---- Root node embeddings ----
        self.root_node_embeddings = nn.Embedding(
            self.num_root_nodes, self.node_embedding_dim
        )

        # ---- Auxiliary node-type prediction head ----
        self.node_type_predictor = nn.Linear(
            self.node_embedding_dim,
            self.num_node_types,
        )

    def evaluate_graph_batch(
        self,
        graphs: Sequence[FixedInDegreeDAGDescription],
        root_node_embeddings: Tensor | None = None,
        rank_batches: Sequence[_BatchedRank] | None = None,
    ) -> Tensor:
        graphs = list(graphs)
        if root_node_embeddings is None:
            root_node_embeddings = self.root_node_embeddings.weight.unsqueeze(0).expand(
                len(graphs), -1, -1
            )

        batch_size = len(graphs)
        num_nodes = graphs[0].num_nodes
        device = root_node_embeddings.device
        embeddings_buffer = torch.empty(
            (batch_size, num_nodes, self.node_embedding_dim),
            dtype=root_node_embeddings.dtype,
            device=device,
        )
        embeddings_buffer[:, : self.num_root_nodes] = root_node_embeddings

        # Walk topological ranks ascending. Rank 0 roots are seeded above. Every
        # non-root node in a rank can be encoded together because node type is an
        # input token to the shared transformer rather than a module selector.
        if rank_batches is None:
            rank_batches = self._make_batched_rank_cache(graphs, device)
        max_ranks = len(rank_batches)
        for rank in range(1, max_ranks):
            rank_batch = rank_batches[rank]
            if rank_batch.node_indices.numel() == 0:
                continue
            parent_embeddings = embeddings_buffer[
                rank_batch.batch_indices[:, None],
                rank_batch.parent_indices,
            ]
            parent_embeddings = parent_embeddings.masked_fill(
                ~rank_batch.valid_parent_mask.unsqueeze(-1),
                0.0,
            )
            embeddings_buffer[
                rank_batch.batch_indices,
                rank_batch.node_indices,
            ] = self.node_encoder.forward_batch(
                parent_embeddings,
                rank_batch.subtypes,
                rank_batch.valid_parent_mask,
            ).to(embeddings_buffer.dtype)

        return embeddings_buffer

    def _empty_batched_rank(self, device: torch.device) -> _BatchedRank:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return _BatchedRank(
            batch_indices=empty,
            node_indices=empty,
            parent_indices=torch.empty(
                0,
                self.maximum_indegree,
                dtype=torch.long,
                device=device,
            ),
            valid_parent_mask=torch.empty(
                0,
                self.maximum_indegree,
                dtype=torch.bool,
                device=device,
            ),
            subtypes=empty,
        )

    def _validate_prepared_rank(self, rank_batch: PreparedRankBatch) -> None:
        parent_width = rank_batch.parent_indices.shape[1]
        if parent_width != self.maximum_indegree:
            raise ValueError(
                "graph rank metadata has maximum in-degree "
                f"{parent_width}, but model expects {self.maximum_indegree}"
            )

    def _make_batched_rank_cache(
        self,
        graphs: Sequence[FixedInDegreeDAGDescription],
        device: torch.device,
    ) -> list[_BatchedRank]:
        max_ranks = max(len(graph.rank_batches) for graph in graphs)
        cpu_ranks = [
            self._make_batched_rank_cpu(graphs, rank) for rank in range(max_ranks)
        ]
        row_counts = [rank.node_indices.shape[0] for rank in cpu_ranks]
        total_rows = sum(row_counts)
        if total_rows == 0:
            return [self._empty_batched_rank(device) for _ in cpu_ranks]

        nonempty_ranks = [
            rank for rank, row_count in zip(cpu_ranks, row_counts) if row_count
        ]
        batch_indices = torch.cat([rank.batch_indices for rank in nonempty_ranks]).to(
            device=device,
            non_blocking=True,
        )
        node_indices = torch.cat([rank.node_indices for rank in nonempty_ranks]).to(
            device=device,
            non_blocking=True,
        )
        parent_indices = torch.cat([rank.parent_indices for rank in nonempty_ranks]).to(
            device=device,
            non_blocking=True,
        )
        valid_parent_mask = torch.cat(
            [rank.valid_parent_mask for rank in nonempty_ranks]
        ).to(
            device=device,
            non_blocking=True,
        )
        subtypes = torch.cat([rank.subtypes for rank in nonempty_ranks]).to(
            device=device,
            non_blocking=True,
        )

        device_ranks: list[_BatchedRank] = []
        offset = 0
        for cpu_rank, row_count in zip(cpu_ranks, row_counts):
            start = offset
            offset += row_count
            end = offset
            device_ranks.append(
                _BatchedRank(
                    batch_indices=batch_indices[start:end],
                    node_indices=node_indices[start:end],
                    parent_indices=parent_indices[start:end],
                    valid_parent_mask=valid_parent_mask[start:end],
                    subtypes=subtypes[start:end],
                    has_valid_parents=cpu_rank.has_valid_parents,
                )
            )
        return device_ranks

    def _make_batched_rank_cpu(
        self,
        graphs: Sequence[FixedInDegreeDAGDescription],
        rank: int,
    ) -> _BatchedRank:
        batch_indices: list[Tensor] = []
        node_indices: list[Tensor] = []
        parent_indices: list[Tensor] = []
        valid_parent_masks: list[Tensor] = []
        subtypes: list[Tensor] = []
        has_valid_parents = False

        for batch_idx, graph in enumerate(graphs):
            if rank >= len(graph.rank_batches):
                continue
            rank_batch = graph.rank_batches[rank]
            self._validate_prepared_rank(rank_batch)
            num_rows = rank_batch.node_indices.shape[0]
            if num_rows == 0:
                continue
            has_valid_parents = has_valid_parents or rank_batch.has_valid_parents
            batch_indices.append(
                torch.full(
                    (num_rows,),
                    batch_idx,
                    dtype=torch.long,
                )
            )
            node_indices.append(rank_batch.node_indices)
            parent_indices.append(rank_batch.parent_indices)
            valid_parent_masks.append(rank_batch.valid_parent_mask)
            subtypes.append(rank_batch.subtypes)

        if not node_indices:
            return self._empty_batched_rank(torch.device("cpu"))

        return _BatchedRank(
            batch_indices=torch.cat(batch_indices),
            node_indices=torch.cat(node_indices),
            parent_indices=torch.cat(parent_indices),
            valid_parent_mask=torch.cat(valid_parent_masks),
            subtypes=torch.cat(subtypes),
            has_valid_parents=has_valid_parents,
        )

    def _in_degree_for_type(self, node_type: int) -> int:
        if node_type < self.num_trunk_node_types:
            return self.trunk_node_in_degrees[node_type]
        output_start = self.num_trunk_node_types + self.num_root_nodes
        if node_type < output_start:
            return 0
        if node_type < self.num_node_types:
            return 1
        raise ValueError(f"unknown node type {node_type}")

    def training_forward_batch(
        self,
        primary_graphs: Sequence[FixedInDegreeDAGDescription],
        return_buffers: bool = False,
    ) -> (
        TrainingStepLossReturnType
        | tuple[
            TrainingStepLossReturnType,
            Tensor,
            Tensor,
        ]
    ):
        """Batched training forward over multiple structurally compatible DAGs.

        Per-node outputs keep an explicit ``[B, N, ...]`` graph-batch dimension.
        Reconstruction losses remain edge-flat across the whole graph batch.
        """

        primary_graphs = list(primary_graphs)
        device = self.root_node_embeddings.weight.device
        rank_batches = self._make_batched_rank_cache(primary_graphs, device)
        primary_buffer = self.evaluate_graph_batch(
            graphs=primary_graphs,
            rank_batches=rank_batches,
        )

        device = primary_buffer.device
        primary_labels = torch.stack(
            [graph.node_types_tensor for graph in primary_graphs],
            dim=0,
        ).to(device=device, non_blocking=True)

        # Two decode passes over the same shared encode pass. The
        # autoregressive pass compounds predictions down each DAG; the
        # teacher-forced pass decodes each node from its true encode-side
        # embedding. Both passes are processed together rank-by-rank.
        autoregressive, teacher_forced = self._decode_pipeline(
            primary_graphs=primary_graphs,
            primary_buffer=primary_buffer,
            primary_labels=primary_labels,
            rank_batches=rank_batches,
            device=device,
        )

        losses = TrainingStepLossReturnType(
            primary_node_classification_losses=autoregressive.primary_class_losses,
            primary_node_predicted_type_logits=autoregressive.primary_logits,
            primary_node_true_types=primary_labels,
            teacher_forced_primary_node_classification_losses=(
                teacher_forced.primary_class_losses
            ),
            teacher_forced_primary_node_predicted_type_logits=(
                teacher_forced.primary_logits
            ),
            primary_node_parent_reconstruction_losses=autoregressive.primary_recon,
            teacher_forced_primary_node_parent_reconstruction_losses=(
                teacher_forced.primary_recon
            ),
            primary_node_parent_consistency_losses=autoregressive.primary_consistency,
            teacher_forced_primary_node_parent_consistency_losses=(
                teacher_forced.primary_consistency
            ),
        )

        if return_buffers:
            return losses, primary_buffer, autoregressive.primary_combined
        return losses

    def _decode_pipeline(
        self,
        primary_graphs: Sequence[FixedInDegreeDAGDescription],
        primary_buffer: Tensor,
        primary_labels: Tensor,
        rank_batches: Sequence[_BatchedRank],
        device: torch.device,
    ) -> tuple["_DecodePipelineResult", "_DecodePipelineResult"]:
        batch_size, num_nodes, _ = primary_buffer.shape

        def _zeros(dtype: torch.dtype):
            child_sum = torch.zeros(
                batch_size,
                num_nodes,
                self.node_embedding_dim,
                dtype=dtype,
                device=device,
            )
            child_count = torch.zeros(
                batch_size,
                num_nodes,
                dtype=dtype,
                device=device,
            )
            child_sumsq = torch.zeros(
                batch_size,
                num_nodes,
                self.node_embedding_dim,
                dtype=dtype,
                device=device,
            )
            return child_sum, child_count, child_sumsq

        ar_primary_sum, ar_primary_count, ar_primary_sumsq = _zeros(
            primary_buffer.dtype
        )
        tf_primary_sum, tf_primary_count, tf_primary_sumsq = _zeros(
            primary_buffer.dtype
        )

        leaf_indices = torch.stack(
            [graph.leaf_node_indices_tensor for graph in primary_graphs],
            dim=0,
        ).to(device=device, non_blocking=True)
        batch_rows = torch.arange(batch_size, dtype=torch.long, device=device)[:, None]
        leaf_embeddings_by_graph = primary_buffer[batch_rows, leaf_indices]
        for child_sum, child_count, child_sumsq in (
            (ar_primary_sum, ar_primary_count, ar_primary_sumsq),
            (tf_primary_sum, tf_primary_count, tf_primary_sumsq),
        ):
            leaf_embeddings = leaf_embeddings_by_graph.to(child_sum.dtype)
            child_sum[batch_rows, leaf_indices] = leaf_embeddings
            child_count[batch_rows, leaf_indices] = 1.0
            child_sumsq[batch_rows, leaf_indices] = leaf_embeddings**2

        (
            (
                ar_primary_combined,
                ar_primary_logits,
                ar_primary_class,
                ar_primary_recon,
                ar_primary_consistency,
            ),
            (
                tf_primary_combined,
                tf_primary_logits,
                tf_primary_class,
                tf_primary_recon,
                tf_primary_consistency,
            ),
        ) = self._decode_graph(
            graphs=primary_graphs,
            encoder_buffer=primary_buffer,
            autoregressive_child_sum=ar_primary_sum,
            autoregressive_child_count=ar_primary_count,
            autoregressive_child_sumsq=ar_primary_sumsq,
            teacher_forced_child_sum=tf_primary_sum,
            teacher_forced_child_count=tf_primary_count,
            teacher_forced_child_sumsq=tf_primary_sumsq,
            labels=primary_labels,
            leaf_embeddings_by_graph=leaf_embeddings_by_graph,
            process_roots=True,
            rank_batches=rank_batches,
            device=device,
        )

        autoregressive = _DecodePipelineResult(
            primary_combined=ar_primary_combined,
            primary_logits=ar_primary_logits,
            primary_class_losses=ar_primary_class,
            primary_recon=ar_primary_recon,
            primary_consistency=ar_primary_consistency,
        )
        teacher_forced = _DecodePipelineResult(
            primary_combined=tf_primary_combined,
            primary_logits=tf_primary_logits,
            primary_class_losses=tf_primary_class,
            primary_recon=tf_primary_recon,
            primary_consistency=tf_primary_consistency,
        )
        return autoregressive, teacher_forced

    def _decode_graph(
        self,
        graphs: Sequence[FixedInDegreeDAGDescription],
        encoder_buffer: Tensor,
        autoregressive_child_sum: Tensor,
        autoregressive_child_count: Tensor,
        autoregressive_child_sumsq: Tensor,
        teacher_forced_child_sum: Tensor,
        teacher_forced_child_count: Tensor,
        teacher_forced_child_sumsq: Tensor,
        labels: Tensor,
        leaf_embeddings_by_graph: Tensor,
        process_roots: bool,
        rank_batches: Sequence[_BatchedRank],
        device: torch.device,
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ]:
        batch_size, num_nodes, _ = encoder_buffer.shape

        def _alloc_pass_buffers():
            combined = torch.zeros(
                batch_size,
                num_nodes,
                self.node_embedding_dim,
                dtype=autoregressive_child_sum.dtype,
                device=device,
            )
            logits = torch.zeros(
                batch_size,
                num_nodes,
                self.num_node_types,
                dtype=encoder_buffer.dtype,
                device=device,
            )
            class_losses = torch.zeros(
                batch_size,
                num_nodes,
                dtype=encoder_buffer.dtype,
                device=device,
            )
            consistency = torch.zeros(
                batch_size,
                num_nodes,
                dtype=encoder_buffer.dtype,
                device=device,
            )
            return combined, logits, class_losses, consistency

        ar_combined, ar_logits, ar_class, ar_consistency = _alloc_pass_buffers()
        tf_combined, tf_logits, tf_class, tf_consistency = _alloc_pass_buffers()

        ar_recon_edge_list: list[Tensor] = []
        tf_recon_edge_list: list[Tensor] = []

        max_ranks = len(rank_batches)
        for rank in reversed(range(max_ranks)):
            if rank == 0 and not process_roots:
                continue

            rank_batch = rank_batches[rank]
            if rank_batch.node_indices.numel() == 0:
                continue

            rows = rank_batch.batch_indices
            nodes = rank_batch.node_indices
            num_rank_rows = nodes.shape[0]

            ar_counts = autoregressive_child_count[rows, nodes]
            tf_counts = teacher_forced_child_count[rows, nodes]
            ar_combined_rank = _normalize_to_model_sphere(
                autoregressive_child_sum[rows, nodes]
                / ar_counts.sqrt().unsqueeze(-1)
            )
            tf_combined_rank = _normalize_to_model_sphere(
                teacher_forced_child_sum[rows, nodes]
                / tf_counts.sqrt().unsqueeze(-1)
            )
            ar_combined[rows, nodes] = ar_combined_rank
            tf_combined[rows, nodes] = tf_combined_rank

            ar_mean = autoregressive_child_sum[rows, nodes] / ar_counts.unsqueeze(-1)
            ar_var = (
                (
                    autoregressive_child_sumsq[rows, nodes] / ar_counts.unsqueeze(-1)
                    - ar_mean**2
                )
                .mean(dim=-1)
                .clamp(min=0)
            )
            ar_consistency[rows, nodes] = torch.where(
                ar_counts > 1, ar_var, torch.zeros_like(ar_var)
            )

            tf_mean = teacher_forced_child_sum[rows, nodes] / tf_counts.unsqueeze(-1)
            tf_var = (
                (
                    teacher_forced_child_sumsq[rows, nodes] / tf_counts.unsqueeze(-1)
                    - tf_mean**2
                )
                .mean(dim=-1)
                .clamp(min=0)
            )
            tf_consistency[rows, nodes] = torch.where(
                tf_counts > 1, tf_var, torch.zeros_like(tf_var)
            )

            combined_both = torch.cat([ar_combined_rank, tf_combined_rank], dim=0)
            logits_both = self.node_type_predictor(combined_both)
            ar_logits[rows, nodes] = logits_both[:num_rank_rows].to(ar_logits.dtype)
            tf_logits[rows, nodes] = logits_both[num_rank_rows:].to(tf_logits.dtype)

            rank_labels = labels[rows, nodes]
            cross_entropy_both = F.cross_entropy(
                logits_both,
                torch.cat([rank_labels, rank_labels]),
                reduction="none",
            )
            ar_class[rows, nodes] = cross_entropy_both[:num_rank_rows].to(
                ar_class.dtype
            )
            tf_class[rows, nodes] = cross_entropy_both[num_rank_rows:].to(
                tf_class.dtype
            )

            if not rank_batch.has_valid_parents:
                continue

            decode_input_both = torch.cat(
                [
                    ar_combined[rows, nodes],
                    encoder_buffer[rows, nodes],
                ],
                dim=0,
            )
            subtypes_both = torch.cat([rank_batch.subtypes, rank_batch.subtypes])
            leaf_embeddings_for_rank = leaf_embeddings_by_graph[rows]
            leaf_embeddings_both = torch.cat(
                [leaf_embeddings_for_rank, leaf_embeddings_for_rank],
                dim=0,
            )
            predicted_both = self.node_decoder.forward_batch(
                decode_input_both,
                subtypes_both,
                leaf_embeddings_both,
            )
            ar_predicted = predicted_both[:num_rank_rows]
            tf_predicted = predicted_both[num_rank_rows:]

            # Scatter every graph's predicted parents into the child buffers
            # without ever reading mask contents back to the host. We keep the
            # full padded [R, K] edge grid and zero out the invalid slots via the
            # mask instead of boolean-compacting it (which would trigger a
            # nonzero() + D2H sync). Invalid slots already carry parent index 0
            # and now contribute a zero vector, so adding them is a true no-op;
            # multiplying valid slots by 1.0 is exact, so the accumulated values
            # are bitwise identical to the compacted version.
            #
            # As in the per-rank batched scatter, flattening the [B, N, ...]
            # buffers to [B*N, ...] and offsetting each edge's parent by its
            # graph row (batch * num_nodes + parent) gives every graph a disjoint
            # index range, so one index_add_ per buffer accumulates the batch.
            mask = rank_batch.valid_parent_mask
            edge_weight = mask.reshape(-1).to(autoregressive_child_count.dtype)
            global_parent_indices = (
                rows[:, None] * num_nodes + rank_batch.parent_indices
            ).reshape(-1)

            ar_contrib = ar_predicted.reshape(-1, self.node_embedding_dim) * (
                edge_weight.unsqueeze(-1)
            )
            tf_contrib = tf_predicted.reshape(-1, self.node_embedding_dim) * (
                edge_weight.unsqueeze(-1)
            )

            if self.compute_reconstruction_loss:
                valid_parent_mask = mask
                flat_parent_indices = rank_batch.parent_indices[valid_parent_mask]
                flat_batch_indices = rows[:, None].expand_as(rank_batch.parent_indices)[
                    valid_parent_mask
                ]
                ar_flat = ar_predicted[valid_parent_mask]
                tf_flat = tf_predicted[valid_parent_mask]
                recon_target = encoder_buffer[flat_batch_indices, flat_parent_indices]
                if self.reconstruction_detach_target:
                    recon_target = recon_target.detach()
                ar_recon_edge_list.append(
                    1.0
                    - F.cosine_similarity(
                        ar_flat.to(recon_target.dtype), recon_target, dim=-1
                    )
                )
                tf_recon_edge_list.append(
                    1.0
                    - F.cosine_similarity(
                        tf_flat.to(recon_target.dtype), recon_target, dim=-1
                    )
                )

            autoregressive_child_sum.view(batch_size * num_nodes, -1).index_add_(
                0, global_parent_indices, ar_contrib.to(autoregressive_child_sum.dtype)
            )
            autoregressive_child_count.view(-1).index_add_(
                0, global_parent_indices, edge_weight
            )
            autoregressive_child_sumsq.view(batch_size * num_nodes, -1).index_add_(
                0,
                global_parent_indices,
                (ar_contrib**2).to(autoregressive_child_sumsq.dtype),
            )
            teacher_forced_child_sum.view(batch_size * num_nodes, -1).index_add_(
                0, global_parent_indices, tf_contrib.to(teacher_forced_child_sum.dtype)
            )
            teacher_forced_child_count.view(-1).index_add_(
                0, global_parent_indices, edge_weight
            )
            teacher_forced_child_sumsq.view(batch_size * num_nodes, -1).index_add_(
                0,
                global_parent_indices,
                (tf_contrib**2).to(teacher_forced_child_sumsq.dtype),
            )

        _empty = encoder_buffer.new_zeros(0)
        ar_recon = torch.cat(ar_recon_edge_list) if ar_recon_edge_list else _empty
        tf_recon = torch.cat(tf_recon_edge_list) if tf_recon_edge_list else _empty

        return (
            (ar_combined, ar_logits, ar_class, ar_recon, ar_consistency),
            (tf_combined, tf_logits, tf_class, tf_recon, tf_consistency),
        )

    @torch.no_grad()
    def blind_autoregressive_decode(
        self,
        output_embeddings: Tensor,
        *,
        similarity_threshold: float = 0.99,
        root_match_margin: float | None = None,
        max_nodes: int = 4096,
        return_diagnostics: bool = False,
    ) -> FixedInDegreeDAGDescription | tuple[FixedInDegreeDAGDescription, dict]:
        """Recover a DAG from ordered output-node embeddings via a global pool.

        The decode walks backward from the fixed output slots. Each discovered
        trunk node is expanded once from the embedding that created it; duplicate
        parent predictions are merged by cosine similarity against one global
        match-target pool (roots plus discovered trunks, never outputs).
        """
        if output_embeddings.ndim != 2:
            raise ValueError("output_embeddings must have shape [num_outputs, D]")
        if output_embeddings.shape != (self.num_output_nodes, self.node_embedding_dim):
            raise ValueError(
                "output_embeddings must have shape "
                f"({self.num_output_nodes}, {self.node_embedding_dim}); got "
                f"{tuple(output_embeddings.shape)}"
            )
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")

        root_table = self.root_node_embeddings.weight.detach()
        device = root_table.device
        dtype = root_table.dtype
        output_embeddings = output_embeddings.to(device=device, dtype=dtype)

        pool: list[_DecodeNode] = []
        root_start = self.num_trunk_node_types
        output_start = self.num_trunk_node_types + self.num_root_nodes

        for root_slot in range(self.num_root_nodes):
            pool.append(
                _DecodeNode(
                    id=len(pool),
                    embedding=root_table[root_slot].detach().clone(),
                    type_id=root_start + root_slot,
                    is_root=True,
                    is_output=False,
                    parents=[],
                    expanded=True,
                )
            )

        frontier: list[int] = []
        for output_slot in range(self.num_output_nodes):
            node_id = len(pool)
            pool.append(
                _DecodeNode(
                    id=node_id,
                    embedding=output_embeddings[output_slot].detach().clone(),
                    type_id=output_start + output_slot,
                    is_root=False,
                    is_output=True,
                    parents=[None],
                )
            )
            frontier.append(node_id)

        termination_reason = "natural"
        if len(pool) > max_nodes:
            termination_reason = "max_nodes"
            frontier = []

        while frontier:
            if len(pool) >= max_nodes:
                termination_reason = "max_nodes"
                break

            predictions = self._blind_decode_expand(
                pool,
                frontier,
                output_embeddings,
            )
            frontier = self._blind_decode_integrate(
                pool,
                predictions,
                similarity_threshold=similarity_threshold,
                root_match_margin=root_match_margin,
            )

            if len(pool) >= max_nodes and frontier:
                termination_reason = "max_nodes"
                break

        description = self._decode_pool_to_description(pool, termination_reason)
        description.termination_reason = termination_reason
        description.decode_pool_size = len(pool)

        if return_diagnostics:
            diagnostics = {
                "termination_reason": termination_reason,
                "pool_size": len(pool),
                "recovered_nodes": description.num_nodes,
                "dropped_pool_nodes": getattr(
                    description, "decode_dropped_pool_nodes", []
                ),
            }
            return description, diagnostics
        return description

    def _blind_decode_expand(
        self,
        pool: list[_DecodeNode],
        frontier: list[int],
        leaf_embeddings: Tensor,
    ) -> list[_ParentPrediction]:
        node_ids: list[int] = []
        subtypes: list[int] = []
        for node_id in frontier:
            node = pool[node_id]
            if node.expanded:
                continue
            if node.is_root:
                continue
            if node.type_id is None:
                raise ValueError(f"frontier node {node_id} has unknown type")
            node_ids.append(node_id)
            subtypes.append(node.type_id)

        predictions: list[_ParentPrediction] = []
        if not node_ids:
            return predictions

        node_embeddings = torch.stack([pool[node_id].embedding for node_id in node_ids])
        subtype_tensor = torch.as_tensor(
            subtypes,
            dtype=torch.long,
            device=node_embeddings.device,
        )
        leaf_embeddings = leaf_embeddings.unsqueeze(0).expand(
            node_embeddings.shape[0],
            -1,
            -1,
        )
        parent_embeddings = self.node_decoder.forward_batch(
            node_embeddings,
            subtype_tensor,
            leaf_embeddings,
        )

        for row_idx, child_id in enumerate(node_ids):
            pool[child_id].expanded = True
            type_id = subtypes[row_idx]
            for slot in range(self._in_degree_for_type(type_id)):
                predictions.append(
                    _ParentPrediction(
                        child_id=child_id,
                        slot=slot,
                        embedding=parent_embeddings[row_idx, slot].detach().clone(),
                    )
                )

        return predictions

    def _blind_decode_integrate(
        self,
        pool: list[_DecodeNode],
        predictions: list[_ParentPrediction],
        *,
        similarity_threshold: float,
        root_match_margin: float | None,
    ) -> list[int]:
        if not predictions:
            return []

        pred_embs = torch.stack([p.embedding for p in predictions])
        target_ids, target_embs = self._decode_match_targets(pool, pred_embs)
        assignments, new_components = self._assign_node_ids(
            pool=pool,
            pred_embs=pred_embs,
            target_ids=target_ids,
            target_embs=target_embs,
            threshold=similarity_threshold,
            root_match_margin=root_match_margin,
        )

        next_frontier: list[int] = []
        for component in new_components:
            representative = pred_embs[component[0]].detach().clone()
            node_id = self._classify_and_add_decode_node(pool, representative)
            if not pool[node_id].is_root:
                next_frontier.append(node_id)
            for pred_idx in component:
                assignments[pred_idx] = node_id

        for prediction, node_id in zip(predictions, assignments):
            if node_id is None:
                raise RuntimeError("unassigned decode prediction")
            pool[prediction.child_id].parents[prediction.slot] = node_id

        return next_frontier

    def _decode_match_targets(
        self,
        pool: list[_DecodeNode],
        pred_embs: Tensor,
    ) -> tuple[list[int], Tensor]:
        target_ids = [node.id for node in pool if not node.is_output]
        if not target_ids:
            return target_ids, pred_embs.new_zeros((0, self.node_embedding_dim))
        return target_ids, torch.stack(
            [pool[node_id].embedding for node_id in target_ids]
        )

    def _assign_node_ids(
        self,
        *,
        pool: list[_DecodeNode],
        pred_embs: Tensor,
        target_ids: list[int],
        target_embs: Tensor,
        threshold: float,
        root_match_margin: float | None,
    ) -> tuple[list[int | None], list[list[int]]]:
        assignments: list[int | None] = [None] * pred_embs.shape[0]

        if target_embs.numel():
            pred_norm = F.normalize(pred_embs.float(), dim=-1)
            target_norm = F.normalize(target_embs.float(), dim=-1)
            similarities = pred_norm @ target_norm.t()
            best_similarities, best_target_positions = similarities.max(dim=1)

            nonroot_target_positions = [
                i for i, node_id in enumerate(target_ids) if not pool[node_id].is_root
            ]
            for pred_idx, best_similarity in enumerate(best_similarities.tolist()):
                if best_similarity <= threshold:
                    continue
                target_pos = int(best_target_positions[pred_idx])
                target_id = target_ids[target_pos]

                if root_match_margin is not None and pool[target_id].is_root:
                    best_nonroot_similarity = float("-inf")
                    if nonroot_target_positions:
                        nonroot_sims = similarities[pred_idx, nonroot_target_positions]
                        best_nonroot_similarity = float(nonroot_sims.max())
                    if best_similarity < best_nonroot_similarity + root_match_margin:
                        continue

                assignments[pred_idx] = target_id

        unmatched = [idx for idx, node_id in enumerate(assignments) if node_id is None]
        if not unmatched:
            return assignments, []

        parent = list(range(len(unmatched)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri = find(i)
            rj = find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        unmatched_embs = pred_embs[unmatched]
        if unmatched_embs.shape[0] > 1:
            unmatched_norm = F.normalize(unmatched_embs.float(), dim=-1)
            similarities = unmatched_norm @ unmatched_norm.t()
            for i in range(unmatched_embs.shape[0]):
                for j in range(i + 1, unmatched_embs.shape[0]):
                    if float(similarities[i, j]) > threshold:
                        union(i, j)

        components_by_root: dict[int, list[int]] = {}
        for local_idx, pred_idx in enumerate(unmatched):
            components_by_root.setdefault(find(local_idx), []).append(pred_idx)

        components = sorted(
            components_by_root.values(),
            key=lambda component: component[0],
        )
        return assignments, components

    def _classify_and_add_decode_node(
        self,
        pool: list[_DecodeNode],
        representative: Tensor,
    ) -> int:
        output_start = self.num_trunk_node_types + self.num_root_nodes

        logits = self.node_type_predictor(representative.unsqueeze(0))[0].float()
        logits = logits.clone()
        logits[output_start:] = float("-inf")
        type_id = int(logits.argmax())

        supertype = subtype_to_supertype(
            type_id,
            num_trunk_node_types=self.num_trunk_node_types,
            num_root_nodes=self.num_root_nodes,
            num_output_nodes=self.num_output_nodes,
        )

        if supertype is NodeSupertype.ROOT:
            root_embeddings = self.root_node_embeddings.weight.detach()
            root_sims = (
                F.normalize(representative.float(), dim=-1).unsqueeze(0)
                @ F.normalize(root_embeddings.float(), dim=-1).t()
            )
            root_slot = int(root_sims.argmax())
            return root_slot

        if supertype is not NodeSupertype.TRUNK:
            raise RuntimeError(f"masked classifier selected non-parent type {type_id}")

        node_id = len(pool)
        pool.append(
            _DecodeNode(
                id=node_id,
                embedding=representative.detach().clone(),
                type_id=type_id,
                is_root=False,
                is_output=False,
                parents=[None] * self.trunk_node_in_degrees[type_id],
            )
        )
        return node_id

    def _decode_pool_to_description(
        self,
        pool: list[_DecodeNode],
        termination_reason: str,
    ) -> FixedInDegreeDAGDescription:
        live = self._live_decode_node_ids(pool)

        trunk_ids = [
            node.id
            for node in pool
            if node.id in live and not node.is_root and not node.is_output
        ]
        trunk_order = self._toposort_decode_trunks(pool, trunk_ids)
        output_ids = [
            node.id
            for node in sorted(
                pool,
                key=lambda n: (
                    n.type_id if n.type_id is not None else self.num_node_types,
                    n.id,
                ),
            )
            if node.id in live and node.is_output
        ]

        root_start = self.num_trunk_node_types
        output_start = self.num_trunk_node_types + self.num_root_nodes

        node_inputs_indices: list[list[int]] = [[] for _ in range(self.num_root_nodes)]
        node_types: list[int] = [
            root_start + root_slot for root_slot in range(self.num_root_nodes)
        ]

        final_index: dict[int, int] = {
            root_slot: root_slot for root_slot in range(self.num_root_nodes)
        }
        for old_id in trunk_order:
            final_index[old_id] = len(node_types)
            node = pool[old_id]
            if node.type_id is None:
                raise ValueError(f"live trunk node {old_id} has unknown type")
            parents = self._require_live_parents(node, final_index)
            node_inputs_indices.append(parents)
            node_types.append(node.type_id)

        for output_slot, old_id in enumerate(output_ids):
            final_index[old_id] = len(node_types)
            node = pool[old_id]
            parents = self._require_live_parents(node, final_index)
            if len(parents) != 1:
                raise ValueError(f"output node {old_id} has {len(parents)} parents")
            node_inputs_indices.append(parents)
            node_types.append(output_start + output_slot)

        description = FixedInDegreeDAGDescription(
            num_root_nodes=self.num_root_nodes,
            num_trunk_nodes=len(trunk_order),
            num_output_nodes=len(output_ids),
            num_trunk_node_types=self.num_trunk_node_types,
            trunk_node_in_degrees=self.trunk_node_in_degrees,
            node_inputs_indices=node_inputs_indices,
            node_types=node_types,
        )
        description.termination_reason = termination_reason
        description.decode_live_pool_nodes = sorted(live)
        description.decode_dropped_pool_nodes = [
            node.id for node in pool if node.id not in live
        ]
        return description

    def _live_decode_node_ids(self, pool: list[_DecodeNode]) -> set[int]:
        live = {node.id for node in pool if node.is_root}
        changed = True
        while changed:
            changed = False
            for node in pool:
                if node.id in live or node.is_root:
                    continue
                if not node.expanded:
                    continue
                if any(parent is None or parent not in live for parent in node.parents):
                    continue
                live.add(node.id)
                changed = True
        return live

    def _toposort_decode_trunks(
        self,
        pool: list[_DecodeNode],
        trunk_ids: list[int],
    ) -> list[int]:
        trunk_set = set(trunk_ids)
        indegree = {node_id: 0 for node_id in trunk_ids}
        children: dict[int, list[int]] = {node_id: [] for node_id in trunk_ids}

        for node_id in trunk_ids:
            for parent_id in pool[node_id].parents:
                if parent_id in trunk_set:
                    indegree[node_id] += 1
                    children[parent_id].append(node_id)

        ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
        order: list[int] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for child_id in sorted(children[node_id]):
                indegree[child_id] -= 1
                if indegree[child_id] == 0:
                    ready.append(child_id)
                    ready.sort()

        if len(order) != len(trunk_ids):
            stalled = sorted(set(trunk_ids) - set(order))
            raise ValueError(
                f"cycle or invalid parent relation among trunks: {stalled}"
            )

        return order

    def _require_live_parents(
        self,
        node: _DecodeNode,
        final_index: dict[int, int],
    ) -> list[int]:
        parents: list[int] = []
        for parent_id in node.parents:
            if parent_id is None:
                raise ValueError(f"live node {node.id} has an unfilled parent slot")
            if parent_id not in final_index:
                raise ValueError(
                    f"live node {node.id} references non-live parent {parent_id}"
                )
            parents.append(final_index[parent_id])
        return parents
