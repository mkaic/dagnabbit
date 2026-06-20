from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    NodeSupertype,
    subtype_to_supertype,
)


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


class ResidualFreeTransformerBlock(nn.Module):
    """Self-attention + MLP block with replacement updates, not residual adds."""

    def __init__(
        self,
        node_embedding_dim: int,
        num_heads: int,
        transformer_mlp_depth: int,
        mlp_expansion_factor: float,
        dropout: float,
    ):
        super().__init__()
        if transformer_mlp_depth < 0:
            raise ValueError("transformer_mlp_depth must be non-negative")
        hidden_dim = max(1, round(node_embedding_dim * mlp_expansion_factor))
        self.attn_norm = nn.LayerNorm(node_embedding_dim)
        self.attn = nn.MultiheadAttention(
            node_embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(node_embedding_dim)
        self.ff = nn.Sequential(
            *_feed_forward_layers(
                [node_embedding_dim]
                + [hidden_dim] * transformer_mlp_depth
                + [node_embedding_dim],
                dropout,
            )
        )
        self.ff_norm = nn.LayerNorm(node_embedding_dim)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        y = self.attn_norm(x)
        y, _ = self.attn(
            y,
            y,
            y,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.post_attn_norm(y)
        return self.ff_norm(self.ff(x))


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
                ResidualFreeTransformerBlock(
                    node_embedding_dim=node_embedding_dim,
                    num_heads=num_heads,
                    transformer_mlp_depth=transformer_mlp_depth,
                    mlp_expansion_factor=mlp_expansion_factor,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(node_embedding_dim)
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

        return self.output_norm(x[:, : self.max_context_length])


class TransformerNodeEncoder(nn.Module):
    def __init__(self, sequence_transformer: TypeConditionedSequenceTransformer):
        super().__init__()
        self.sequence_transformer = sequence_transformer
        self.output_norm = nn.LayerNorm(sequence_transformer.node_embedding_dim)

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
        return self.output_norm(pooled)


class TransformerNodeDecoder(nn.Module):
    def __init__(self, sequence_transformer: TypeConditionedSequenceTransformer):
        super().__init__()
        self.sequence_transformer = sequence_transformer

    def forward_batch(self, node_embeddings: Tensor, subtypes: Tensor) -> Tensor:
        context = node_embeddings.unsqueeze(1).expand(
            -1,
            self.sequence_transformer.max_context_length,
            -1,
        )
        return self.sequence_transformer(context, subtypes)


@dataclass
class TrainingStepLossReturnType:
    # 1-D tensor over all nodes in the primary graph.
    primary_node_classification_losses: Tensor
    # Raw per-node logits ``[N, num_types]`` and true type labels (1-D
    # ``LongTensor``) for downstream diagnostics. ``primary_node_predicted_type_logits[i]``
    # corresponds to ``primary_node_true_types[i]``.
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
    # averaged over D. Node-indexed 1-D tensor; zero for count <= 1 nodes.
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
class _RankBatch:
    node_indices: Tensor
    parent_indices: Tensor
    valid_parent_mask: Tensor
    subtypes: Tensor


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
        transformer_num_layers: int = 1,
        transformer_mlp_depth: int = 1,
        transformer_num_register_tokens: int = 2,
        transformer_num_heads: int = 4,
        transformer_dropout: float = 0.0,
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
        self.maximum_indegree = max([1, *self.trunk_node_in_degrees])
        self.transformer_num_layers = transformer_num_layers
        self.transformer_mlp_depth = transformer_mlp_depth
        self.transformer_num_register_tokens = transformer_num_register_tokens
        self.transformer_num_heads = transformer_num_heads
        self.transformer_dropout = transformer_dropout

        # ---- Shared, type-conditioned node transformers ----
        # The public context is the ordered parent-slot sequence padded to
        # maximum_indegree. Each transformer appends its own registers and one
        # node-type token internally, so all trunk and output types share weights.
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
            )
        )
        self.node_decoder = TransformerNodeDecoder(
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
            )
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

    def evaluate_graph(
        self,
        graph: FixedInDegreeDAGDescription,
        root_node_embeddings: Tensor | None = None,
    ) -> Tensor:
        if root_node_embeddings is None:
            root_node_embeddings = self.root_node_embeddings.weight
        assert root_node_embeddings.shape[0] == graph.num_root_nodes

        device = root_node_embeddings.device
        embeddings_buffer = torch.empty(
            (graph.num_nodes, self.node_embedding_dim),
            dtype=root_node_embeddings.dtype,
            device=device,
        )
        embeddings_buffer[: graph.num_root_nodes] = root_node_embeddings

        # Walk topological ranks ascending. Rank 0 roots are seeded above. Every
        # non-root node in a rank can be encoded together because node type is an
        # input token to the shared transformer rather than a module selector.
        for rank, groups in enumerate(graph.rank_groups):
            if rank == 0:
                continue
            rank_batch = self._make_rank_batch(groups, device)
            parent_embeddings = embeddings_buffer[rank_batch.parent_indices]
            parent_embeddings = parent_embeddings.masked_fill(
                ~rank_batch.valid_parent_mask.unsqueeze(-1),
                0.0,
            )
            embeddings_buffer[rank_batch.node_indices] = (
                self.node_encoder.forward_batch(
                    parent_embeddings,
                    rank_batch.subtypes,
                    rank_batch.valid_parent_mask,
                ).to(embeddings_buffer.dtype)
            )

        return embeddings_buffer

    def _make_rank_batch(
        self,
        groups: list,
        device: torch.device,
    ) -> _RankBatch:
        if not groups:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return _RankBatch(
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

        node_indices = torch.cat([g.node_buffer_indices for g in groups]).to(device)
        subtypes = torch.cat([g.subtypes for g in groups]).to(device)
        parent_indices = torch.zeros(
            node_indices.shape[0],
            self.maximum_indegree,
            dtype=torch.long,
            device=device,
        )
        valid_parent_mask = torch.zeros(
            node_indices.shape[0],
            self.maximum_indegree,
            dtype=torch.bool,
            device=device,
        )

        offset = 0
        for group in groups:
            group_parents = group.parent_buffer_gather_indices.to(device)
            group_size, in_degree = group_parents.shape
            if in_degree > self.maximum_indegree:
                raise ValueError(
                    f"rank contains in-degree {in_degree}, above maximum "
                    f"{self.maximum_indegree}"
                )
            if in_degree:
                parent_indices[offset : offset + group_size, :in_degree] = group_parents
                valid_parent_mask[offset : offset + group_size, :in_degree] = True
            offset += group_size

        order = node_indices.argsort()
        return _RankBatch(
            node_indices=node_indices[order],
            parent_indices=parent_indices[order],
            valid_parent_mask=valid_parent_mask[order],
            subtypes=subtypes[order],
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

    def training_forward(
        self,
        primary_graph: FixedInDegreeDAGDescription,
        return_buffers: bool = False,
    ) -> (
        TrainingStepLossReturnType
        | tuple[
            TrainingStepLossReturnType,
            Tensor,
            Tensor,
        ]
    ):
        """Returns losses for the training step.

        When ``return_buffers`` is True, additionally returns the primary-graph
        encode buffer (per-node embeddings from the forward pass) and the primary
        decode buffer: a node-indexed ``[num_nodes, D]`` tensor of each node's
        decode-side combined prediction, for diagnostics such as signal
        propagation analysis.
        """

        primary_buffer = self.evaluate_graph(
            graph=primary_graph,
            root_node_embeddings=self.root_node_embeddings.weight,
        )

        device = primary_buffer.device

        # ---- Per-graph labels and class-balance weights ----
        # Shared by both decode passes (same graph, same node types).
        primary_labels = torch.as_tensor(
            list(primary_graph.node_types), dtype=torch.long, device=device
        )
        primary_weights = torch.as_tensor(
            _class_balance_weights(list(primary_graph.node_types)),
            dtype=primary_buffer.dtype,
            device=device,
        )

        # Two decode passes over the *same* (shared) encode pass. The
        # autoregressive pass is the genuine model -- predictions compound down
        # each DAG. The teacher-forced pass instead feeds every node its true
        # encode embedding, severing the autoregressive chain so the decoders are
        # asked to recover each parent's identity from a clean input. The combine
        # / classify targets are identical between the two; only what the
        # decoders are fed differs (see ``_decode_graph``).
        #
        # The two passes are independent of each other at every rank (each only
        # depends on its own higher-rank predictions), so they are run in
        # lockstep and batched together: at each rank their inputs are
        # concatenated along the batch axis for a single ``node_type_predictor``
        # / ``decode_batch`` (and loss) launch, then split back apart. Every op
        # involved is row-wise, so this is numerically identical to two separate
        # pipelines but roughly halves the kernel-launch count.
        autoregressive, teacher_forced = self._decode_pipeline(
            primary_graph=primary_graph,
            primary_buffer=primary_buffer,
            primary_labels=primary_labels,
            primary_weights=primary_weights,
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
            # `autoregressive.primary_combined` is the node-indexed [num_nodes, D]
            # decode-side combined-prediction buffer from the genuine
            # (autoregressive) pass, consumed directly by diagnostics.
            return losses, primary_buffer, autoregressive.primary_combined
        return losses

    def _decode_pipeline(
        self,
        primary_graph: FixedInDegreeDAGDescription,
        primary_buffer: Tensor,
        primary_labels: Tensor,
        primary_weights: Tensor,
        device: torch.device,
    ) -> tuple["_DecodePipelineResult", "_DecodePipelineResult"]:
        """Run the full decode over the primary graph for both the autoregressive
        and teacher-forced passes at once.

        The decode propagates predicted parent embeddings up the DAG. We hold
        two dense buffers per pass instead of per-node prediction lists:
          child_sum[n]   = running sum of embeddings predicted for node n by its
                           (already-decoded) children, and
          child_count[n] = how many such predictions have landed on n.
        A node's combined prediction is then ``child_sum / sqrt(child_count)``,
        and a child pushes into its parents with ``index_add_`` (an
        order-independent sum). The autoregressive and teacher-forced passes get
        their own fresh buffers, so they never share decode state -- but they are
        decoded in lockstep inside :meth:`_decode_graph`, which concatenates the
        two passes for each classification/loss launch.

        Returns ``(autoregressive_result, teacher_forced_result)``.
        """

        def _zeros(graph: FixedInDegreeDAGDescription, dtype: torch.dtype):
            child_sum = torch.zeros(
                graph.num_nodes,
                self.node_embedding_dim,
                dtype=dtype,
                device=device,
            )
            child_count = torch.zeros(graph.num_nodes, dtype=dtype, device=device)
            child_sumsq = torch.zeros(
                graph.num_nodes,
                self.node_embedding_dim,
                dtype=dtype,
                device=device,
            )
            return child_sum, child_count, child_sumsq

        ar_primary_sum, ar_primary_count, ar_primary_sumsq = _zeros(
            primary_graph, primary_buffer.dtype
        )
        tf_primary_sum, tf_primary_count, tf_primary_sumsq = _zeros(
            primary_graph, primary_buffer.dtype
        )

        # Seed the leaf nodes' child buffers with their own encode-side
        # embeddings. Leaves are (by definition) referenced by no other node, so
        # nothing scatters predictions onto them during the descending decode --
        # without a seed their ``child_count`` stays 0 and the combine
        # ``child_sum / sqrt(child_count)`` divides by zero. Previously the
        # condenser graph decoded the graph embedding down into these leaves; now
        # each leaf simply starts the autoregressive chain from its true encode
        # embedding (one prediction, so its combine is that embedding itself).
        # Both passes seed identically from the shared encode buffer.
        leaf_indices = torch.as_tensor(
            primary_graph.leaf_node_indices, dtype=torch.long, device=device
        )
        for child_sum, child_count, child_sumsq in (
            (ar_primary_sum, ar_primary_count, ar_primary_sumsq),
            (tf_primary_sum, tf_primary_count, tf_primary_sumsq),
        ):
            child_sum[leaf_indices] = primary_buffer[leaf_indices].to(child_sum.dtype)
            child_count[leaf_indices] = 1.0
            child_sumsq[leaf_indices] = (
                primary_buffer[leaf_indices].to(child_sumsq.dtype) ** 2
            )

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
            graph=primary_graph,
            encoder_buffer=primary_buffer,
            autoregressive_child_sum=ar_primary_sum,
            autoregressive_child_count=ar_primary_count,
            autoregressive_child_sumsq=ar_primary_sumsq,
            teacher_forced_child_sum=tf_primary_sum,
            teacher_forced_child_count=tf_primary_count,
            teacher_forced_child_sumsq=tf_primary_sumsq,
            labels=primary_labels,
            class_weights=primary_weights,
            process_roots=True,
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
        graph: FixedInDegreeDAGDescription,
        encoder_buffer: Tensor,
        autoregressive_child_sum: Tensor,
        autoregressive_child_count: Tensor,
        autoregressive_child_sumsq: Tensor,
        teacher_forced_child_sum: Tensor,
        teacher_forced_child_count: Tensor,
        teacher_forced_child_sumsq: Tensor,
        labels: Tensor,
        class_weights: Tensor,
        process_roots: bool,
        device: torch.device,
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ]:
        """Run the batched guided-autoregressive decode over one graph for the
        autoregressive and teacher-forced passes **simultaneously**.

        Walks topological ranks **descending** (every edge strictly increases
        rank, so all of a node's children -- which live at higher ranks -- have
        already pushed their predictions into the ``child_sum`` / ``child_count``
        buffers by the time the node is processed, and no two same-rank nodes are
        parent/child). At each rank it batches:

        1. the variance-preserving combine ``child_sum / sqrt(child_count)``, and
        2. classification (``node_type_predictor``) with class-balanced
           cross-entropy;

        then, per supertype group with parents, one ``decode_batch`` whose
        predicted parent embeddings are ``index_add_``-scattered into the
        parents' ``child_sum`` (and ``child_count`` incremented).

        The two passes differ only in *what each node's decoder is fed*. The
        autoregressive pass decodes each node's own predicted ``combined``
        embedding, so prediction error compounds down the DAG. The teacher-forced
        pass instead decodes each node's *true* encode-side embedding
        (``encoder_buffer``), so each parent prediction is produced from ground
        truth -- this severs the autoregressive chain and tests whether the
        decoders can recover a parent's identity (e.g. which root) from a clean
        input. Both passes' combine / classify still use their own *accumulated*
        predictions, so loss targets match between the two; only the decoder
        inputs differ.

        Because the two passes are independent of each other at every rank, their
        inputs are concatenated along the batch axis (autoregressive rows first,
        teacher-forced rows second) for a single ``node_type_predictor`` /
        ``decode_batch`` (and cross-entropy) launch, then split back apart. Every
        op is row-wise, so this is numerically identical to running the two
        passes separately while roughly halving the kernel-launch count.

        ``process_roots`` controls whether rank-0 (root) nodes are classified:
        roots have no parents so nothing is scattered regardless, but this flag
        gates whether they get a classification loss.

        Returns ``(autoregressive_tensors, teacher_forced_tensors)`` where each
        is a dense per-node tuple ``(combined, logits, classification_loss)``;
        entries for nodes that were not processed stay zero.
        """
        num_nodes = graph.num_nodes

        def _alloc_pass_buffers():
            combined = torch.zeros(
                num_nodes,
                self.node_embedding_dim,
                dtype=autoregressive_child_sum.dtype,
                device=device,
            )
            logits = torch.zeros(
                num_nodes,
                self.num_node_types,
                dtype=encoder_buffer.dtype,
                device=device,
            )
            class_losses = torch.zeros(
                num_nodes, dtype=encoder_buffer.dtype, device=device
            )
            consistency = torch.zeros(
                num_nodes, dtype=encoder_buffer.dtype, device=device
            )
            return combined, logits, class_losses, consistency

        ar_combined, ar_logits, ar_class, ar_consistency = _alloc_pass_buffers()
        tf_combined, tf_logits, tf_class, tf_consistency = _alloc_pass_buffers()

        ar_recon_edge_list: list[Tensor] = []
        tf_recon_edge_list: list[Tensor] = []

        for rank in reversed(range(len(graph.rank_groups))):
            if rank == 0 and not process_roots:
                continue
            groups = graph.rank_groups[rank]

            # All nodes at this rank, across supertype groups: combine and
            # classify uniformly (these use the shared type head regardless of
            # supertype).
            rank_node_indices = torch.cat([g.node_buffer_indices for g in groups]).to(
                device
            )
            num_rank_nodes = rank_node_indices.shape[0]

            # Per-pass combine (each pass accumulates into its own buffers).
            ar_counts = autoregressive_child_count[rank_node_indices]
            tf_counts = teacher_forced_child_count[rank_node_indices]
            ar_combined_rank = autoregressive_child_sum[
                rank_node_indices
            ] / ar_counts.sqrt().unsqueeze(-1)
            tf_combined_rank = teacher_forced_child_sum[
                rank_node_indices
            ] / tf_counts.sqrt().unsqueeze(-1)
            ar_combined[rank_node_indices] = ar_combined_rank
            tf_combined[rank_node_indices] = tf_combined_rank

            # Per-node consistency: variance of child predictions averaged over D
            # (mean, not sum, so the magnitude is invariant to embedding dim),
            # masked to nodes that received > 1 prediction.
            ar_mean = autoregressive_child_sum[rank_node_indices] / ar_counts.unsqueeze(
                -1
            )
            ar_var = (
                (
                    autoregressive_child_sumsq[rank_node_indices]
                    / ar_counts.unsqueeze(-1)
                    - ar_mean**2
                )
                .mean(dim=-1)
                .clamp(min=0)
            )
            ar_consistency[rank_node_indices] = torch.where(
                ar_counts > 1, ar_var, torch.zeros_like(ar_var)
            )

            tf_mean = teacher_forced_child_sum[rank_node_indices] / tf_counts.unsqueeze(
                -1
            )
            tf_var = (
                (
                    teacher_forced_child_sumsq[rank_node_indices]
                    / tf_counts.unsqueeze(-1)
                    - tf_mean**2
                )
                .mean(dim=-1)
                .clamp(min=0)
            )
            tf_consistency[rank_node_indices] = torch.where(
                tf_counts > 1, tf_var, torch.zeros_like(tf_var)
            )

            # Classify both passes in one launch. Rows ``[:num_rank_nodes]`` are
            # autoregressive, ``[num_rank_nodes:]`` are teacher-forced; the
            # type head is row-wise, so this is identical to two separate calls.
            combined_both = torch.cat([ar_combined_rank, tf_combined_rank], dim=0)
            logits_both = self.node_type_predictor(combined_both)
            ar_logits[rank_node_indices] = logits_both[:num_rank_nodes].to(
                ar_logits.dtype
            )
            tf_logits[rank_node_indices] = logits_both[num_rank_nodes:].to(
                tf_logits.dtype
            )

            rank_labels = labels[rank_node_indices]
            rank_weights = class_weights[rank_node_indices]
            cross_entropy_both = F.cross_entropy(
                logits_both,
                torch.cat([rank_labels, rank_labels]),
                reduction="none",
            ) * torch.cat([rank_weights, rank_weights])
            ar_class[rank_node_indices] = cross_entropy_both[:num_rank_nodes].to(
                ar_class.dtype
            )
            tf_class[rank_node_indices] = cross_entropy_both[num_rank_nodes:].to(
                tf_class.dtype
            )

            # Decode the whole rank at once. Parentless root rows are harmless:
            # their valid mask is empty, so no predicted slot is scattered.
            rank_batch = self._make_rank_batch(groups, device)
            if not rank_batch.valid_parent_mask.any():
                continue

            decode_input_both = torch.cat(
                [
                    ar_combined[rank_batch.node_indices],
                    encoder_buffer[rank_batch.node_indices],
                ],
                dim=0,
            )
            subtypes_both = torch.cat([rank_batch.subtypes, rank_batch.subtypes])
            predicted_both = self.node_decoder.forward_batch(
                decode_input_both,
                subtypes_both,
            )
            ar_predicted = predicted_both[:num_rank_nodes]
            tf_predicted = predicted_both[num_rank_nodes:]

            flat_parent_indices = rank_batch.parent_indices[
                rank_batch.valid_parent_mask
            ]
            ones = torch.ones_like(
                flat_parent_indices, dtype=autoregressive_child_count.dtype
            )

            ar_flat = ar_predicted[rank_batch.valid_parent_mask]
            tf_flat = tf_predicted[rank_batch.valid_parent_mask]

            # Per-edge reconstruction: 1 - cos(predicted, true_parent_embed).
            recon_target = encoder_buffer[flat_parent_indices]
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

            autoregressive_child_sum.index_add_(
                0,
                flat_parent_indices,
                ar_flat.to(autoregressive_child_sum.dtype),
            )
            autoregressive_child_count.index_add_(0, flat_parent_indices, ones)
            autoregressive_child_sumsq.index_add_(
                0,
                flat_parent_indices,
                (ar_flat**2).to(autoregressive_child_sumsq.dtype),
            )
            teacher_forced_child_sum.index_add_(
                0,
                flat_parent_indices,
                tf_flat.to(teacher_forced_child_sum.dtype),
            )
            teacher_forced_child_count.index_add_(0, flat_parent_indices, ones)
            teacher_forced_child_sumsq.index_add_(
                0,
                flat_parent_indices,
                (tf_flat**2).to(teacher_forced_child_sumsq.dtype),
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

            predictions = self._blind_decode_expand(pool, frontier)
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
        parent_embeddings = self.node_decoder.forward_batch(
            node_embeddings, subtype_tensor
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
