import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dagnabbit.dag.description import (
    CollatedRanks,
    FixedInDegreeDAGDescription,
    collate_rank_partitions,
)
from dagnabbit.scripts import config as cfg

# Fixed geometry of the recursive encoder's per-node transformer. Head count is
# derived from the embedding dim and the globally configured head width
# (cfg.ATTENTION_HEAD_DIM), read at construction time.
ENCODER_MLP_DEPTH = 1
ENCODER_NUM_REGISTER_TOKENS = 2
ENCODER_DROPOUT = 0.0

# Fixed geometry of the sequence-space compressor/decoder blocks, independent
# of the encoder's apart from the shared head width.
SEQUENCE_MLP_DEPTH = 1
SEQUENCE_DROPOUT = 0.0


def _derive_num_heads(node_embedding_dim: int, what: str) -> int:
    """Split ``node_embedding_dim`` into ``cfg.ATTENTION_HEAD_DIM``-wide heads.

    Read at construction time rather than bound at import so tests and scripts
    can override the config value before building a model.
    """
    head_dim = cfg.ATTENTION_HEAD_DIM
    if head_dim <= 0:
        raise ValueError(f"cfg.ATTENTION_HEAD_DIM must be positive; got {head_dim}")
    if node_embedding_dim % head_dim != 0:
        raise ValueError(
            "node_embedding_dim must be a multiple of the configured "
            f"{head_dim}-wide {what} head dim; got {node_embedding_dim}"
        )
    return node_embedding_dim // head_dim


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


class TransformerBlock(nn.Module):
    """Pre-norm self-attention + MLP block with a standard residual stream."""

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
        self.attn_dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            *_feed_forward_layers(
                [node_embedding_dim]
                + [hidden_dim] * transformer_mlp_depth
                + [node_embedding_dim],
                dropout,
            )
        )
        self.ff_norm = nn.LayerNorm(node_embedding_dim)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        y = self.attn_norm(x)
        y, _ = self.attn(
            y,
            y,
            y,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.attn_dropout(y)
        return x + self.ff_dropout(self.ff(self.ff_norm(x)))


class TypeConditionedSequenceTransformer(nn.Module):
    """Shared sequence processor for node parent slots.

    The public context length is ``max_context_length``. The module appends
    learnable register tokens and one node-type token internally, then returns
    only the transformed public context slots.

    Attention here is unmasked and order-free, so a public slot's identity
    comes entirely from its learned row of ``position_embeddings``. What each
    slot *means* is the caller's business.
    """

    def __init__(
        self,
        node_embedding_dim: int,
        num_node_types: int,
        max_context_length: int,
        num_layers: int,
        mlp_expansion_factor: float,
    ):
        super().__init__()
        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        num_heads = _derive_num_heads(node_embedding_dim, "encoder")

        self.node_embedding_dim = node_embedding_dim
        self.max_context_length = max_context_length
        self.num_heads = num_heads
        self.num_register_tokens = ENCODER_NUM_REGISTER_TOKENS
        self.position_embeddings = nn.Parameter(
            torch.empty(max_context_length, node_embedding_dim)
        )
        self.register_tokens = nn.Parameter(
            torch.empty(ENCODER_NUM_REGISTER_TOKENS, node_embedding_dim)
        )
        self.node_type_embeddings = nn.Embedding(num_node_types, node_embedding_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    node_embedding_dim=node_embedding_dim,
                    num_heads=num_heads,
                    transformer_mlp_depth=ENCODER_MLP_DEPTH,
                    mlp_expansion_factor=mlp_expansion_factor,
                    dropout=ENCODER_DROPOUT,
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
    """Pools one node's parent context into that node's embedding.

    The context is three blocks: the K parent embeddings, then the K matching
    absolute-position tokens, then one absolute-position token for the node
    being encoded. Only the embedding block is pooled into the result; the
    position tokens exist purely as attention context, telling the encoder
    *where* in the graph each parent lives rather than just which slot it
    fills. Their block layout is arbitrary -- attention is order-free, and the
    transformer's learned per-slot embeddings are what actually name the slots.
    """

    def __init__(self, sequence_transformer: TypeConditionedSequenceTransformer):
        super().__init__()
        self.sequence_transformer = sequence_transformer
        self.output_norm = nn.LayerNorm(sequence_transformer.node_embedding_dim)

    def forward_batch(
        self,
        parent_embeddings: Tensor,
        parent_position_encodings: Tensor,
        self_position_encoding: Tensor,
        subtypes: Tensor,
        valid_parent_mask: Tensor,
    ) -> Tensor:
        num_slots = parent_embeddings.shape[1]
        context = torch.cat(
            [
                parent_embeddings,
                parent_position_encodings,
                self_position_encoding.unsqueeze(1),
            ],
            dim=1,
        )
        # A padded slot masks out both of its tokens; the node's own position
        # token is always valid.
        valid_context_mask = torch.cat(
            [
                valid_parent_mask,
                valid_parent_mask,
                torch.ones_like(valid_parent_mask[:, :1]),
            ],
            dim=1,
        )
        transformed = self.sequence_transformer(
            context,
            subtypes,
            valid_context_mask,
        )
        content = transformed[:, :num_slots]
        weights = valid_parent_mask.to(dtype=content.dtype).unsqueeze(-1)
        counts = weights.sum(dim=1).clamp(min=1.0)
        pooled = (content * weights).sum(dim=1) / counts.sqrt()
        return self.output_norm(pooled)


def sinusoidal_position_encodings(length: int, dim: int) -> Tensor:
    """Fixed sin/cos position table [length, dim] (Vaswani et al. layout)."""
    if dim % 2 != 0:
        raise ValueError("dim must be even for sinusoidal position encodings")
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    encodings = torch.zeros(length, dim)
    encodings[:, 0::2] = torch.sin(positions * frequencies)
    encodings[:, 1::2] = torch.cos(positions * frequencies)
    return encodings


class SequenceTransformer(nn.Module):
    """Plain bidirectional transformer over the full canonical token sequence.

    A stack of the shared pre-norm :class:`TransformerBlock`s with a final
    LayerNorm. No causal mask and no padding: every graph in a batch has the
    same fixed node count, so attention runs dense over all positions
    (``nn.MultiheadAttention`` routes through ``scaled_dot_product_attention``).
    """

    def __init__(
        self,
        node_embedding_dim: int,
        num_layers: int,
        mlp_expansion_factor: float,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        num_heads = _derive_num_heads(node_embedding_dim, "sequence")
        self.num_heads = num_heads
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    node_embedding_dim=node_embedding_dim,
                    num_heads=num_heads,
                    transformer_mlp_depth=SEQUENCE_MLP_DEPTH,
                    mlp_expansion_factor=mlp_expansion_factor,
                    dropout=SEQUENCE_DROPOUT,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(node_embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, N, D]")
        for block in self.blocks:
            x = block(x, None)
        return self.output_norm(x)


@dataclass
class TrainingStepLossReturnType:
    """Per-position/per-slot training losses for one graph batch.

    Every tensor's position axis is the graph's canonical sequence order
    (roots first, outputs last), not original node-storage order.
    """

    # [B, T] per-trunk-position type cross-entropy. Root and output positions
    # are never classified -- the canonical layout fixes their identities by
    # construction.
    node_classification_losses: Tensor
    # [B, T, num_trunk_node_types] raw classifier logits over the trunk slice
    # of the reconstruction; ``node_true_types`` is the aligned [B, T] label
    # tensor of trunk-type ids.
    node_predicted_type_logits: Tensor
    node_true_types: Tensor
    # [B, N, S] per-slot parent-pointer cross-entropy. Invalid slots (roots,
    # padding beyond a type's in-degree) are exactly zero.
    parent_pointer_losses: Tensor
    # [B, N, S, N] pointer logits; -inf outside each position's candidate set
    # (strictly-earlier, non-output positions).
    parent_pointer_logits: Tensor
    # [B, N, S] true parent canonical positions (0-padded) and the matching
    # slot-validity mask.
    parent_pointer_true_positions: Tensor
    parent_pointer_slot_mask: Tensor


@dataclass
class _BatchedRank:
    batch_indices: Tensor
    node_indices: Tensor
    parent_indices: Tensor
    valid_parent_mask: Tensor
    subtypes: Tensor
    has_valid_parents: bool = False


class DagnabbitAutoEncoder(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        trunk_node_type_in_degrees: int | list[int],
        num_trunk_node_types: int,
        num_root_nodes: int,
        num_trunk_nodes: int,
        num_output_nodes: int,
        mlp_expansion_factor: float,
        encoder_num_layers: int = 1,
        compressor_num_layers: int | None = 4,
        decoder_num_layers: int = 4,
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
        self.num_trunk_nodes = num_trunk_nodes
        self.num_output_nodes = num_output_nodes
        self.trunk_node_in_degrees = trunk_node_type_in_degrees
        # All output nodes share one output class (they are identifiable by their
        # fixed slot positions), so outputs contribute a single type index rather
        # than one per output slot.
        self.num_node_types = num_trunk_node_types + num_root_nodes + 1
        # The canonical sequence has a fixed length and layout: roots at
        # positions [0, R), trunks at [R, output_start), outputs at
        # [output_start, num_nodes).
        self.num_nodes = num_root_nodes + num_trunk_nodes + num_output_nodes
        self.output_start = num_root_nodes + num_trunk_nodes
        self.mlp_expansion_factor = mlp_expansion_factor
        self.maximum_indegree = max([1, *self.trunk_node_in_degrees])
        self.encoder_num_layers = encoder_num_layers
        self.compressor_num_layers = compressor_num_layers
        self.decoder_num_layers = decoder_num_layers

        # ---- Recursive structural encoder ----
        # The context is the ordered parent-slot sequence padded to
        # maximum_indegree, followed by each of those parents' absolute
        # (canonical) graph position, followed by the encoded node's own
        # absolute position: 2K + 1 slots. The transformer appends its own
        # registers and one node-type token internally, so all trunk and
        # output types share weights. Block geometry is hardcoded in the
        # transformer class.
        self.node_encoder = TransformerNodeEncoder(
            TypeConditionedSequenceTransformer(
                node_embedding_dim=node_embedding_dim,
                num_node_types=self.num_node_types,
                max_context_length=2 * self.maximum_indegree + 1,
                num_layers=encoder_num_layers,
                mlp_expansion_factor=mlp_expansion_factor,
            )
        )

        # ---- Root node embeddings ----
        self.root_node_embeddings = nn.Embedding(
            self.num_root_nodes, self.node_embedding_dim
        )

        # ---- Sequence-space compressor and decoder ----
        # Both are dense bidirectional transformers over the fixed-length
        # canonical sequence, with their own weights, layer counts, and
        # hardcoded block geometry (independent of the encoder's).
        #
        # ``compressor_num_layers`` of None or 0 removes the compressor
        # entirely: the latent is then the encoder's own output-node
        # embeddings, taken straight out of the canonical sequence.
        self.compressor = (
            SequenceTransformer(
                node_embedding_dim=node_embedding_dim,
                num_layers=compressor_num_layers,
                mlp_expansion_factor=mlp_expansion_factor,
            )
            if compressor_num_layers
            else None
        )
        self.decoder = SequenceTransformer(
            node_embedding_dim=node_embedding_dim,
            num_layers=decoder_num_layers,
            mlp_expansion_factor=mlp_expansion_factor,
        )
        # Shared learned placeholder for all masked (non-output) decoder input
        # positions; position identity comes from the additive posenc.
        self.mask_token = nn.Parameter(torch.empty(node_embedding_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer(
            "position_encodings",
            sinusoidal_position_encodings(self.num_nodes, node_embedding_dim),
            persistent=False,
        )
        # Pointer candidates for position i: strictly-earlier positions that
        # are not outputs (outputs are leaves and can never be parents).
        positions = torch.arange(self.num_nodes)
        self.register_buffer(
            "pointer_candidate_mask",
            (positions[None, :] < positions[:, None])
            & (positions[None, :] < self.output_start),
            persistent=False,
        )

        # ---- Heads ----
        # Trunk-type classifier only: the canonical layout pins roots to the
        # first positions and outputs to the last positions in slot order, so
        # their identities are known by construction and never predicted.
        self.node_type_predictor = nn.Linear(
            self.node_embedding_dim,
            self.num_trunk_node_types,
        )
        # One key projector shared by all positions; one query projector per
        # input-slot index (type-agnostic -- slot 0 always uses projector 0,
        # and a node's type only decides how many slots are active).
        self.pointer_key_proj = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.pointer_slot_query_projs = nn.ModuleList(
            [
                nn.Linear(node_embedding_dim, node_embedding_dim)
                for _ in range(self.maximum_indegree)
            ]
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

        # Absolute graph position of every node, in the same coordinate system
        # the compressor, decoder, and pointer head use: canonical sequence
        # position, indexed here by node-storage index.
        canonical_positions = self._stacked_canonical_positions(graphs, device)
        position_encodings = self.position_encodings.to(
            dtype=embeddings_buffer.dtype,
        )

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
            parent_position_encodings = position_encodings[
                canonical_positions[
                    rank_batch.batch_indices[:, None],
                    rank_batch.parent_indices,
                ]
            ].masked_fill(
                ~rank_batch.valid_parent_mask.unsqueeze(-1),
                0.0,
            )
            self_position_encoding = position_encodings[
                canonical_positions[
                    rank_batch.batch_indices,
                    rank_batch.node_indices,
                ]
            ]
            embeddings_buffer[
                rank_batch.batch_indices,
                rank_batch.node_indices,
            ] = self.node_encoder.forward_batch(
                parent_embeddings,
                parent_position_encodings,
                self_position_encoding,
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

    def _validate_collated_ranks(self, collated: CollatedRanks) -> None:
        parent_width = collated.parent_indices.shape[1]
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
        collated = collate_rank_partitions(graphs, device)
        self._validate_collated_ranks(collated)

        device_ranks: list[_BatchedRank] = []
        for rank in range(collated.num_ranks):
            if not collated.counts[rank]:
                device_ranks.append(self._empty_batched_rank(device))
                continue
            rows = collated.rank_slice(rank)
            device_ranks.append(
                _BatchedRank(
                    batch_indices=collated.batch_indices[rows],
                    node_indices=collated.node_indices[rows],
                    parent_indices=collated.parent_indices[rows],
                    valid_parent_mask=collated.valid_parent_mask[rows],
                    subtypes=collated.subtypes[rows],
                    has_valid_parents=collated.has_valid_parents[rank],
                )
            )
        return device_ranks

    def _in_degree_for_type(self, node_type: int) -> int:
        if node_type < self.num_trunk_node_types:
            return self.trunk_node_in_degrees[node_type]
        output_start = self.num_trunk_node_types + self.num_root_nodes
        if node_type < output_start:
            return 0
        if node_type < self.num_node_types:
            return 1
        raise ValueError(f"unknown node type {node_type}")

    def _stacked_canonical_positions(
        self,
        graphs: Sequence[FixedInDegreeDAGDescription],
        device: torch.device,
    ) -> Tensor:
        """Stack per-graph node-index -> canonical-position maps into [B, N]."""
        return torch.stack([graph.canonical_positions_tensor for graph in graphs]).to(
            device=device, non_blocking=True
        )

    def _stacked_canonical_tensors(
        self,
        graphs: Sequence[FixedInDegreeDAGDescription],
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Stack each graph's cached canonical tensors into device batches."""
        order = torch.stack([graph.canonical_order_tensor for graph in graphs]).to(
            device=device, non_blocking=True
        )
        labels = torch.stack([graph.canonical_node_types for graph in graphs]).to(
            device=device, non_blocking=True
        )
        parent_positions = torch.stack(
            [graph.canonical_parent_positions for graph in graphs]
        ).to(device=device, non_blocking=True)
        slot_mask = torch.stack(
            [graph.canonical_parent_slot_mask for graph in graphs]
        ).to(device=device, non_blocking=True)
        return order, labels, parent_positions, slot_mask

    def compress(self, sequence: Tensor) -> Tensor:
        """Canonical node-embedding sequence [B, N, D] -> graph latent [B, K, D].

        The latent is the compressor's output at the K fixed output-node
        positions (the tail of the canonical sequence); everything else is
        discarded. With the compressor disabled
        (``compressor_num_layers=None``/``0``) the latent is the encoder's own
        embeddings at those same positions, passed through untouched.
        """
        if sequence.ndim != 3 or sequence.shape[1:] != (
            self.num_nodes,
            self.node_embedding_dim,
        ):
            raise ValueError(
                "sequence must have shape [B, "
                f"{self.num_nodes}, {self.node_embedding_dim}]"
            )
        if self.compressor is None:
            return sequence[:, self.output_start :]
        posenc = self.position_encodings.to(dtype=sequence.dtype)
        compressed = self.compressor(sequence + posenc)
        return compressed[:, self.output_start :]

    def decode_latent(self, latent: Tensor) -> Tensor:
        """Graph latent [B, K, D] -> reconstructed embeddings [B, N, D].

        Non-output positions are filled with the shared learned mask token;
        the latent tokens sit at their fixed output positions at the end of
        the sequence. Additive sinusoidal posenc gives every token (masked or
        latent) its position identity.
        """
        if latent.ndim != 3 or latent.shape[1:] != (
            self.num_output_nodes,
            self.node_embedding_dim,
        ):
            raise ValueError(
                "latent must have shape [B, "
                f"{self.num_output_nodes}, {self.node_embedding_dim}]"
            )
        batch_size = latent.shape[0]
        placeholders = self.mask_token.to(latent.dtype).expand(
            batch_size,
            self.output_start,
            -1,
        )
        tokens = torch.cat([placeholders, latent], dim=1)
        posenc = self.position_encodings.to(dtype=latent.dtype)
        return self.decoder(tokens + posenc)

    def parent_pointer_logits(self, reconstructed: Tensor) -> Tensor:
        """Reconstructed embeddings [B, N, D] -> pointer logits [B, N, S, N].

        Position i's slot-s query is a slot-specific projection of its
        reconstructed embedding; keys are one shared projection of all
        positions. Non-candidate keys (j >= i, or j in the output block) are
        -inf, so a softmax over the last axis is a distribution over exactly
        the positions that may legally be parents of i.
        """
        keys = self.pointer_key_proj(reconstructed)
        queries = torch.stack(
            [proj(reconstructed) for proj in self.pointer_slot_query_projs],
            dim=2,
        )
        logits = torch.einsum("bisd,bjd->bisj", queries, keys) / math.sqrt(
            self.node_embedding_dim
        )
        candidate = self.pointer_candidate_mask.view(
            1, self.num_nodes, 1, self.num_nodes
        )
        return logits.masked_fill(~candidate, float("-inf"))

    def training_forward_batch(
        self,
        primary_graphs: Sequence[FixedInDegreeDAGDescription],
        return_buffers: bool = False,
    ) -> TrainingStepLossReturnType | tuple[TrainingStepLossReturnType, Tensor, Tensor]:
        """Batched training forward over structurally compatible DAGs.

        Encode recursively, line the node embeddings up in canonical
        topological order, compress to the output-position latent, decode the
        masked sequence, then score node-type classification at every position
        and parent-pointer prediction at every valid input slot.

        With ``return_buffers`` the encoder buffer (original node order) and
        the reconstructed sequence (canonical order) are returned as well.
        """
        primary_graphs = list(primary_graphs)
        device = self.root_node_embeddings.weight.device
        rank_batches = self._make_batched_rank_cache(primary_graphs, device)
        primary_buffer = self.evaluate_graph_batch(
            graphs=primary_graphs,
            rank_batches=rank_batches,
        )
        batch_size, num_nodes, _ = primary_buffer.shape
        if num_nodes != self.num_nodes:
            raise ValueError(
                f"graphs have {num_nodes} nodes, but model expects {self.num_nodes}"
            )

        order, labels, parent_positions, slot_mask = self._stacked_canonical_tensors(
            primary_graphs, device
        )
        batch_rows = torch.arange(batch_size, dtype=torch.long, device=device)[:, None]
        sequence = primary_buffer[batch_rows, order]

        latent = self.compress(sequence)
        reconstructed = self.decode_latent(latent)

        # Only trunk positions are classified: the canonical layout already
        # fixes root and output identities by position.
        trunk_labels = labels[:, self.num_root_nodes : self.output_start]
        type_logits = self.node_type_predictor(
            reconstructed[:, self.num_root_nodes : self.output_start]
        )
        class_losses = F.cross_entropy(
            type_logits.reshape(-1, self.num_trunk_node_types),
            trunk_labels.reshape(-1),
            reduction="none",
        ).reshape(batch_size, self.num_trunk_nodes)

        pointer_logits = self.parent_pointer_logits(reconstructed)
        # Invalid slots (roots, padding beyond a type's in-degree) can have
        # fully -inf candidate rows (position 0) or meaningless finite rows;
        # zero their logits so cross_entropy stays finite everywhere, then
        # zero their losses via the slot mask. Valid rows keep their -inf
        # non-candidates, which softmax treats as exact zeros.
        safe_logits = torch.where(
            slot_mask.unsqueeze(-1),
            pointer_logits,
            torch.zeros_like(pointer_logits),
        )
        pointer_losses = F.cross_entropy(
            safe_logits.reshape(-1, num_nodes),
            parent_positions.reshape(-1),
            reduction="none",
        ).reshape(batch_size, num_nodes, self.maximum_indegree)
        pointer_losses = pointer_losses * slot_mask.to(pointer_losses.dtype)

        losses = TrainingStepLossReturnType(
            node_classification_losses=class_losses,
            node_predicted_type_logits=type_logits,
            node_true_types=trunk_labels,
            parent_pointer_losses=pointer_losses,
            parent_pointer_logits=pointer_logits,
            parent_pointer_true_positions=parent_positions,
            parent_pointer_slot_mask=slot_mask,
        )
        if return_buffers:
            return losses, primary_buffer, reconstructed
        return losses

    @torch.no_grad()
    def encode_to_latent(
        self,
        graphs: Sequence[FixedInDegreeDAGDescription],
    ) -> Tensor:
        """Encode graphs and compress to their [B, K, D] graph latents."""
        graphs = list(graphs)
        device = self.root_node_embeddings.weight.device
        rank_batches = self._make_batched_rank_cache(graphs, device)
        buffer = self.evaluate_graph_batch(graphs=graphs, rank_batches=rank_batches)
        order = torch.stack([graph.canonical_order_tensor for graph in graphs]).to(
            device=device, non_blocking=True
        )
        batch_rows = torch.arange(len(graphs), dtype=torch.long, device=device)[:, None]
        return self.compress(buffer[batch_rows, order])

    @torch.no_grad()
    def generate_choices(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """Decode ``[B, K, D]`` latents into argmax choice tensors, on-device.

        Returns ``(trunk_types, parent_choices)``: ``[B, T]`` trunk class ids
        and ``[B, N, S]`` parent canonical positions. This is the whole of
        generation except the Python description objects -- callers that only
        need to *evaluate* the graphs (e.g. bitpacked circuit evaluation) can
        consume these tensors directly and never leave the device;
        :meth:`generate` builds the descriptions on top.
        """
        reconstructed = self.decode_latent(latent)
        trunk_types = self.node_type_predictor(
            reconstructed[:, self.num_root_nodes : self.output_start]
        ).argmax(dim=-1)
        parent_choices = self.parent_pointer_logits(reconstructed).argmax(dim=-1)
        return trunk_types, parent_choices

    @torch.no_grad()
    def generate(
        self,
        latent: Tensor,
    ) -> FixedInDegreeDAGDescription | list[FixedInDegreeDAGDescription]:
        """Decode graph latents into guaranteed-valid DAG descriptions.

        ``latent`` is ``[B, K, D]``, or ``[K, D]`` for a single graph (which
        returns a single description). Types at root/output positions are
        fixed by the sequence layout; each trunk position takes the argmax
        over trunk classes, and its predicted type decides how many input
        slots to fill. Every valid slot then points at its argmax candidate.
        Candidates are restricted to strictly-earlier non-output positions,
        so parents always precede children: the result is a valid DAG by
        construction and generation cannot fail to terminate.

        Descriptions are built directly in canonical position space (node
        index == canonical position). A slot may select the same parent as a
        sibling slot, and producers that no slot selects stay as dead nodes;
        ``graphs_match`` ignores both when comparing graphs.
        """
        single = latent.ndim == 2
        if single:
            latent = latent.unsqueeze(0)

        trunk_types, parent_choices = self.generate_choices(latent)
        descriptions = self.descriptions_from_choices(trunk_types, parent_choices)
        return descriptions[0] if single else descriptions

    def descriptions_from_choices(
        self,
        trunk_types: Tensor,
        parent_choices: Tensor,
    ) -> list[FixedInDegreeDAGDescription]:
        """Assemble DAG descriptions from already-chosen types and pointers.

        ``trunk_types`` is ``[B, T]`` of trunk class ids and ``parent_choices``
        is ``[B, N, S]`` of canonical positions, indexed by canonical position
        over *all* nodes (root rows are ignored). How those choices were made
        is the caller's business: :meth:`generate` takes argmaxes, while an RL
        policy samples them.

        Validity does not depend on the choices. Candidates were already
        restricted to strictly-earlier non-output positions when the logits
        were built, so parents always precede children and the result is a
        valid DAG however the selection was made.
        """
        trunk_types_by_graph = trunk_types.detach().cpu().tolist()
        parents_by_graph = parent_choices.detach().cpu().tolist()

        root_types_start = self.num_trunk_node_types
        output_type = self.num_trunk_node_types + self.num_root_nodes
        descriptions: list[FixedInDegreeDAGDescription] = []
        for graph_idx in range(len(trunk_types_by_graph)):
            node_types = [
                root_types_start + root_slot for root_slot in range(self.num_root_nodes)
            ]
            node_inputs_indices: list[list[int]] = [
                [] for _ in range(self.num_root_nodes)
            ]
            for trunk_offset, trunk_type in enumerate(trunk_types_by_graph[graph_idx]):
                position = self.num_root_nodes + trunk_offset
                in_degree = self.trunk_node_in_degrees[trunk_type]
                node_types.append(trunk_type)
                node_inputs_indices.append(
                    parents_by_graph[graph_idx][position][:in_degree]
                )
            for output_slot in range(self.num_output_nodes):
                position = self.output_start + output_slot
                node_types.append(output_type)
                node_inputs_indices.append(parents_by_graph[graph_idx][position][:1])
            descriptions.append(
                FixedInDegreeDAGDescription(
                    num_root_nodes=self.num_root_nodes,
                    num_trunk_nodes=self.num_trunk_nodes,
                    num_output_nodes=self.num_output_nodes,
                    num_trunk_node_types=self.num_trunk_node_types,
                    trunk_node_in_degrees=self.trunk_node_in_degrees,
                    node_inputs_indices=node_inputs_indices,
                    node_types=node_types,
                )
            )

        return descriptions
