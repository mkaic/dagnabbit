"""The simulator: a DAG in, its truth table out.

Three stages, and deliberately nothing else:

1. :class:`NodeTokens` turns each node's ``(type, own position, parent
   positions)`` into one token by summing embeddings. No transformer, no
   pooling, no parent *embeddings* -- so it is a gather plus three matmuls and
   every node is computed in parallel.
2. :class:`Simulator` runs an unmasked transformer over the 152-token sequence.
   This is where the actual work happens: composing a node's gate with its
   ancestors' values is a hop of message passing, and the layer count bounds how
   many hops are available. See :mod:`dagnabbit.scripts.train_simulator` for the
   depth-stratified accuracy that measures whether that bound is binding.
3. :class:`TruthTablePatchDecoder` cross-attends a grid of learned patch queries
   against the simulated sequence. Patch ``p`` owns a contiguous block of truth
   table rows across every output, so one query decodes thousands of bits and
   the loss can be taken on a random subset of patches per step.

There is no reconstruction loss and no decode back to a graph. The token space
is never inverted: Phase 1 emits categorical choices and *constructs* tokens
from them through :class:`NodeTokens`, so what the simulator reads is the
encoding of a real graph by construction.

Position identity
-----------------
One shared table indexes canonical positions, read through a different learned
projection depending on the role the position is playing (mine, slot-0 parent,
slot-1 parent). Sharing it is what makes "find my parent's token" a dot product:
position 37 has the same underlying identity whether it appears as a node's own
index or in someone's parent pointer. For the same reason the simulator adds no
positional encoding of its own -- a second, unrelated notion of "where node 37
is" would have to be reconciled with this one.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dagnabbit.dag.canonical import CanonicalGraphs, Geometry


@dataclass(frozen=True)
class SimulatorConfig:
    embedding_dim: int = 128
    attention_head_dim: int = 32
    mlp_expansion_factor: float = 4.0
    num_simulator_layers: int = 16
    num_decoder_layers: int = 2
    # Truth-table rows are split into this many contiguous patches, each decoded
    # from one cross-attention query. 256 patches over a 65536-row table is 256
    # rows per patch, which for the 16-input geometry is "one value of a, every
    # value of b" -- one row of the truth-table image.
    num_patches: int = 256
    dropout: float = 0.0

    def num_heads(self) -> int:
        if self.embedding_dim % self.attention_head_dim != 0:
            raise ValueError(
                f"embedding_dim {self.embedding_dim} is not a multiple of "
                f"attention_head_dim {self.attention_head_dim}"
            )
        return self.embedding_dim // self.attention_head_dim


def _mlp(dim: int, expansion_factor: float, dropout: float) -> nn.Sequential:
    hidden = int(dim * expansion_factor)
    return nn.Sequential(
        nn.Linear(dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, dim),
        nn.Dropout(dropout),
    )


class SelfAttention(nn.Module):
    """Unmasked multi-head self-attention. No positional term of its own."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.projection = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0
        )
        return self.projection(attended.transpose(1, 2).reshape(batch, length, dim))


class CrossAttention(nn.Module):
    """Multi-head attention from a query sequence onto a context sequence."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.to_query = nn.Linear(dim, dim, bias=False)
        self.to_key_value = nn.Linear(dim, 2 * dim, bias=False)
        self.projection = nn.Linear(dim, dim, bias=False)

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        batch, num_queries, dim = queries.shape
        length = context.shape[1]

        query = self.to_query(queries).reshape(
            batch, num_queries, self.num_heads, self.head_dim
        )
        key_value = self.to_key_value(context).reshape(
            batch, length, 2, self.num_heads, self.head_dim
        )
        key, value = key_value.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.projection(
            attended.transpose(1, 2).reshape(batch, num_queries, dim)
        )


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + MLP."""

    def __init__(self, config: SimulatorConfig):
        super().__init__()
        dim = config.embedding_dim
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = SelfAttention(dim, config.num_heads(), config.dropout)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = _mlp(dim, config.mlp_expansion_factor, config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention onto a fixed context, then an MLP."""

    def __init__(self, config: SimulatorConfig):
        super().__init__()
        dim = config.embedding_dim
        self.norm_query = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.attention = CrossAttention(dim, config.num_heads(), config.dropout)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = _mlp(dim, config.mlp_expansion_factor, config.dropout)

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        queries = queries + self.attention(
            self.norm_query(queries), self.norm_context(context)
        )
        return queries + self.mlp(self.norm_mlp(queries))


class NodeTokens(nn.Module):
    """One token per node, from a sum of four embeddings.

    ``x_i = type[t_i] + W_self P[i] + sum_s W_s P[parent_s(i)]``, with a learned
    null vector standing in for slots the node's in-degree does not fill. Every
    table is public because Phase 1 reads them directly: a soft distribution
    over choices times a table is the expectation of the embedding, which is
    what makes the straight-through path exact in the forward direction.
    """

    def __init__(self, geometry: Geometry, config: SimulatorConfig):
        super().__init__()
        dim = config.embedding_dim
        self.geometry = geometry
        self.type_embeddings = nn.Embedding(geometry.num_node_types, dim)
        self.position_embeddings = nn.Parameter(torch.empty(geometry.num_nodes, dim))
        self.self_projection = nn.Linear(dim, dim, bias=False)
        self.parent_projections = nn.ModuleList(
            nn.Linear(dim, dim, bias=False) for _ in range(geometry.maximum_indegree)
        )
        self.null_parent = nn.Parameter(torch.empty(geometry.maximum_indegree, dim))
        self.output_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.position_embeddings, std=0.02)
        nn.init.normal_(self.null_parent, std=0.02)

    def forward(
        self,
        node_types: Tensor,
        parent_positions: Tensor,
        parent_slot_mask: Tensor,
    ) -> Tensor:
        """``[B, N] , [B, N, S], [B, N, S] -> [B, N, D]``."""
        positions = self.position_embeddings.to(dtype=self.type_embeddings.weight.dtype)
        tokens = self.type_embeddings(node_types) + self.self_projection(positions)

        for slot, projection in enumerate(self.parent_projections):
            gathered = positions[parent_positions[..., slot]]
            contribution = torch.where(
                parent_slot_mask[..., slot].unsqueeze(-1),
                projection(gathered),
                self.null_parent[slot],
            )
            tokens = tokens + contribution

        return self.output_norm(tokens)


class Simulator(nn.Module):
    """Stack of unmasked self-attention blocks over the node sequence."""

    def __init__(self, config: SimulatorConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            SelfAttentionBlock(config) for _ in range(config.num_simulator_layers)
        )
        self.output_norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens)


class TruthTablePatchDecoder(nn.Module):
    """Learned patch queries -> predicted truth-table bit logits.

    Patch ``p`` covers rows ``[p * rows_per_patch, (p + 1) * rows_per_patch)``
    for every output node, so its head emits ``num_output_nodes *
    rows_per_patch`` logits. Selecting a subset of patches per step is what
    keeps the loss affordable: the full table is half a million bits per graph.
    """

    def __init__(self, geometry: Geometry, config: SimulatorConfig):
        super().__init__()
        num_rows = geometry.num_truth_table_rows
        if num_rows % config.num_patches != 0:
            raise ValueError(
                f"{num_rows} truth-table rows do not divide into "
                f"{config.num_patches} patches"
            )
        self.rows_per_patch = num_rows // config.num_patches
        self.num_patches = config.num_patches
        self.num_output_nodes = geometry.num_output_nodes

        dim = config.embedding_dim
        self.patch_queries = nn.Parameter(torch.empty(config.num_patches, dim))
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(config) for _ in range(config.num_decoder_layers)
        )
        self.output_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, geometry.num_output_nodes * self.rows_per_patch)
        nn.init.normal_(self.patch_queries, std=0.02)

    def forward(self, sequence: Tensor, patch_indices: Tensor | None = None) -> Tensor:
        """``[B, N, D]`` -> ``[B, K, num_output_nodes, rows_per_patch]`` logits.

        ``patch_indices`` is a ``[K]`` tensor selecting which patches to decode,
        shared across the batch so the cross-attention stays one dense call.
        ``None`` decodes the whole table.
        """
        queries = self.patch_queries
        if patch_indices is not None:
            queries = queries[patch_indices]
        queries = queries.unsqueeze(0).expand(sequence.shape[0], -1, -1)

        for block in self.blocks:
            queries = block(queries, sequence)

        logits = self.head(self.output_norm(queries))
        return logits.unflatten(-1, (self.num_output_nodes, self.rows_per_patch))


class GraphSimulator(nn.Module):
    """The three stages wired together."""

    def __init__(self, geometry: Geometry, config: SimulatorConfig):
        super().__init__()
        self.geometry = geometry
        self.config = config
        self.node_tokens = NodeTokens(geometry, config)
        self.simulator = Simulator(config)
        self.decoder = TruthTablePatchDecoder(geometry, config)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        node_types: Tensor,
        parent_positions: Tensor,
        parent_slot_mask: Tensor,
        patch_indices: Tensor | None = None,
    ) -> Tensor:
        tokens = self.node_tokens(node_types, parent_positions, parent_slot_mask)
        return self.decoder(self.simulator(tokens), patch_indices)

    def forward_graphs(
        self, graphs: CanonicalGraphs, patch_indices: Tensor | None = None
    ) -> Tensor:
        return self(
            graphs.node_types,
            graphs.parent_positions,
            graphs.parent_slot_mask,
            patch_indices,
        )


def sample_patch_indices(
    num_patches: int, count: int, device: torch.device | str
) -> Tensor:
    """``[count]`` distinct patch indices, uniformly without replacement."""
    if count > num_patches:
        raise ValueError(f"cannot draw {count} of {num_patches} patches")
    return torch.randperm(num_patches, device=device)[:count]


def patch_targets(
    packed_outputs: Tensor, patch_indices: Tensor, rows_per_patch: int
) -> Tensor:
    """Packed circuit outputs -> ``[B, K, C, rows_per_patch]`` float 0/1 labels.

    The words are sliced *before* unpacking, so only the selected patches are
    ever expanded to one byte per bit. Unpacking the whole table first would
    cost 512 kB per graph for a batch that only scores a few percent of it.
    """
    from dagnabbit.tasks.logic_gates.evaluate import BITS_PER_WORD, unpack_bits

    if rows_per_patch % BITS_PER_WORD != 0:
        raise ValueError(f"{rows_per_patch} rows per patch is not a whole byte count")
    words_per_patch = rows_per_patch // BITS_PER_WORD

    batch, channels, num_words = packed_outputs.shape
    by_patch = packed_outputs.reshape(batch, channels, -1, words_per_patch)
    selected = by_patch[:, :, patch_indices]
    bits = unpack_bits(selected)
    return bits.permute(0, 2, 1, 3).float()


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def format_parameter_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


__all__ = [
    "CrossAttentionBlock",
    "GraphSimulator",
    "NodeTokens",
    "SelfAttentionBlock",
    "Simulator",
    "SimulatorConfig",
    "TruthTablePatchDecoder",
    "format_parameter_count",
    "parameter_count",
    "patch_targets",
    "sample_patch_indices",
]
