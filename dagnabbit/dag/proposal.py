"""Task-agnostic machinery for training something to *write* a graph latent.

The autoencoder's latent is an interchange format: ``decode_latent`` turns any
``[K, D]`` tensor into a structurally valid DAG. A **proposer** is anything that
produces such a tensor from a task's specification -- a truth table, a set of
I/O examples, a target trajectory. What the specification looks like, and how
it is read, is the task's business. What happens *after* the latent exists is
not, and that is what lives here:

* :class:`LatentReadout` -- the final stage every proposer shares, turning a
  variable number of context tokens into the fixed ``[K, D]`` latent shape.
* :func:`reconstruction_losses` -- scoring a proposed latent against the graph
  it was supposed to describe, reusing the autoencoder's own type and pointer
  cross-entropy.

Neither knows what a circuit is, and nothing in this module may import from
:mod:`dagnabbit.tasks`. A new task supplies its own specification encoder, ends
it with :class:`LatentReadout`, and trains it with
:func:`reconstruction_losses`.
"""

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import FixedInDegreeDAGDescription

READOUT_DROPOUT = 0.0


class LatentReadout(nn.Module):
    """Context tokens ``[B, T, D]`` -> graph latent ``[B, K, D]``.

    ``K`` learned queries cross-attend over however many context tokens the
    specification encoder produced, so the latent shape is decoupled from the
    specification's size: a task with 256 patch tokens and one with 40 example
    tokens both land on the same ``[K, D]``.

    The output is projected onto a shell of radius ``sqrt(D)``. Encoded latents
    come off a ``LayerNorm`` and so live on that shell already; emitting onto it
    hands the decoder vectors at the scale it was trained on rather than
    whatever magnitude the proposer happens to settle at.
    """

    def __init__(self, embedding_dim: int, num_latent_tokens: int, num_heads: int):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_latent_tokens, embedding_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.context_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=READOUT_DROPOUT,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.latent_scale = math.sqrt(embedding_dim)

    def forward(self, context: Tensor) -> Tensor:
        if context.ndim != 3:
            raise ValueError("context must have shape [B, T, D]")
        batch_size = context.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        context = self.context_norm(context)
        attended, _ = self.attention(
            self.query_norm(queries),
            context,
            context,
            need_weights=False,
        )
        latent = self.output_norm(queries + attended)
        normalized = latent / latent.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return normalized * self.latent_scale


@dataclass(frozen=True)
class CanonicalTargets:
    """The label tensors a batch of graphs contributes, stacked onto a device."""

    node_types: Tensor  # [B, N] canonical type ids
    parent_positions: Tensor  # [B, N, S] true parent canonical positions
    slot_mask: Tensor  # [B, N, S] which slots actually exist


def canonical_targets(
    graphs: Sequence[FixedInDegreeDAGDescription],
    device: torch.device,
) -> CanonicalTargets:
    """Stack each graph's cached canonical label tensors into device batches."""
    graphs = list(graphs)
    return CanonicalTargets(
        node_types=torch.stack([graph.canonical_node_types for graph in graphs]).to(
            device, non_blocking=True
        ),
        parent_positions=torch.stack(
            [graph.canonical_parent_positions for graph in graphs]
        ).to(device, non_blocking=True),
        slot_mask=torch.stack(
            [graph.canonical_parent_slot_mask for graph in graphs]
        ).to(device, non_blocking=True),
    )


@dataclass(frozen=True)
class ReconstructionMetrics:
    """What a proposed latent scored against the graph it should have described."""

    type_loss: Tensor
    pointer_loss: Tensor
    type_accuracy: Tensor
    pointer_accuracy: Tensor

    def total(self, type_weight: float, pointer_weight: float) -> Tensor:
        return type_weight * self.type_loss + pointer_weight * self.pointer_loss

    def scalars(self, prefix: str) -> dict[str, float]:
        """Flat ``{"<prefix>/<name>": value}`` floats, ready for TensorBoard.

        Detached, so logging can never hold a reference to the graph.
        """
        return {
            f"{prefix}/type_loss": self.type_loss.item(),
            f"{prefix}/pointer_loss": self.pointer_loss.item(),
            f"{prefix}/type_accuracy": self.type_accuracy.item(),
            f"{prefix}/pointer_accuracy": self.pointer_accuracy.item(),
        }


def reconstruction_losses(
    model: DagnabbitAutoEncoder,
    latent: Tensor,
    targets: CanonicalTargets,
) -> ReconstructionMetrics:
    """Decode a proposed latent and score it against the true graphs.

    Mirrors :meth:`DagnabbitAutoEncoder.training_forward_batch` from the decode
    step onward, but takes the latent as given rather than encoding a graph to
    produce it -- which is the whole difference between training an autoencoder
    and training a proposer.
    """
    reconstructed = model.decode_latent(latent)
    batch_size = latent.shape[0]

    trunk_labels = targets.node_types[:, model.num_root_nodes : model.output_start]
    type_logits = model.node_type_predictor(
        reconstructed[:, model.num_root_nodes : model.output_start]
    )
    type_loss = F.cross_entropy(
        type_logits.reshape(-1, model.num_trunk_node_types),
        trunk_labels.reshape(-1),
    )
    type_accuracy = (type_logits.argmax(dim=-1) == trunk_labels).float().mean()

    pointer_logits = model.parent_pointer_logits(reconstructed)
    # Invalid slots can have fully -inf candidate rows; zero them so the
    # cross-entropy stays finite, then mask their contribution out.
    safe_logits = torch.where(
        targets.slot_mask.unsqueeze(-1),
        pointer_logits,
        torch.zeros_like(pointer_logits),
    )
    pointer_losses = F.cross_entropy(
        safe_logits.reshape(-1, model.num_nodes),
        targets.parent_positions.reshape(-1),
        reduction="none",
    ).reshape(batch_size, model.num_nodes, model.maximum_indegree)
    valid = targets.slot_mask.to(pointer_losses.dtype)
    pointer_loss = (pointer_losses * valid).sum() / valid.sum().clamp(min=1.0)

    correct = (
        pointer_logits.argmax(dim=-1) == targets.parent_positions
    ) & targets.slot_mask
    pointer_accuracy = correct.sum() / targets.slot_mask.sum().clamp(min=1)

    return ReconstructionMetrics(
        type_loss=type_loss,
        pointer_loss=pointer_loss,
        type_accuracy=type_accuracy,
        pointer_accuracy=pointer_accuracy,
    )
