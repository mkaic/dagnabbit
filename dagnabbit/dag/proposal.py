"""Task-agnostic machinery for training something to *write* a graph latent.

The autoencoder's latent is an interchange format: ``decode_latent`` turns any
``[K, D]`` tensor into a structurally valid DAG. A **proposer** is anything that
produces such a tensor from a task's specification -- a truth table, a set of
I/O examples, a target trajectory. What the specification looks like, and how
it is read, is the task's business. What happens *after* the latent exists is
not, and that is what lives here:

* :func:`project_to_shell` -- the scaling every latent must pass through before
  the decoder sees it.
* :class:`LatentReadout` -- the final stage a *deterministic* proposer shares,
  turning a variable number of context tokens into the fixed ``[K, D]`` latent
  shape.
* :func:`reconstruction_losses` -- scoring a proposed latent against the graph
  it was supposed to describe, reusing the autoencoder's own type and pointer
  cross-entropy.

A deterministic proposer is only one of the two ways to write a latent, and it
is the weaker one: behaviour -> graph is massively one-to-many, so the
cross-entropy minimizer is a per-slot marginal whose argmax need not be a
coherent graph. :mod:`dagnabbit.dag.flow` is the generative alternative, and it
reuses :func:`project_to_shell` from here rather than the readout.

Neither knows what a circuit is, and nothing in this module may import from
:mod:`dagnabbit.tasks`. A new task supplies its own specification encoder, ends
it with :class:`LatentReadout`, and trains it with
:func:`reconstruction_losses`.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import FixedInDegreeDAGDescription

READOUT_DROPOUT = 0.0


def project_to_shell(latent: Tensor, radius: float | Tensor | None = None) -> Tensor:
    """Rescale every latent token onto a shell of the given radius.

    Anything handed to ``decode_latent`` should sit at the magnitude the decoder
    was trained at -- an off-shell vector is a scale it has never seen, and
    because ``decode_latent`` adds its position encoding *before* the first
    normalization, a wrong magnitude quietly reweights latent against position.

    ``radius`` defaults to ``sqrt(D)``, which is where encoded latents would sit
    if the encoder's final ``LayerNorm`` had no learned gain. It does have one,
    so the true radius is a property of the trained checkpoint and is measurably
    not ``sqrt(D)`` -- on the d128 checkpoint it is 11.99 against a ``sqrt(D)``
    of 11.31. Pass the measured value when you have it;
    :class:`~dagnabbit.dag.flow.LatentNormalizer` fits it alongside its
    per-dimension statistics for exactly this reason.
    """
    if radius is None:
        radius = latent.shape[-1] ** 0.5
    norms = latent.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return latent * (radius / norms)


class LatentReadout(nn.Module):
    """Context tokens ``[B, T, C]`` -> graph latent ``[B, K, D]``.

    ``K`` learned queries cross-attend over however many context tokens the
    specification encoder produced, so the latent shape is decoupled from the
    specification's size: a task with 256 patch tokens and one with 40 example
    tokens both land on the same ``[K, D]``.

    Attention runs at the encoder's width ``C``, and a final projection maps to
    the autoencoder's latent width ``D``. Those two are deliberately separate.
    How much capacity it takes to *read* a specification has nothing to do with
    how many dimensions the autoencoder happens to need to *describe a graph*,
    and tying them would cap the encoder's width at whatever the latent budget
    is -- which is exactly backwards, since the point of a tight latent is that
    it is tight.

    The output is projected onto a shell of radius ``sqrt(D)``. Encoded latents
    come off a ``LayerNorm`` and so live on that shell already; emitting onto it
    hands the decoder vectors at the scale it was trained on rather than
    whatever magnitude the proposer happens to settle at.
    """

    def __init__(
        self,
        context_dim: int,
        latent_dim: int,
        num_latent_tokens: int,
        num_heads: int,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_latent_tokens, context_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.query_norm = nn.LayerNorm(context_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.attention = nn.MultiheadAttention(
            context_dim,
            num_heads,
            dropout=READOUT_DROPOUT,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(context_dim)
        self.to_latent = nn.Linear(context_dim, latent_dim)

    def forward(self, context: Tensor) -> Tensor:
        if context.ndim != 3:
            raise ValueError("context must have shape [B, T, C]")
        batch_size = context.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        context = self.context_norm(context)
        attended, _ = self.attention(
            self.query_norm(queries),
            context,
            context,
            need_weights=False,
        )
        return project_to_shell(self.to_latent(self.output_norm(queries + attended)))


def choice_signatures(
    model: DagnabbitAutoEncoder,
    trunk_types: Tensor,
    parent_choices: Tensor,
) -> Tensor:
    """Decoded graphs as comparable integer rows: ``[B, T + N * S]``.

    Two graphs are the same graph iff their signatures match. Slots beyond a
    position's in-degree are set to -1 before comparison, because a slot that
    does not exist in the decoded circuit cannot be a difference between two
    circuits -- comparing them raw would report structural diversity that is
    pure padding noise.

    Kept in tensor form on purpose. The alternatives are
    :func:`~dagnabbit.dag.description.graphs_match`, which is exact but costs a
    Python pass per *pair*, and nothing -- and counting distinct samples among a
    few hundred candidates is precisely a place where a per-pair Python loop is
    the whole cost of the measurement.
    """
    in_degrees = torch.tensor(model.trunk_node_in_degrees, device=trunk_types.device)
    active = torch.zeros(
        trunk_types.shape[0],
        model.num_nodes,
        dtype=torch.long,
        device=trunk_types.device,
    )
    active[:, model.num_root_nodes : model.output_start] = in_degrees[trunk_types]
    active[:, model.output_start :] = 1

    slot_index = torch.arange(model.maximum_indegree, device=trunk_types.device)
    slot_mask = slot_index.view(1, 1, -1) < active.unsqueeze(-1)
    masked_parents = parent_choices.masked_fill(~slot_mask, -1)
    return torch.cat([trunk_types, masked_parents.flatten(start_dim=1)], dim=1)


def count_distinct_signatures(signatures: Tensor, group_size: int) -> Tensor:
    """Distinct graphs per group of ``group_size`` consecutive rows -> ``[P]``.

    ``signatures`` is ``[P * group_size, L]`` laid out group-major, matching what
    ``repeat_interleave`` on the conditioning produces. Reported as a fraction of
    ``group_size`` by callers: near 1 means the sampler is exploring, near
    1/group_size means every draw collapsed to the same circuit and best-of-N is
    buying nothing.
    """
    if signatures.shape[0] % group_size != 0:
        raise ValueError(
            f"{signatures.shape[0]} signatures do not divide into groups of "
            f"{group_size}"
        )
    grouped = signatures.reshape(-1, group_size, signatures.shape[-1]).cpu()
    return torch.tensor(
        [len({tuple(row.tolist()) for row in group}) for group in grouped],
        dtype=torch.long,
    )


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
