"""Reading a circuit's behaviour and proposing a latent that reproduces it.

This is the logic-gates half of the proposal setup: everything that knows a
specification is a *truth table*. The task-agnostic half -- the readout that
emits the latent, and the loss that scores it -- lives in
:mod:`dagnabbit.dag.proposal`.

The architecture is deliberately dull. A truth-table image (see
:mod:`.truth_table_image`) is patchified, given a learned position embedding,
run through plain transformer blocks, and read out into the ``[K, D]`` latent
the frozen decoder consumes. Position in the image *is* the input, so no
separate addressing scheme is needed.

What makes this task-specific is only that behaviour happens to be renderable
as a 2D grid. A task whose specification is a set of I/O examples would swap
this module for a set encoder, keep
:class:`~dagnabbit.dag.proposal.LatentReadout`, and reuse the same training
script unchanged.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder, TransformerBlock
from dagnabbit.dag.description import FixedInDegreeDAGDescription, graphs_match
from dagnabbit.dag.proposal import (
    LatentReadout,
    canonical_targets,
    reconstruction_losses,
)
from dagnabbit.tasks.logic_gates.evaluate import (
    BitpackedTask,
    bit_accuracy,
    evaluate_graphs,
)
from dagnabbit.tasks.logic_gates.truth_table_image import (
    image_dimensions,
    outputs_to_image,
)

PROPOSER_ATTENTION_HEAD_DIM = 64
PROPOSER_MLP_DEPTH = 1
PROPOSER_DROPOUT = 0.0


def behaviour_images(
    graphs: Sequence[FixedInDegreeDAGDescription],
    task: BitpackedTask,
    gray: bool = False,
) -> Tensor:
    """What each graph actually computes, as a ``[B, C, H, W]`` float image.

    The conditioning input during training. Note this is a graph's *own*
    behaviour, not its fitness against ``task`` -- the task supplies the input
    columns to evaluate against and nothing else. That is what makes the labels
    free: every graph is a perfect example of computing its own function.
    """
    height, width = image_dimensions(task.root_values.shape[0])
    packed = evaluate_graphs(list(graphs), task)
    return outputs_to_image(packed, height, width, gray=gray).float()


class TruthTableProposer(nn.Module):
    """Truth-table image ``[B, C, H, W]`` -> graph latent ``[B, K, D]``.

    ``patch_size`` must divide both image axes. With the 256x256 adder table
    and the default 16, that is 256 patch tokens -- an ordinary ViT sequence.
    """

    def __init__(
        self,
        num_output_bits: int,
        image_height: int,
        image_width: int,
        patch_size: int,
        embedding_dim: int,
        latent_dim: int,
        num_latent_tokens: int,
        num_layers: int,
        mlp_expansion_factor: float,
    ):
        super().__init__()
        if image_height % patch_size or image_width % patch_size:
            raise ValueError(
                f"patch_size {patch_size} must divide both image axes "
                f"({image_height}x{image_width})"
            )
        if embedding_dim % PROPOSER_ATTENTION_HEAD_DIM != 0:
            raise ValueError(
                "embedding_dim is the ViT's own width and must be a multiple of "
                f"the fixed {PROPOSER_ATTENTION_HEAD_DIM}-wide head dim; got "
                f"{embedding_dim}"
            )
        num_heads = embedding_dim // PROPOSER_ATTENTION_HEAD_DIM
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = (image_height // patch_size) * (image_width // patch_size)

        # The one layer that knows anything about the task: how many bit planes
        # a behaviour has, and how a patch of them becomes a token.
        self.patch_embedding = nn.Conv2d(
            num_output_bits,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.empty(self.num_patches, embedding_dim)
        )
        nn.init.normal_(self.position_embeddings, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    node_embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    transformer_mlp_depth=PROPOSER_MLP_DEPTH,
                    mlp_expansion_factor=mlp_expansion_factor,
                    dropout=PROPOSER_DROPOUT,
                )
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(embedding_dim)
        self.readout = LatentReadout(
            context_dim=embedding_dim,
            latent_dim=latent_dim,
            num_latent_tokens=num_latent_tokens,
            num_heads=num_heads,
        )

    @classmethod
    def for_task(
        cls,
        task: BitpackedTask,
        model: DagnabbitAutoEncoder,
        patch_size: int,
        embedding_dim: int,
        num_layers: int,
        mlp_expansion_factor: float,
    ) -> "TruthTableProposer":
        """Build a proposer whose shapes match a task and a frozen autoencoder.

        Keeps the "which dimension comes from where" wiring in one place rather
        than in every script that wants a proposer. Note what the autoencoder
        does *not* determine: ``embedding_dim`` is the ViT's own width and is
        the caller's to choose. Only the latent shape is dictated by the
        checkpoint.
        """
        height, width = image_dimensions(task.root_values.shape[0])
        return cls(
            num_output_bits=task.target_values.shape[0],
            image_height=height,
            image_width=width,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            latent_dim=model.node_embedding_dim,
            num_latent_tokens=model.num_output_nodes,
            num_layers=num_layers,
            mlp_expansion_factor=mlp_expansion_factor,
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")
        tokens = self.patch_embedding(images.to(self.position_embeddings.dtype))
        tokens = tokens.flatten(start_dim=2).transpose(1, 2)
        tokens = tokens + self.position_embeddings.unsqueeze(0)
        for block in self.blocks:
            tokens = block(tokens, None)
        return self.readout(self.encoder_norm(tokens))


@torch.no_grad()
def evaluate_proposals(
    model: DagnabbitAutoEncoder,
    proposer: TruthTableProposer,
    graphs: Sequence[FixedInDegreeDAGDescription],
    tasks: Sequence[BitpackedTask],
    images: Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Propose from behaviour, decode, and score against the true circuits.

    ``tasks[i]`` is the task ``graphs[i]``'s behaviour came from, so the fitness
    reported answers "does the proposed circuit compute the function it was
    asked for" rather than fitness against some unrelated objective.

    Structural metrics (type/pointer accuracy) say whether the right *graph*
    came back; fitness says whether the right *function* did. Those come apart
    badly -- most wires reproduced can still score at chance -- which is why
    both are reported.
    """
    graphs = list(graphs)
    was_training = proposer.training
    proposer.eval()
    try:
        latent = proposer(images.to(device))
        metrics = reconstruction_losses(
            model, latent, canonical_targets(graphs, device)
        )
        rebuilt = model.generate(latent)
        fitnesses = []
        exact = 0
        for index, (graph, task) in enumerate(zip(graphs, tasks)):
            score, _ = bit_accuracy(evaluate_graphs([rebuilt[index]], task), task)
            fitnesses.append(float(score[0]))
            exact += int(graphs_match(graph, rebuilt[index]))
    finally:
        if was_training:
            proposer.train()

    return {
        "type_loss": float(metrics.type_loss),
        "pointer_loss": float(metrics.pointer_loss),
        "type_accuracy": float(metrics.type_accuracy),
        "pointer_accuracy": float(metrics.pointer_accuracy),
        "fitness_mean": sum(fitnesses) / len(fitnesses),
        "fitness_best": max(fitnesses),
        "exact_match": exact / len(graphs),
    }
