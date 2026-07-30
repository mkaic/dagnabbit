"""Reading a circuit's behaviour and proposing a latent that reproduces it.

This is the logic-gates half of the proposal setup: everything that knows a
specification is a *truth table*. The task-agnostic halves live in
:mod:`dagnabbit.dag.proposal` (the readout and the reconstruction loss) and
:mod:`dagnabbit.dag.flow` (the velocity model and the sampler).

The encoder is deliberately dull. A truth-table image (see
:mod:`.truth_table_image`) is patchified, given a learned position embedding,
and run through plain transformer blocks. Position in the image *is* the input,
so no separate addressing scheme is needed.

Two proposers are built on that shared :class:`BehaviourEncoder`:

* :class:`TruthTableProposer` -- deterministic. Pools the context tokens into
  one latent and is trained with the frozen decoder's own cross-entropy. The
  baseline.
* :class:`TruthTableFlowProposer` -- generative. Cross-attends into the same
  context tokens once per Euler step, and models the whole *distribution* of
  latents consistent with a behaviour. This exists because the deterministic
  one has a ceiling that is not about capacity: behaviour -> graph is
  one-to-many, so the cross-entropy minimizer is a per-slot marginal whose
  argmax need not be a coherent circuit.

What makes any of this task-specific is only that behaviour happens to be
renderable as a 2D grid. A task whose specification is a set of I/O examples
would swap the encoder for a set encoder and reuse everything else.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder, TransformerBlock
from dagnabbit.dag.description import FixedInDegreeDAGDescription, graphs_match
from dagnabbit.dag.flow import (
    FlowMatchingLoss,
    LatentNormalizer,
    LatentVelocityModel,
    flow_matching_loss,
    sample_latents,
)
from dagnabbit.dag.proposal import (
    LatentReadout,
    canonical_targets,
    choice_signatures,
    count_distinct_signatures,
    reconstruction_losses,
)
from dagnabbit.tasks.logic_gates.evaluate import (
    BitpackedTask,
    bit_accuracy,
    evaluate_choices,
    evaluate_graphs,
)
from dagnabbit.tasks.logic_gates.rewards import (
    behaviour_match_score,
    constant_output_fraction,
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


class BehaviourEncoder(nn.Module):
    """Truth-table image ``[B, C, H, W]`` -> context tokens ``[B, T, E]``.

    The plain ViT trunk, stopping short of any readout. Two consumers want
    exactly this and disagree about what comes next: the deterministic
    :class:`TruthTableProposer` pools the tokens into a single latent, while the
    flow-matching proposer cross-attends into them once per Euler step. Keeping
    the trunk separate is what lets the second reuse the first's architecture
    (and, if wanted, its weights) unchanged.

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
        self.num_heads = embedding_dim // PROPOSER_ATTENTION_HEAD_DIM
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
                    num_heads=self.num_heads,
                    transformer_mlp_depth=PROPOSER_MLP_DEPTH,
                    mlp_expansion_factor=mlp_expansion_factor,
                    dropout=PROPOSER_DROPOUT,
                )
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(embedding_dim)

    @classmethod
    def for_task(
        cls,
        task: BitpackedTask,
        patch_size: int,
        embedding_dim: int,
        num_layers: int,
        mlp_expansion_factor: float,
    ) -> "BehaviourEncoder":
        """Build an encoder whose input shape matches a task's truth table."""
        height, width = image_dimensions(task.root_values.shape[0])
        return cls(
            num_output_bits=task.target_values.shape[0],
            image_height=height,
            image_width=width,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
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
        return self.encoder_norm(tokens)


class TruthTableProposer(nn.Module):
    """Truth-table image ``[B, C, H, W]`` -> graph latent ``[B, K, D]``.

    The deterministic baseline, kept as the thing a generative proposer has to
    beat. Its ceiling is not capacity: behaviour -> graph is one-to-many, so the
    cross-entropy minimizer here is a per-slot marginal whose argmax need not be
    a coherent circuit. See :mod:`dagnabbit.dag.flow`.
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
        self.encoder = BehaviourEncoder(
            num_output_bits=num_output_bits,
            image_height=image_height,
            image_width=image_width,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            mlp_expansion_factor=mlp_expansion_factor,
        )
        self.readout = LatentReadout(
            context_dim=embedding_dim,
            latent_dim=latent_dim,
            num_latent_tokens=num_latent_tokens,
            num_heads=self.encoder.num_heads,
        )

    @property
    def num_patches(self) -> int:
        return self.encoder.num_patches

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
        return self.readout(self.encoder(images))


class TruthTableFlowProposer(nn.Module):
    """The generative proposer: truth table -> *distribution* over graph latents.

    Wraps the two halves that have to travel together -- the behaviour encoder
    and the velocity model that cross-attends into it -- plus the latent
    statistics the sampler needs. One module so one checkpoint carries
    everything, including the fitted normalizer.

    The asymmetry is the point and is worth keeping in view: the encoder is the
    larger model and runs *once* per specification, while the velocity model is
    smaller and runs once per Euler step. So the condition tokens are computed
    up front and reused, and ``sample`` takes images rather than tokens only to
    keep that ordering impossible to get wrong.
    """

    def __init__(
        self,
        encoder: BehaviourEncoder,
        velocity_model: LatentVelocityModel,
        normalizer: LatentNormalizer,
    ):
        super().__init__()
        if velocity_model.condition_dim != encoder.embedding_dim:
            raise ValueError(
                "velocity model's condition_dim "
                f"({velocity_model.condition_dim}) must match the encoder's "
                f"width ({encoder.embedding_dim})"
            )
        self.encoder = encoder
        self.velocity_model = velocity_model
        self.normalizer = normalizer

    @classmethod
    def for_task(
        cls,
        task: BitpackedTask,
        model: DagnabbitAutoEncoder,
        patch_size: int = 16,
        embedding_dim: int = 512,
        encoder_num_layers: int = 8,
        velocity_num_layers: int = 4,
        mlp_expansion_factor: float = 4.0,
    ) -> "TruthTableFlowProposer":
        """Build a flow proposer matching a task and a frozen autoencoder.

        ``velocity_num_layers`` defaults to 4 against the encoder's 8 on
        purpose: the velocity model sees only ``num_output_nodes`` tokens, so
        every one of its kernels is launch-bound rather than FLOP-bound and
        depth costs serial latency once per sampler step for nothing width
        cannot buy more cheaply. Its width is not a parameter at all -- it is
        the encoder's, so cross-attention reads the condition tokens without
        reprojecting them.
        """
        encoder = BehaviourEncoder.for_task(
            task=task,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_layers=encoder_num_layers,
            mlp_expansion_factor=mlp_expansion_factor,
        )
        velocity_model = LatentVelocityModel(
            latent_dim=model.node_embedding_dim,
            num_latent_tokens=model.num_output_nodes,
            condition_dim=embedding_dim,
            model_width=embedding_dim,
            num_layers=velocity_num_layers,
        )
        return cls(
            encoder=encoder,
            velocity_model=velocity_model,
            normalizer=LatentNormalizer(model.node_embedding_dim),
        )

    @property
    def num_patches(self) -> int:
        return self.encoder.num_patches

    def forward(
        self,
        images: Tensor,
        clean_latent: Tensor,
        condition_dropout: float = 0.1,
        generator: torch.Generator | None = None,
    ) -> FlowMatchingLoss:
        """One training step's loss, from raw images and *unnormalized* latents.

        Normalization happens here rather than in the caller so there is exactly
        one place that decides which space the loss lives in.
        """
        condition_tokens = self.velocity_model.drop_condition(
            self.encoder(images),
            condition_dropout,
            generator=generator,
        )
        return flow_matching_loss(
            self.velocity_model,
            self.normalizer.normalize(clean_latent),
            condition_tokens,
            generator=generator,
        )

    @torch.no_grad()
    def sample(
        self,
        images: Tensor,
        num_candidates: int = 1,
        num_steps: int = 32,
        guidance_strength: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw ``[B * num_candidates, K, D]`` decoder-ready latents.

        Candidates are laid out group-major -- all of image 0's candidates, then
        all of image 1's -- matching ``repeat_interleave``, so a
        ``[B, num_candidates]`` view of any per-candidate score reshapes without
        a permute.

        The encoder runs once for all candidates of an image; only the velocity
        model is repeated. That is the whole reason best-of-256 costs barely
        more than best-of-1 here.
        """
        condition_tokens = self.encoder(images)
        if num_candidates > 1:
            condition_tokens = condition_tokens.repeat_interleave(num_candidates, dim=0)
        return sample_latents(
            self.velocity_model,
            condition_tokens,
            self.normalizer,
            num_steps=num_steps,
            guidance_strength=guidance_strength,
            generator=generator,
        )


@torch.no_grad()
def evaluate_flow_proposals(
    model: DagnabbitAutoEncoder,
    proposer: TruthTableFlowProposer,
    images: Tensor,
    targets: Tensor,
    task: BitpackedTask,
    device: torch.device,
    num_candidates: int = 16,
    num_steps: int = 32,
    guidance_strength: float = 1.0,
    source_graphs: Sequence[FixedInDegreeDAGDescription] | None = None,
) -> dict[str, float]:
    """Draw ``num_candidates`` circuits per specification and score the lot.

    ``images`` is ``[P, C, H, W]`` and ``targets`` is ``[P, C, W]`` packed goal
    behaviour -- a graph's own behaviour for in-distribution evaluation, or a
    task's ``target_values`` for a structured reference. Both cases are the same
    call, which is the point of scoring behaviour rather than structure.

    What to read:

    * ``best_of_n_correlation`` is the headline. It must beat the deterministic
      proposer and it must *rise with* ``num_candidates``.
    * ``distinct_fraction`` is why. If it collapses toward ``1/num_candidates``
      the sampler is not exploring and best-of-N is buying nothing; suspect the
      guidance strength before anything else.
    * ``mean_correlation`` next to the best says whether the whole distribution
      is good or whether one lucky draw is carrying it.
    * ``constant_output_fraction`` guards the degenerate answer. If it climbs
      while accuracy climbs and correlation does not, the constant-output hack
      is back -- see :mod:`dagnabbit.tasks.logic_gates.rewards`.

    ``source_graphs`` is optional and only meaningful in-distribution: it adds
    the fraction of specifications where *some* candidate recovered the exact
    original circuit. That is a bonus, not the objective -- many different
    circuits are correct answers, which is the entire premise here.
    """
    was_training = proposer.training
    proposer.eval()
    try:
        num_specifications = images.shape[0]
        latents = proposer.sample(
            images.to(device),
            num_candidates=num_candidates,
            num_steps=num_steps,
            guidance_strength=guidance_strength,
        )
        trunk_types, parent_choices = model.generate_choices(latents)

        target_indices = torch.arange(
            num_specifications, device=device
        ).repeat_interleave(num_candidates)
        scores = {
            name: behaviour_match_score(
                trunk_types,
                parent_choices,
                target_indices,
                targets.to(device),
                task,
                model.trunk_node_in_degrees,
                statistic=name,
            )
            .mean(dim=-1)
            .reshape(num_specifications, num_candidates)
            for name in ("correlation", "accuracy")
        }

        behaviours = evaluate_choices(
            trunk_types, parent_choices, task, model.trunk_node_in_degrees
        )
        signatures = choice_signatures(model, trunk_types, parent_choices)
        distinct = count_distinct_signatures(signatures, num_candidates)

        results = {
            "best_of_n_correlation": float(
                scores["correlation"].max(dim=1).values.mean()
            ),
            "mean_correlation": float(scores["correlation"].mean()),
            "best_of_n_accuracy": float(scores["accuracy"].max(dim=1).values.mean()),
            "mean_accuracy": float(scores["accuracy"].mean()),
            "distinct_fraction": float(distinct.double().mean() / num_candidates),
            "constant_output_fraction": constant_output_fraction(behaviours, task),
        }

        if source_graphs is not None:
            rebuilt = model.descriptions_from_choices(trunk_types, parent_choices)
            recovered = sum(
                any(
                    graphs_match(
                        source_graphs[index],
                        rebuilt[index * num_candidates + candidate],
                    )
                    for candidate in range(num_candidates)
                )
                for index in range(num_specifications)
            )
            results["exact_match_any"] = recovered / num_specifications
    finally:
        if was_training:
            proposer.train()

    return results


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
