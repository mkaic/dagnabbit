"""Conditional flow matching over graph latents.

How a specification becomes a graph. The autoencoder's latent is an interchange
format -- ``decode_latent`` turns any ``[K, D]`` tensor into a structurally valid
DAG -- so writing a latent is the whole job, and this module models a
*distribution* over the latents a specification admits rather than picking one.

Why a distribution and not a regression
---------------------------------------
Behaviour -> graph is massively one-to-many, and worse here than usual: dead
nodes and unselected producers mean whole families of structurally different
graphs have bit-identical behaviour. A network trained to minimize the
decoder's cross-entropy therefore converges on the per-slot *marginal* over
parents consistent with the specification, and the argmax of independent
marginals need not be a coherent graph at all. It can be perfectly calibrated
about every wire and still emit a circuit that computes nothing. That is why the
deterministic regressor this replaced is gone rather than kept as a baseline.

Sampling from a modelled distribution has no such failure: a draw is a
coherent joint configuration. And because a draw is cheap, the proposer stops
being a one-shot guess and becomes a *search distribution* -- draw N, score
them, keep the best.

What flow matching is
---------------------
Pick a real latent and a Gaussian noise sample of the same shape. Interpolate
linearly between them. Train a network to predict, from a point on that line
and how far along it is, the direction of travel -- which for a straight line
is ``clean - noise``, constant along the path. Sampling is Euler integration of
that direction field starting from pure noise.

That is the whole method. No noise schedule, no variance bookkeeping, no
per-noise-level loss weighting: plain unweighted mean-squared error is already
the correct objective under this parameterization.

Conventions this module fixes
-----------------------------
``blend`` runs 0 -> 1 with **0 meaning pure noise and 1 meaning pure data**.
Half the literature and half the codebases use the opposite convention and it
is the single most common source of sign errors, so it is stated here once and
never varied: :func:`flow_matching_loss` and :func:`sample_latents` agree.

Nothing here knows what a circuit is. The specification arrives already encoded
as a ``[B, T, C]`` token sequence, so a task supplies its own encoder and reuses
everything below.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

VELOCITY_ATTENTION_HEAD_DIM = 64
VELOCITY_MLP_EXPANSION_FACTOR = 4.0
# Width of the blend-fraction embedding that drives every block's modulation.
# Deliberately much narrower than the residual stream: modulation is a
# per-block Linear onto 9 * model_width, so projecting from the full stream
# width would make it the largest parameter group in the model for no benefit.
BLEND_EMBEDDING_DIM = 128
# Sinusoidal features computed for the raw blend scalar before that embedding.
BLEND_NUM_FREQUENCIES = 128


def project_to_shell(latent: Tensor, radius: float | Tensor) -> Tensor:
    """Rescale every latent token onto a shell of the given radius.

    Anything handed to ``decode_latent`` should sit at the magnitude the decoder
    was trained at -- an off-shell vector is a scale it has never seen, and
    because ``decode_latent`` adds its position encoding *before* the first
    normalization, a wrong magnitude quietly reweights latent against position.

    ``radius`` is required rather than defaulting to ``sqrt(D)``, which is what
    an earlier version of this codebase assumed. ``sqrt(D)`` is where encoded
    latents would sit if the encoder's final ``LayerNorm`` had no learned gain.
    It has one, so the true radius is a property of the trained checkpoint and is
    measurably not ``sqrt(D)``: on the d128 checkpoint it is 11.99 against a
    ``sqrt(D)`` of 11.31. Nobody should be guessing it, so nobody may.
    :meth:`LatentNormalizer.fit` measures it.
    """
    norms = latent.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return latent * (radius / norms)


def blend_frequency_features(blend: Tensor, num_frequencies: int) -> Tensor:
    """``[B]`` blend fractions in [0, 1] -> ``[B, 2 * num_frequencies]``.

    The standard sinusoidal position encoding, applied to a continuous scalar
    rather than an integer index. A bare scalar is a poor conditioning signal
    for a network that has to behave very differently at 0.05 than at 0.95;
    a Fourier basis makes "how far along am I" linearly decodable at every
    scale the network might need it.
    """
    if blend.ndim != 1:
        raise ValueError(f"blend must have shape [B]; got {tuple(blend.shape)}")
    # Periods spanning roughly [2/1000, 2]: the low frequencies carry coarse
    # position, the high ones resolve the ends of the trip.
    frequencies = torch.exp(
        torch.arange(num_frequencies, dtype=torch.float32, device=blend.device)
        * (-math.log(1000.0) / num_frequencies)
    )
    angles = blend.float().unsqueeze(-1) * frequencies.unsqueeze(0) * math.pi
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class LatentNormalizer(nn.Module):
    """Per-dimension standardization of graph latents, fitted once and frozen.

    The sampler starts from a standard Gaussian, so the data it is asked to
    reach has to be at that scale. Encoded latents come off a ``LayerNorm`` and
    already sit on the ``sqrt(D)`` shell -- which is where a D-dimensional
    Gaussian concentrates too, so the mismatch is mild -- but ``LayerNorm``
    normalizes each token independently and guarantees nothing about how the
    *dataset* is distributed along any given dimension. Fitting explicit
    statistics costs one pass and removes the question.

    Stored as buffers so they travel with a checkpoint. Fitted from encoded
    latents by :meth:`fit`, then left alone: these are properties of the frozen
    autoencoder, not learned parameters.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("std", torch.ones(latent_dim))
        self.register_buffer("shell_radius", torch.tensor(float(latent_dim) ** 0.5))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, latents: Tensor, epsilon: float = 1e-4) -> "LatentNormalizer":
        """Fit from ``[N, K, D]`` (or ``[N, D]``) encoded latents.

        Statistics are pooled over every token position rather than kept
        per-position. The K latent tokens are the K output-node embeddings of
        one graph and share a representation space, so per-position statistics
        would fit sampling noise, not structure.

        The mean token norm is fitted too, because it is *not* ``sqrt(D)`` and
        assuming it is puts every sample at the wrong magnitude. Measured on the
        d128 checkpoint: real tokens sit at 11.9865 +/- 0.0349 while ``sqrt(D)``
        is 11.3137, the difference being the learned gain on the encoder's final
        ``LayerNorm``.

        That spread is worth noticing on its own. A Gaussian matched to these
        per-dimension moments has the right mean norm (11.9875 here, essentially
        exact) but a norm spread some twenty times wider, so the real latents
        occupy a far thinner shell than the noise the sampler starts from. The
        flow model can learn that thinness, but its terminal error cannot be
        expected to respect it, which is why the sampler ends with a projection
        rather than trusting denormalization alone.
        """
        if latents.ndim not in (2, 3):
            raise ValueError(
                f"latents must be [N, D] or [N, K, D]; got {tuple(latents.shape)}"
            )
        flattened = latents.reshape(-1, latents.shape[-1]).float()
        if flattened.shape[0] < 2:
            raise ValueError("need at least two latents to fit statistics")
        self.mean.copy_(flattened.mean(dim=0))
        # Clamped rather than epsilon-added: a genuinely dead dimension should
        # normalize to zero, not be amplified into noise by a tiny divisor.
        self.std.copy_(flattened.std(dim=0).clamp(min=epsilon))
        self.shell_radius.fill_(float(flattened.norm(dim=-1).mean()))
        self.fitted.fill_(True)
        return self

    def normalize(self, latents: Tensor) -> Tensor:
        return (latents - self.mean) / self.std

    def denormalize(self, latents: Tensor) -> Tensor:
        return latents * self.std + self.mean

    def to_decoder_space(self, latents: Tensor) -> Tensor:
        """Normalized samples -> what ``decode_latent`` expects to be handed.

        The one place that knows the full round trip: undo the per-dimension
        standardization, then land on the fitted shell. Both halves matter and
        neither subsumes the other -- denormalization fixes the *anisotropy*,
        the projection fixes the *radius*.
        """
        return project_to_shell(self.denormalize(latents), self.shell_radius)

    def require_fitted(self) -> None:
        if not bool(self.fitted):
            raise RuntimeError(
                "LatentNormalizer was never fitted; call fit() on encoded "
                "latents (or load a checkpoint that carries its statistics) "
                "before training or sampling"
            )


class ModulatedBlock(nn.Module):
    """One velocity-model block: self-attention, cross-attention, MLP.

    Every sub-layer is pre-norm with an affine-free ``LayerNorm``, scaled and
    shifted by parameters the blend fraction produces, and gated on the way out
    by a third set. All nine modulation vectors come from one ``Linear`` whose
    weights start at zero, so every gate starts at zero and the whole block
    starts as an exact no-op. The untrained model therefore predicts zero
    velocity everywhere, which makes the initial sampler the identity map
    instead of a noise injector -- the "AdaLN-Zero" trick from DiT, and worth
    more here than usual because there are only four blocks to get wrong.

    Cross-attention is where the specification enters, and it is the only
    expensive operation in the model: 8 queries against however many
    specification tokens there are.
    """

    def __init__(self, model_width: int, condition_dim: int, num_heads: int):
        super().__init__()
        hidden_dim = max(1, round(model_width * VELOCITY_MLP_EXPANSION_FACTOR))

        self.self_attention_norm = nn.LayerNorm(model_width, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(
            model_width, num_heads, batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(model_width, elementwise_affine=False)
        self.cross_attention = nn.MultiheadAttention(
            model_width,
            num_heads,
            batch_first=True,
            kdim=condition_dim,
            vdim=condition_dim,
        )
        self.feed_forward_norm = nn.LayerNorm(model_width, elementwise_affine=False)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_width, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_width),
        )

        # scale, shift and gate for each of the three sub-layers.
        self.modulation = nn.Linear(BLEND_EMBEDDING_DIM, 9 * model_width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        tokens: Tensor,
        blend_embedding: Tensor,
        condition_tokens: Tensor,
    ) -> Tensor:
        modulation = self.modulation(blend_embedding).unsqueeze(1).chunk(9, dim=-1)
        (
            self_scale,
            self_shift,
            self_gate,
            cross_scale,
            cross_shift,
            cross_gate,
            ff_scale,
            ff_shift,
            ff_gate,
        ) = modulation

        normed = self.self_attention_norm(tokens) * (1 + self_scale) + self_shift
        attended, _ = self.self_attention(normed, normed, normed, need_weights=False)
        tokens = tokens + self_gate * attended

        normed = self.cross_attention_norm(tokens) * (1 + cross_scale) + cross_shift
        attended, _ = self.cross_attention(
            normed,
            condition_tokens,
            condition_tokens,
            need_weights=False,
        )
        tokens = tokens + cross_gate * attended

        normed = self.feed_forward_norm(tokens) * (1 + ff_scale) + ff_shift
        return tokens + ff_gate * self.feed_forward(normed)


class LatentVelocityModel(nn.Module):
    """Noisy graph latent + blend fraction + specification tokens -> velocity.

    Wide and shallow on purpose, which inverts the usual instinct. The latent is
    only K tokens (8 for this repo's configuration), so an ``[8, width]`` matmul
    does not come close to filling a GPU and every kernel here is launch-bound
    rather than FLOP-bound. Depth then costs serial latency multiplied by the
    sampler's step count and buys nothing width cannot, while width is nearly
    free until the hardware is actually saturated. Four blocks at the
    specification encoder's own width is the shape that falls out -- matching
    widths also lets cross-attention read the condition tokens without
    reprojecting them.

    The latent itself stays narrow by construction; the 4x expansion happens in
    the residual stream around it, via the projections on the way in and out.
    """

    def __init__(
        self,
        latent_dim: int,
        num_latent_tokens: int,
        condition_dim: int,
        model_width: int,
        num_layers: int,
    ):
        super().__init__()
        if model_width % VELOCITY_ATTENTION_HEAD_DIM != 0:
            raise ValueError(
                "model_width must be a multiple of the fixed "
                f"{VELOCITY_ATTENTION_HEAD_DIM}-wide head dim; got {model_width}"
            )
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        num_heads = model_width // VELOCITY_ATTENTION_HEAD_DIM

        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        self.condition_dim = condition_dim
        self.model_width = model_width

        self.latent_in = nn.Linear(latent_dim, model_width)
        self.token_positions = nn.Parameter(
            torch.randn(num_latent_tokens, model_width) * 0.02
        )
        self.blend_embedding = nn.Sequential(
            nn.Linear(2 * BLEND_NUM_FREQUENCIES, BLEND_EMBEDDING_DIM),
            nn.SiLU(),
            nn.Linear(BLEND_EMBEDDING_DIM, BLEND_EMBEDDING_DIM),
        )
        # Learned stand-in for "no specification", so one model serves both the
        # conditional and unconditional roles classifier-free guidance needs.
        # Also makes the trained model usable as an unconditional sampler for
        # free, which is a diagnostic worth having.
        self.null_condition = nn.Parameter(torch.randn(1, 1, condition_dim) * 0.02)

        self.blocks = nn.ModuleList(
            [
                ModulatedBlock(model_width, condition_dim, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(model_width, elementwise_affine=False)
        self.velocity_out = nn.Linear(model_width, latent_dim)
        # Predict zero velocity until trained; see ModulatedBlock.
        nn.init.zeros_(self.velocity_out.weight)
        nn.init.zeros_(self.velocity_out.bias)

    def null_condition_for(self, batch_size: int) -> Tensor:
        """The unconditional specification, as a ``[B, 1, C]`` token sequence."""
        return self.null_condition.expand(batch_size, -1, -1)

    def drop_condition(
        self,
        condition_tokens: Tensor,
        dropout_probability: float,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Replace whole examples' conditions with the null token.

        Dropping *per example* rather than per token is the point: guidance
        needs a model that has seen "no specification at all", not one that has
        seen partially masked specifications.
        """
        if dropout_probability <= 0.0:
            return condition_tokens
        batch_size = condition_tokens.shape[0]
        dropped = (
            torch.rand(
                batch_size,
                device=condition_tokens.device,
                generator=generator,
            )
            < dropout_probability
        )
        null = self.null_condition.to(condition_tokens.dtype).expand(
            batch_size, condition_tokens.shape[1], -1
        )
        return torch.where(dropped[:, None, None], null, condition_tokens)

    def forward(
        self,
        noisy_latent: Tensor,
        blend: Tensor,
        condition_tokens: Tensor,
    ) -> Tensor:
        if noisy_latent.ndim != 3:
            raise ValueError(
                f"noisy_latent must be [B, K, D]; got {tuple(noisy_latent.shape)}"
            )
        if noisy_latent.shape[1:] != (self.num_latent_tokens, self.latent_dim):
            raise ValueError(
                f"noisy_latent must be [B, {self.num_latent_tokens}, "
                f"{self.latent_dim}]; got {tuple(noisy_latent.shape)}"
            )
        if blend.shape != (noisy_latent.shape[0],):
            raise ValueError(
                f"blend must be [B] matching the latent batch; got {tuple(blend.shape)}"
            )
        if condition_tokens.ndim != 3:
            raise ValueError(
                "condition_tokens must be [B, T, C]; got "
                f"{tuple(condition_tokens.shape)}"
            )

        tokens = self.latent_in(noisy_latent) + self.token_positions
        blend_embedding = self.blend_embedding(
            blend_frequency_features(blend, BLEND_NUM_FREQUENCIES).to(tokens.dtype)
        )
        for block in self.blocks:
            tokens = block(tokens, blend_embedding, condition_tokens)
        return self.velocity_out(self.output_norm(tokens))


def sample_blend_fractions(
    batch_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Tensor:
    """``[B]`` blend fractions, logit-normally distributed around 0.5.

    Where along the trip to supervise. Uniform sampling spends a third of its
    budget near the ends, which are nearly free to get right -- at blend 0 the
    answer barely depends on the input and at blend 1 the input is the answer.
    A logit-normal concentrates on the middle, where the direction field is
    genuinely ambiguous and the model actually has to learn something.

    Deliberately *not* shifted for dimensionality. The schedule shifts that
    high-resolution image models need exist because a large latent needs more
    noise at a given blend to actually destroy its information; at K*D on the
    order of a thousand dimensions that correction does not apply.
    """
    return torch.sigmoid(torch.randn(batch_size, device=device, generator=generator))


def interpolate_towards_data(
    clean_latent: Tensor,
    noise: Tensor,
    blend: Tensor,
) -> tuple[Tensor, Tensor]:
    """``(midpoint, true_velocity)`` for a straight noise -> data path.

    The one place the sign convention this module fixes actually lives, split
    out from :func:`flow_matching_loss` so it can be tested on its own: at
    ``blend == 0`` the midpoint is exactly the noise, at ``blend == 1`` it is
    exactly the data, and the velocity is the whole displacement between them
    regardless of where along the path you stand. Get this backwards and the
    model trains to a plausible-looking loss curve while sampling walks the
    wrong way.
    """
    if blend.shape != (clean_latent.shape[0],):
        raise ValueError(
            f"blend must be [B] matching the latent batch; got {tuple(blend.shape)}"
        )
    blend_broadcast = blend.reshape(-1, *([1] * (clean_latent.ndim - 1))).to(
        clean_latent.dtype
    )
    midpoint = blend_broadcast * clean_latent + (1 - blend_broadcast) * noise
    return midpoint, clean_latent - noise


@dataclass(frozen=True)
class FlowMatchingLoss:
    """The loss and the diagnostics worth logging beside it."""

    loss: Tensor
    # Mean squared error split by which half of the trip the example came from.
    # Reported because the total is dominated by irreducible noise variance and
    # moves very little; the noisy half is where real learning shows up, and a
    # model that is only improving on the easy half is stuck.
    noisy_half_loss: Tensor
    clean_half_loss: Tensor

    def scalars(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}/loss": self.loss.item(),
            f"{prefix}/loss_noisy_half": self.noisy_half_loss.item(),
            f"{prefix}/loss_clean_half": self.clean_half_loss.item(),
        }


def flow_matching_loss(
    velocity_model: LatentVelocityModel,
    clean_latent: Tensor,
    condition_tokens: Tensor,
    generator: torch.Generator | None = None,
) -> FlowMatchingLoss:
    """One flow-matching training step's loss on already-normalized latents.

    ``clean_latent`` is ``[B, K, D]`` and must already be in normalized space
    (see :class:`LatentNormalizer`); ``condition_tokens`` is ``[B, T, C]`` with
    condition dropout already applied by the caller, since whether to drop is a
    training-schedule decision rather than a loss one.

    Straight-line interpolation, and the target is the whole line as a single
    vector. Note there is no loss weighting anywhere: under this
    parameterization unweighted mean-squared error is already correct, which is
    most of why the field moved to it.
    """
    if clean_latent.ndim != 3:
        raise ValueError(
            f"clean_latent must be [B, K, D]; got {tuple(clean_latent.shape)}"
        )
    batch_size = clean_latent.shape[0]
    noise = torch.empty_like(clean_latent).normal_(generator=generator)
    blend = sample_blend_fractions(batch_size, clean_latent.device, generator)
    midpoint, true_velocity = interpolate_towards_data(clean_latent, noise, blend)

    predicted_velocity = velocity_model(midpoint, blend, condition_tokens)
    squared_error = (predicted_velocity - true_velocity).pow(2).mean(dim=(1, 2))

    noisy_half = blend < 0.5

    # A batch can land entirely on one side; an empty mean is NaN, so fall back
    # to zero rather than poisoning the logs.
    def half_mean(mask: Tensor) -> Tensor:
        if not bool(mask.any()):
            return torch.zeros(
                (), device=squared_error.device, dtype=squared_error.dtype
            )
        return squared_error[mask].mean()

    return FlowMatchingLoss(
        loss=squared_error.mean(),
        noisy_half_loss=half_mean(noisy_half).detach(),
        clean_half_loss=half_mean(~noisy_half).detach(),
    )


@torch.no_grad()
def sample_latents(
    velocity_model: LatentVelocityModel,
    condition_tokens: Tensor,
    normalizer: LatentNormalizer,
    num_steps: int = 32,
    guidance_strength: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw ``[B, K, D]`` decoder-ready latents by Euler integration.

    ``condition_tokens`` is ``[B, T, C]``. The returned latents are already in
    decoder space -- denormalized and on the normalizer's fitted shell -- so they
    can go straight into ``decode_latent``.

    ``guidance_strength`` of 1.0 disables classifier-free guidance and halves
    the cost, since the unconditional pass is then never needed. Guidance above
    1 samples a deliberately sharpened distribution: it trades coverage for
    fidelity, which is the wrong trade when the caller intends to draw many
    candidates and keep the best. Start at 1.0 and sweep upward only if
    single-draw quality matters more than best-of-N.

    The conditional and unconditional passes are batched into one forward call
    rather than run in sequence -- at these tensor sizes two launches cost
    roughly twice one, whatever the arithmetic says.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    normalizer.require_fitted()

    batch_size = condition_tokens.shape[0]
    device = condition_tokens.device
    latent = torch.empty(
        batch_size,
        velocity_model.num_latent_tokens,
        velocity_model.latent_dim,
        device=device,
        dtype=condition_tokens.dtype,
    ).normal_(generator=generator)

    guided = guidance_strength != 1.0
    if guided:
        null_tokens = velocity_model.null_condition.to(condition_tokens.dtype).expand(
            batch_size, condition_tokens.shape[1], -1
        )
        paired_condition = torch.cat([condition_tokens, null_tokens], dim=0)

    step_size = 1.0 / num_steps
    for step in range(num_steps):
        blend = torch.full((batch_size,), step * step_size, device=device)
        if guided:
            velocity = velocity_model(
                torch.cat([latent, latent], dim=0),
                torch.cat([blend, blend], dim=0),
                paired_condition,
            )
            conditional, unconditional = velocity.chunk(2, dim=0)
            velocity = unconditional + guidance_strength * (conditional - unconditional)
        else:
            velocity = velocity_model(latent, blend, condition_tokens)
        latent = latent + step_size * velocity

    return normalizer.to_decoder_space(latent)


@torch.no_grad()
def sample_latents_unconditional(
    velocity_model: LatentVelocityModel,
    batch_size: int,
    normalizer: LatentNormalizer,
    device: torch.device | str,
    num_steps: int = 32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample with the null specification -- a free diagnostic, not a mode.

    Training with condition dropout means the model is *also* an unconditional
    generator, at no extra cost. Decoding these samples answers "does the model
    know what a graph looks like at all", separately from any question about
    whether it read the specification correctly. Reach for it when conditional
    samples disappoint.
    """
    null_tokens = velocity_model.null_condition_for(batch_size).to(device)
    return sample_latents(
        velocity_model,
        null_tokens,
        normalizer,
        num_steps=num_steps,
        guidance_strength=1.0,
        generator=generator,
    )
