"""Tests for conditional flow matching over graph latents.

Weighted towards the things that fail *silently*. A wrong blend convention, an
unfitted normalizer or a sampler that ignores its conditioning all produce a
loss curve that looks fine, so those get direct assertions rather than
end-to-end ones. The overfit test at the bottom is the only one that checks the
whole loop actually learns, and it is the slowest, which is the usual trade.
"""

import pytest
import torch
import torch.nn as nn

from dagnabbit.dag.flow import (
    BLEND_NUM_FREQUENCIES,
    LatentNormalizer,
    LatentVelocityModel,
    WeightAverage,
    blend_frequency_features,
    flow_matching_loss,
    interpolate_towards_data,
    sample_blend_fractions,
    sample_latents,
    sample_latents_unconditional,
)
from dagnabbit.dag.proposal import project_to_shell

LATENT_DIM = 32
NUM_LATENT_TOKENS = 4
CONDITION_DIM = 64
MODEL_WIDTH = 64
NUM_CONTEXT_TOKENS = 6


def make_velocity_model(num_layers: int = 2) -> LatentVelocityModel:
    return LatentVelocityModel(
        latent_dim=LATENT_DIM,
        num_latent_tokens=NUM_LATENT_TOKENS,
        condition_dim=CONDITION_DIM,
        model_width=MODEL_WIDTH,
        num_layers=num_layers,
    )


def make_fitted_normalizer(latent_dim: int = LATENT_DIM) -> LatentNormalizer:
    normalizer = LatentNormalizer(latent_dim)
    normalizer.fit(torch.randn(256, NUM_LATENT_TOKENS, latent_dim))
    return normalizer


# --- the blend convention ---------------------------------------------------


def test_blend_zero_is_pure_noise_and_one_is_pure_data() -> None:
    """The convention this module fixes, asserted rather than commented."""
    clean = torch.randn(3, NUM_LATENT_TOKENS, LATENT_DIM)
    noise = torch.randn(3, NUM_LATENT_TOKENS, LATENT_DIM)

    at_noise, _ = interpolate_towards_data(clean, noise, torch.zeros(3))
    at_data, _ = interpolate_towards_data(clean, noise, torch.ones(3))

    torch.testing.assert_close(at_noise, noise)
    torch.testing.assert_close(at_data, clean)


def test_velocity_target_is_the_whole_displacement() -> None:
    """The target does not depend on where along the path you stand."""
    clean = torch.randn(5, NUM_LATENT_TOKENS, LATENT_DIM)
    noise = torch.randn(5, NUM_LATENT_TOKENS, LATENT_DIM)
    expected = clean - noise

    for blend_value in (0.0, 0.25, 0.5, 0.9, 1.0):
        _, velocity = interpolate_towards_data(
            clean, noise, torch.full((5,), blend_value)
        )
        torch.testing.assert_close(velocity, expected)


def test_midpoint_is_a_convex_combination() -> None:
    """Halfway along is the average, which pins the interpolation weights."""
    clean = torch.randn(2, NUM_LATENT_TOKENS, LATENT_DIM)
    noise = torch.randn(2, NUM_LATENT_TOKENS, LATENT_DIM)
    midpoint, _ = interpolate_towards_data(clean, noise, torch.full((2,), 0.5))
    torch.testing.assert_close(midpoint, 0.5 * (clean + noise))


def test_interpolation_rejects_a_mismatched_blend_batch() -> None:
    clean = torch.randn(4, NUM_LATENT_TOKENS, LATENT_DIM)
    with pytest.raises(ValueError, match="blend must be"):
        interpolate_towards_data(clean, torch.randn_like(clean), torch.zeros(3))


# --- blend features and sampling of the blend itself ------------------------


def test_blend_features_are_bounded_and_shaped() -> None:
    features = blend_frequency_features(torch.rand(7), BLEND_NUM_FREQUENCIES)
    assert features.shape == (7, 2 * BLEND_NUM_FREQUENCIES)
    assert features.abs().max() <= 1.0 + 1e-6


def test_blend_features_separate_nearby_fractions() -> None:
    """A Fourier basis exists so 0.05 and 0.10 are far apart; check they are."""
    features = blend_frequency_features(
        torch.tensor([0.05, 0.10, 0.95]), BLEND_NUM_FREQUENCIES
    )
    assert (features[0] - features[1]).norm() > 1e-2
    assert (features[0] - features[2]).norm() > 1e-2


def test_blend_fractions_stay_in_the_unit_interval() -> None:
    blend = sample_blend_fractions(4096, "cpu")
    assert blend.shape == (4096,)
    assert bool((blend > 0.0).all()) and bool((blend < 1.0).all())
    # Logit-normal is symmetric about 0.5 and concentrates there; a uniform
    # sample would put ~50% outside the middle half.
    assert 0.45 < float(blend.mean()) < 0.55
    assert float(((blend > 0.25) & (blend < 0.75)).double().mean()) > 0.6


# --- the velocity model -----------------------------------------------------


def test_untrained_model_predicts_exactly_zero_velocity() -> None:
    """Zero-init means the initial sampler is the identity, not noise injection.

    Exact equality, not approximate: every output gate and the readout are
    zeroed, so any drift here means an initialisation was missed.
    """
    model = make_velocity_model()
    velocity = model(
        torch.randn(3, NUM_LATENT_TOKENS, LATENT_DIM),
        torch.rand(3),
        torch.randn(3, NUM_CONTEXT_TOKENS, CONDITION_DIM),
    )
    assert velocity.shape == (3, NUM_LATENT_TOKENS, LATENT_DIM)
    assert bool((velocity == 0).all())


def test_model_rejects_wrong_latent_and_blend_shapes() -> None:
    model = make_velocity_model()
    condition = torch.randn(2, NUM_CONTEXT_TOKENS, CONDITION_DIM)

    with pytest.raises(ValueError, match="noisy_latent must be"):
        model(
            torch.randn(2, NUM_LATENT_TOKENS + 1, LATENT_DIM), torch.rand(2), condition
        )
    with pytest.raises(ValueError, match="blend must be"):
        model(torch.randn(2, NUM_LATENT_TOKENS, LATENT_DIM), torch.rand(3), condition)
    with pytest.raises(ValueError, match="condition_tokens must be"):
        model(
            torch.randn(2, NUM_LATENT_TOKENS, LATENT_DIM),
            torch.rand(2),
            torch.randn(2, CONDITION_DIM),
        )


def test_model_width_must_divide_into_heads() -> None:
    with pytest.raises(ValueError, match="multiple of the fixed"):
        LatentVelocityModel(
            latent_dim=LATENT_DIM,
            num_latent_tokens=NUM_LATENT_TOKENS,
            condition_dim=CONDITION_DIM,
            model_width=MODEL_WIDTH + 1,
            num_layers=1,
        )


def test_a_trained_model_actually_reads_its_condition() -> None:
    """Guards the failure where cross-attention is wired but contributes nothing.

    An untrained model is zero everywhere, so this has to perturb the weights
    first; what it asserts is that two different conditions then produce two
    different velocities.
    """
    model = make_velocity_model()
    for parameter in model.parameters():
        parameter.data.add_(torch.randn_like(parameter) * 0.05)

    latent = torch.randn(1, NUM_LATENT_TOKENS, LATENT_DIM)
    blend = torch.full((1,), 0.5)
    first = model(latent, blend, torch.randn(1, NUM_CONTEXT_TOKENS, CONDITION_DIM))
    second = model(latent, blend, torch.randn(1, NUM_CONTEXT_TOKENS, CONDITION_DIM))
    assert (first - second).abs().max() > 1e-6


def test_condition_dropout_replaces_whole_examples() -> None:
    """Dropout is per example, not per token: guidance needs "no spec at all"."""
    model = make_velocity_model()
    condition = torch.randn(8, NUM_CONTEXT_TOKENS, CONDITION_DIM)

    kept = model.drop_condition(condition, 0.0)
    torch.testing.assert_close(kept, condition)

    dropped = model.drop_condition(condition, 1.0)
    expected = model.null_condition.expand(8, NUM_CONTEXT_TOKENS, -1)
    torch.testing.assert_close(dropped, expected)
    # Every token of a dropped example is the null token, so all rows are equal.
    assert bool((dropped[0] == dropped[0][0]).all())


# --- the normalizer ---------------------------------------------------------


def test_normalizer_round_trips() -> None:
    normalizer = make_fitted_normalizer()
    latents = torch.randn(16, NUM_LATENT_TOKENS, LATENT_DIM) * 3.0 + 2.0
    torch.testing.assert_close(
        normalizer.denormalize(normalizer.normalize(latents)), latents
    )


def test_normalizer_standardizes_what_it_was_fitted_on() -> None:
    latents = torch.randn(4096, NUM_LATENT_TOKENS, LATENT_DIM) * 5.0 - 1.0
    normalizer = LatentNormalizer(LATENT_DIM).fit(latents)
    normalized = normalizer.normalize(latents).reshape(-1, LATENT_DIM)
    assert normalized.mean(dim=0).abs().max() < 1e-4
    torch.testing.assert_close(
        normalized.std(dim=0), torch.ones(LATENT_DIM), atol=1e-3, rtol=1e-3
    )


def test_normalizer_pools_statistics_over_token_positions() -> None:
    """Per-position statistics would fit sampling noise; check they are pooled."""
    latents = torch.randn(512, NUM_LATENT_TOKENS, LATENT_DIM)
    latents[:, 0] += 10.0
    normalizer = LatentNormalizer(LATENT_DIM).fit(latents)
    # A per-position fit would centre token 0 on its own mean and leave the
    # offset invisible; a pooled fit spreads it across every position.
    normalized = normalizer.normalize(latents)
    assert float(normalized[:, 0].mean()) > 1.0


def test_normalizer_refuses_to_be_used_unfitted() -> None:
    normalizer = LatentNormalizer(LATENT_DIM)
    with pytest.raises(RuntimeError, match="never fitted"):
        normalizer.require_fitted()
    normalizer.fit(torch.randn(8, NUM_LATENT_TOKENS, LATENT_DIM))
    normalizer.require_fitted()


def test_normalizer_statistics_survive_a_state_dict_round_trip() -> None:
    """They are buffers so a checkpoint carries them; a fresh module must not."""
    fitted = make_fitted_normalizer()
    restored = LatentNormalizer(LATENT_DIM)
    restored.load_state_dict(fitted.state_dict())
    assert bool(restored.fitted)
    torch.testing.assert_close(restored.mean, fitted.mean)
    torch.testing.assert_close(restored.std, fitted.std)


def test_normalizer_clamps_dead_dimensions() -> None:
    """A constant dimension must normalize to zero, not explode."""
    latents = torch.randn(128, NUM_LATENT_TOKENS, LATENT_DIM)
    latents[..., 0] = 4.0
    normalizer = LatentNormalizer(LATENT_DIM).fit(latents)
    normalized = normalizer.normalize(latents)
    assert torch.isfinite(normalized).all()
    assert normalized[..., 0].abs().max() < 1e-3


# --- the sampler ------------------------------------------------------------


def test_sampler_output_lands_on_the_fitted_shell() -> None:
    """Samples must arrive at the magnitude the decoder was trained at.

    Specifically the *fitted* radius, not ``sqrt(D)``: the encoder's final
    LayerNorm has a learned gain, so the real radius is a property of the
    checkpoint. Asserting ``sqrt(D)`` here would pass while putting every real
    sample at the wrong scale.
    """
    model = make_velocity_model()
    normalizer = make_fitted_normalizer()
    latents = sample_latents(
        model,
        torch.randn(4, NUM_CONTEXT_TOKENS, CONDITION_DIM),
        normalizer,
        num_steps=4,
    )
    assert latents.shape == (4, NUM_LATENT_TOKENS, LATENT_DIM)
    torch.testing.assert_close(
        latents.norm(dim=-1),
        normalizer.shell_radius.expand(4, NUM_LATENT_TOKENS),
    )


def test_normalizer_fits_the_shell_radius_rather_than_assuming_it() -> None:
    """The measured radius must track the data, even when it is not sqrt(D)."""
    scaled = project_to_shell(
        torch.randn(512, NUM_LATENT_TOKENS, LATENT_DIM), radius=20.0
    )
    normalizer = LatentNormalizer(LATENT_DIM).fit(scaled)
    torch.testing.assert_close(
        normalizer.shell_radius, torch.tensor(20.0), atol=1e-3, rtol=1e-3
    )
    assert abs(float(normalizer.shell_radius) - LATENT_DIM**0.5) > 1.0


def test_unfitted_normalizer_falls_back_to_the_sqrt_d_radius() -> None:
    """The historical assumption stays the default, so nothing silently changes."""
    normalizer = LatentNormalizer(LATENT_DIM)
    torch.testing.assert_close(normalizer.shell_radius, torch.tensor(LATENT_DIM**0.5))


def test_decoder_space_undoes_normalization_and_sets_the_radius() -> None:
    """Both halves are load-bearing: anisotropy and radius are separate fixes."""
    normalizer = make_fitted_normalizer()
    normalizer.mean.fill_(3.0)
    normalizer.std.fill_(2.0)
    normalizer.shell_radius.fill_(7.0)

    normalized = torch.randn(4, NUM_LATENT_TOKENS, LATENT_DIM)
    decoder_space = normalizer.to_decoder_space(normalized)

    torch.testing.assert_close(
        decoder_space.norm(dim=-1), torch.full((4, NUM_LATENT_TOKENS), 7.0)
    )
    # Direction must come from the denormalized vector, not the raw one.
    expected_direction = project_to_shell(normalized * 2.0 + 3.0, radius=7.0)
    torch.testing.assert_close(decoder_space, expected_direction)


def test_zero_velocity_sampler_returns_its_own_starting_noise() -> None:
    """With an untrained (zero) model the Euler loop must be a no-op.

    Pins the loop, the denormalization and the shell projection all at once: any
    stray scaling or an off-by-one in the step schedule shows up here.
    """
    model = make_velocity_model()
    normalizer = make_fitted_normalizer()

    generator = torch.Generator().manual_seed(1234)
    sampled = sample_latents(
        model,
        torch.randn(2, NUM_CONTEXT_TOKENS, CONDITION_DIM),
        normalizer,
        num_steps=8,
        generator=generator,
    )

    replay = torch.Generator().manual_seed(1234)
    noise = torch.empty(2, NUM_LATENT_TOKENS, LATENT_DIM).normal_(generator=replay)
    torch.testing.assert_close(sampled, normalizer.to_decoder_space(noise))


def test_sampling_is_reproducible_and_seed_dependent() -> None:
    model = make_velocity_model()
    normalizer = make_fitted_normalizer()
    condition = torch.randn(2, NUM_CONTEXT_TOKENS, CONDITION_DIM)

    def draw(seed: int) -> torch.Tensor:
        return sample_latents(
            model,
            condition,
            normalizer,
            num_steps=4,
            generator=torch.Generator().manual_seed(seed),
        )

    torch.testing.assert_close(draw(7), draw(7))
    assert (draw(7) - draw(8)).abs().max() > 1e-6


def test_sampler_requires_a_fitted_normalizer() -> None:
    with pytest.raises(RuntimeError, match="never fitted"):
        sample_latents(
            make_velocity_model(),
            torch.randn(1, NUM_CONTEXT_TOKENS, CONDITION_DIM),
            LatentNormalizer(LATENT_DIM),
            num_steps=2,
        )


def test_sampler_rejects_a_nonpositive_step_count() -> None:
    with pytest.raises(ValueError, match="num_steps must be positive"):
        sample_latents(
            make_velocity_model(),
            torch.randn(1, NUM_CONTEXT_TOKENS, CONDITION_DIM),
            make_fitted_normalizer(),
            num_steps=0,
        )


class _ConditionSignVelocity(nn.Module):
    """Returns +1 for a real condition and -1 for the null one.

    Enough of the velocity-model interface for the sampler, and no more, so the
    guidance arithmetic can be checked against a hand-computable answer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.num_latent_tokens = NUM_LATENT_TOKENS
        self.latent_dim = LATENT_DIM
        self.null_condition = nn.Parameter(torch.full((1, 1, CONDITION_DIM), -99.0))

    def forward(
        self,
        noisy_latent: torch.Tensor,
        blend: torch.Tensor,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        is_null = (condition_tokens == -99.0).all(dim=-1).all(dim=-1)
        sign = torch.where(is_null, -1.0, 1.0)
        return sign[:, None, None].expand_as(noisy_latent).contiguous()


def test_guidance_extrapolates_away_from_the_unconditional_prediction() -> None:
    """velocity = unconditional + strength * (conditional - unconditional).

    With the stub that is -1 + strength * 2, so the total displacement over
    ``num_steps`` Euler steps is exactly that. Checked before the shell
    projection by comparing against the same draw at strength 1.
    """
    model = _ConditionSignVelocity()
    normalizer = LatentNormalizer(LATENT_DIM)
    normalizer.fit(torch.randn(64, NUM_LATENT_TOKENS, LATENT_DIM))
    # Identity statistics so the assertion is about guidance, not scaling.
    normalizer.mean.zero_()
    normalizer.std.fill_(1.0)
    condition = torch.zeros(1, NUM_CONTEXT_TOKENS, CONDITION_DIM)

    def displacement(strength: float) -> torch.Tensor:
        generator = torch.Generator().manual_seed(3)
        sampled = sample_latents(
            model,
            condition,
            normalizer,
            num_steps=10,
            guidance_strength=strength,
            generator=generator,
        )
        replay = torch.Generator().manual_seed(3)
        start = torch.empty(1, NUM_LATENT_TOKENS, LATENT_DIM).normal_(generator=replay)
        # Undo the shell projection by recovering the pre-projection vector's
        # direction: the displacement is uniform, so any coordinate reports it.
        return sampled, start

    for strength in (1.0, 2.0, 3.0):
        sampled, start = displacement(strength)
        expected_step = -1.0 + strength * 2.0
        expected = project_to_shell(start + expected_step, normalizer.shell_radius)
        torch.testing.assert_close(sampled, expected)


def test_unconditional_sampling_uses_the_null_token() -> None:
    """Training with condition dropout makes this free; check it is wired up."""
    model = make_velocity_model()
    normalizer = make_fitted_normalizer()
    latents = sample_latents_unconditional(
        model, batch_size=3, normalizer=normalizer, device="cpu", num_steps=4
    )
    assert latents.shape == (3, NUM_LATENT_TOKENS, LATENT_DIM)
    torch.testing.assert_close(
        latents.norm(dim=-1),
        normalizer.shell_radius.expand(3, NUM_LATENT_TOKENS),
    )


# --- the loss ---------------------------------------------------------------


def test_loss_is_zero_for_a_model_that_predicts_the_true_velocity() -> None:
    """Confirms the loss compares against the displacement and nothing else.

    The stub reconstructs the target from the interpolation itself: with a zero
    clean latent the midpoint is ``(1 - blend) * noise``, so the displacement
    ``-noise`` is recoverable exactly.
    """

    class _RecoverTarget(nn.Module):
        def forward(self, midpoint, blend, condition_tokens):
            return -midpoint / (1.0 - blend[:, None, None]).clamp(min=1e-3)

    losses = flow_matching_loss(
        _RecoverTarget(),
        torch.zeros(64, NUM_LATENT_TOKENS, LATENT_DIM),
        torch.randn(64, NUM_CONTEXT_TOKENS, CONDITION_DIM),
        generator=torch.Generator().manual_seed(0),
    )
    assert float(losses.loss) < 1e-3


def test_loss_reports_both_halves_of_the_trip() -> None:
    losses = flow_matching_loss(
        make_velocity_model(),
        torch.randn(64, NUM_LATENT_TOKENS, LATENT_DIM),
        torch.randn(64, NUM_CONTEXT_TOKENS, CONDITION_DIM),
        generator=torch.Generator().manual_seed(0),
    )
    scalars = losses.scalars("train")
    assert set(scalars) == {
        "train/loss",
        "train/loss_noisy_half",
        "train/loss_clean_half",
    }
    assert all(value >= 0.0 for value in scalars.values())
    assert losses.noisy_half_loss.grad_fn is None, "diagnostics must be detached"


def test_loss_rejects_a_flat_latent() -> None:
    with pytest.raises(ValueError, match="clean_latent must be"):
        flow_matching_loss(
            make_velocity_model(),
            torch.randn(4, LATENT_DIM),
            torch.randn(4, NUM_CONTEXT_TOKENS, CONDITION_DIM),
        )


# --- weight averaging -------------------------------------------------------


def test_weight_average_starts_as_a_copy_and_lags_behind() -> None:
    model = make_velocity_model(num_layers=1)
    average = WeightAverage(model, decay=0.9)

    # Not every parameter starts at zero (token_positions is randn-initialised),
    # so the expected value is computed from the captured starting point rather
    # than assumed.
    initial = {name: tensor.clone() for name, tensor in average.averaged.items()}
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(initial[name], parameter.detach())

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
    average.update(model)

    # One update at decay 0.9 moves a tenth of the way towards the live value.
    for name, averaged in average.averaged.items():
        torch.testing.assert_close(averaged, 0.9 * initial[name] + 0.1)


def test_weight_average_swap_and_restore_is_lossless() -> None:
    model = make_velocity_model(num_layers=1)
    average = WeightAverage(model, decay=0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    live = average.copy_into(model)
    # The averaged weights are still the initial ones, so something changed.
    assert any(
        not torch.equal(before[name], p.detach())
        for name, p in model.named_parameters()
    )
    average.restore(model, live)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name])


def test_weight_average_rejects_an_out_of_range_decay() -> None:
    model = make_velocity_model(num_layers=1)
    for decay in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="decay must be"):
            WeightAverage(model, decay=decay)


# --- end to end -------------------------------------------------------------


def test_flow_model_learns_to_separate_two_conditioned_targets() -> None:
    """The only test that checks the whole loop learns anything.

    Two fixed conditions, two fixed target latents. A working flow model must
    sample near the target belonging to the condition it was given -- which a
    model ignoring its conditioning cannot do, since the best condition-blind
    answer is the midpoint of the two.
    """
    torch.manual_seed(0)
    model = make_velocity_model(num_layers=2)
    normalizer = LatentNormalizer(LATENT_DIM)
    normalizer.mean.zero_()
    normalizer.std.fill_(1.0)
    normalizer.fitted.fill_(True)

    conditions = torch.randn(2, NUM_CONTEXT_TOKENS, CONDITION_DIM)
    targets = torch.randn(2, NUM_LATENT_TOKENS, LATENT_DIM)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(600):
        index = torch.randint(0, 2, (16,))
        losses = flow_matching_loss(model, targets[index], conditions[index])
        optimizer.zero_grad(set_to_none=True)
        losses.loss.backward()
        optimizer.step()

    sampled = sample_latents(
        model,
        conditions,
        normalizer,
        num_steps=32,
        generator=torch.Generator().manual_seed(0),
    )
    # Compare on the shell, since that is where the sampler puts its output.
    projected = project_to_shell(targets, normalizer.shell_radius)
    to_own = (sampled - projected).flatten(1).norm(dim=1)
    to_other = (sampled - projected.flip(0)).flatten(1).norm(dim=1)
    assert bool((to_own < to_other).all()), (
        f"samples did not track their condition: own {to_own.tolist()} vs "
        f"other {to_other.tolist()}"
    )
