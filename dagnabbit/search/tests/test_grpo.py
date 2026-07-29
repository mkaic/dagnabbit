"""Tests for the group-relative advantage arithmetic.

The advantage is the whole algorithm: it is the only thing standing in for a
value critic, and a sign error there trains the policy backwards while every
other number in the run still looks plausible. Tested against hand-worked
values, with no model involved.

Run directly for a quick pass::

    python -m dagnabbit.search.tests.test_grpo
"""

import torch

from dagnabbit.dag.policy import PolicySample
from dagnabbit.search.grpo import GRPOConfig, group_advantages, grpo_step


def test_advantages_are_centred_within_each_group() -> None:
    rewards = torch.tensor([[0.1, 0.2, 0.6], [10.0, 20.0, 30.0]])
    advantages = group_advantages(rewards, GRPOConfig())
    assert torch.allclose(advantages.mean(dim=1), torch.zeros(2), atol=1e-6), (
        "each group must be centred on its own mean"
    )


def test_groups_are_scored_independently() -> None:
    """A group's advantages must not move when a *different* group shifts.

    This is the property that lets one batch mix easy and hard targets: a
    prompt whose whole group scores poorly still ranks its own samples.
    """
    config = GRPOConfig()
    first = torch.tensor([[0.1, 0.2, 0.6]])
    together = torch.cat([first, torch.tensor([[100.0, 200.0, 300.0]])])
    assert torch.allclose(
        group_advantages(first, config)[0],
        group_advantages(together, config)[0],
        atol=1e-6,
    )


def test_ordering_is_preserved() -> None:
    """Higher reward must mean higher advantage -- the sign-error canary."""
    rewards = torch.tensor([[0.4, 0.9, 0.1, 0.7]])
    advantages = group_advantages(rewards, GRPOConfig())[0]
    assert torch.equal(rewards[0].argsort(), advantages.argsort())
    assert advantages[1] > 0 and advantages[2] < 0


def test_normalization_divides_by_group_spread() -> None:
    """Groups differing only in scale normalize to the same advantages.

    Both spreads here sit far above ``advantage_epsilon``; see
    :func:`test_epsilon_damps_near_flat_groups` for what happens when they do
    not.
    """
    config = GRPOConfig()
    narrow = torch.tensor([[1.0, 2.0, 3.0]])
    wide = torch.tensor([[10.0, 20.0, 30.0]])
    assert torch.allclose(
        group_advantages(narrow, config),
        group_advantages(wide, config),
        atol=1e-3,
    )

    unnormalized = GRPOConfig(normalize_advantages=False)
    assert not torch.allclose(
        group_advantages(narrow, unnormalized),
        group_advantages(wide, unnormalized),
        atol=1e-3,
    )


def test_epsilon_damps_near_flat_groups() -> None:
    """A group whose spread is below the epsilon gets shrunk, not amplified.

    Division by the group standard deviation is scale-invariant only while that
    deviation dominates ``advantage_epsilon``. Below it the epsilon takes over
    and advantages shrink toward zero, which is the behaviour worth having:
    a near-flat group carries almost no signal, and normalizing it to unit
    scale would hand the update pure sampling noise at full strength.

    This is not a corner case. Scored against the adder, a group of random
    circuits spreads by about 0.002 -- the regime this guards.
    """
    config = GRPOConfig()
    separated = group_advantages(torch.tensor([[1.0, 2.0, 3.0]]), config)
    near_flat = group_advantages(torch.tensor([[0.50000, 0.50001, 0.50002]]), config)
    assert separated.abs().max() > 0.9
    assert near_flat.abs().max() < 0.5


def test_identical_rewards_do_not_explode() -> None:
    """A group that scores identically has no signal, and must yield none.

    Without the epsilon this divides zero by zero and poisons the update with
    NaNs -- and a flat group is exactly what happens when the reward is scored
    against a target nothing in the group can get traction on.
    """
    rewards = torch.full((2, 8), 0.4996)
    advantages = group_advantages(rewards, GRPOConfig())
    assert torch.isfinite(advantages).all()
    assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-6)


def test_reward_parts_are_scored_independently() -> None:
    """Each part of a ``[P, G, C]`` reward is centred on its own group column.

    This is what per-output scoring buys: an output plane every sample finds
    equally hard contributes nothing, instead of shifting the ranking on a
    plane they actually differ on.
    """
    rewards = torch.tensor([[[0.1, 5.0], [0.2, 5.0], [0.6, 5.0]]])
    advantages = group_advantages(rewards, GRPOConfig())

    assert torch.allclose(advantages[..., 1], torch.zeros(1, 3), atol=1e-3), (
        "a plane that is flat across the group must yield no advantage"
    )
    alone = group_advantages(rewards[..., :1], GRPOConfig())
    assert torch.allclose(advantages[..., 0], alone[..., 0], atol=1e-6), (
        "a plane's advantages must not depend on the other planes"
    )


def _stub_step(
    reward: torch.Tensor,
    config: GRPOConfig,
    num_tokens: int = 2,
    with_token_log_probs: bool = True,
):
    """Run ``grpo_step`` over a fixed reward with a differentiable stand-in.

    One prompt, so the whole reward is a single group. The proposer emits one
    learnable mean per latent token, and the sampler scores fixed stand-in
    draws against it with the Gaussian log-density the real latent-noise policy
    uses. That shape matters: a log-probability that ignored which sample was
    drawn would make every group member's gradient identical, and centred
    advantages would then cancel it to exactly zero.
    """
    group_size = config.group_size
    latent = torch.nn.Parameter(torch.zeros(1, num_tokens))
    draws = torch.arange(
        1.0, group_size * num_tokens + 1.0
    ).view(group_size, num_tokens)

    class Proposer(torch.nn.Module):
        def forward(self, specifications):
            return latent

    def sampler(proposed):
        # d/d(mean) of a unit-variance Gaussian log-density is (draw - mean).
        token_log_probs = -0.5 * (draws - proposed) ** 2
        return PolicySample(
            trunk_types=torch.zeros(group_size, 1, dtype=torch.long),
            parent_choices=torch.zeros(group_size, 1, 1, dtype=torch.long),
            log_probs=token_log_probs.sum(dim=1),
            entropies=torch.zeros(group_size),
            num_actions=torch.ones(group_size),
            token_log_probs=token_log_probs if with_token_log_probs else None,
        )

    loss, stats = grpo_step(
        proposer=Proposer(),
        specifications=torch.zeros(1, 1),
        sampler=sampler,
        reward_fn=lambda sample, prompt_indices: reward,
        config=config,
    )
    loss.backward()
    return latent.grad, stats


def test_factored_credit_pays_each_plane_to_its_own_token() -> None:
    """Plane c's advantage must reach latent token c and no other.

    Plane 0 separates the group and plane 1 does not, so token 0 must receive
    gradient and token 1 must receive none -- the property that makes this
    worth the bias documented on ``GRPOConfig.factored_credit``.
    """
    reward = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    gradient, _ = _stub_step(reward, GRPOConfig(group_size=3, factored_credit=True))

    assert gradient[:, 0].abs().sum() > 0.1, "the separating plane must teach"
    assert gradient[:, 1].abs().sum() < 1e-6, (
        "a flat plane must not push the token it was scored against"
    )


def test_aggregate_credit_pushes_tokens_no_plane_separated() -> None:
    """The contrast with factored credit, on the same reward.

    Averaging the planes into one scalar multiplies *every* token's
    log-probability by it, so the token whose own plane was flat is still
    pushed by the other plane's signal. That is the dilution factored credit
    exists to avoid.
    """
    reward = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    gradient, _ = _stub_step(reward, GRPOConfig(group_size=3, factored_credit=False))

    assert gradient[:, 1].abs().sum() > 0.1, (
        "aggregate credit cannot tell the flat plane's token apart"
    )


def test_scalar_rewards_still_work() -> None:
    """A ``[B]`` reward is the C == 1 case and must not need special handling."""
    reward = torch.tensor([0.0, 1.0, 2.0])
    gradient, stats = _stub_step(
        reward, GRPOConfig(group_size=3, factored_credit=False), num_tokens=1
    )
    assert gradient.abs().sum() > 0.1
    assert stats.dead_plane_fraction == 0.0


def test_dead_planes_are_counted() -> None:
    """The flat-plane diagnostic reports what fraction carries no signal."""
    reward = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    _, stats = _stub_step(reward, GRPOConfig(group_size=3, factored_credit=True))
    assert stats.dead_plane_fraction == 0.5, "one of the two planes is flat"


def test_factored_credit_rejects_a_sampler_without_token_log_probs() -> None:
    """Decoder sampling has no latent-token axis; that must fail loudly."""
    reward = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    try:
        _stub_step(
            reward,
            GRPOConfig(group_size=3, factored_credit=True),
            with_token_log_probs=False,
        )
    except ValueError as error:
        assert "token_log_probs" in str(error)
    else:
        raise AssertionError("expected a ValueError naming the missing field")


def main() -> None:
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")


if __name__ == "__main__":
    main()
