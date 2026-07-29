"""Tests for the group-relative advantage arithmetic.

The advantage is the whole algorithm: it is the only thing standing in for a
value critic, and a sign error there trains the policy backwards while every
other number in the run still looks plausible. Tested against hand-worked
values, with no model involved.

Run directly for a quick pass::

    python -m dagnabbit.search.tests.test_grpo
"""

import torch

from dagnabbit.search.grpo import GRPOConfig, group_advantages


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


def main() -> None:
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")


if __name__ == "__main__":
    main()
