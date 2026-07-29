"""Group-relative policy optimization over sampled graphs.

The policy is "read a specification, write a latent, sample a graph from the
decoder". The reward is whatever the task says a good graph is. GRPO's appeal
here is that it needs no value critic: for each specification it draws a
*group* of samples and scores each one against that group's own mean, so the
group is the baseline.

That property is what makes this viable at all. A learned critic would have to
predict reward from a specification it has never seen, which is the same
extrapolation problem that made surrogate-based latent search unattractive.
A group mean is measured, not predicted.

Why the reward has variance to work with
----------------------------------------
Group-relative advantages are useless if every sample in a group scores the
same. Scored against a *fixed structured target* like the 8-bit adder, random
circuits sit at 0.4996 +/- 0.0019 -- flat, and no amount of reweighting
recovers a signal that is not there.

Scored against a **hindsight-relabelled** target -- some random graph's own
behaviour, treated as the goal -- the same circuits spread over 0.47 +/- 0.11,
with best-in-group near 0.75. Roughly fifty times the spread, for free, with no
curriculum. Supplying targets that way is the caller's job; this module only
requires that the reward function has something to say.

Nothing here imports from :mod:`dagnabbit.tasks`. The reward arrives as a
callable, so a new task swaps that and keeps the optimizer.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import FixedInDegreeDAGDescription
from dagnabbit.dag.policy import sample_graphs_from_latent

# Given the sampled graphs and, for each one, the index of the prompt it was
# drawn for, return a [B] float tensor of rewards.
RewardFunction = Callable[
    [Sequence[FixedInDegreeDAGDescription], Tensor],
    Tensor,
]


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 32
    temperature: float = 1.0
    # Entropy bonus. Training starts from a randomly initialized proposer with
    # no supervised reference to anchor it, so the usual KL-to-reference term
    # has nothing to point at; an entropy floor is what keeps the policy from
    # collapsing onto one graph before the reward has taught it anything.
    entropy_weight: float = 0.01
    # Guards the group-std division when a group happens to score identically.
    advantage_epsilon: float = 1e-4
    # Divide advantages by the group standard deviation. Standard GRPO does;
    # turning it off gives the unnormalized ("Dr. GRPO") variant, which drops
    # the bias that scaling by std introduces toward low-variance groups.
    normalize_advantages: bool = True


@dataclass(frozen=True)
class GRPOStats:
    """One step's diagnostics. All plain floats, ready for TensorBoard."""

    loss: float
    policy_loss: float
    reward_mean: float
    reward_best: float
    reward_group_std: float
    advantage_abs_mean: float
    entropy_per_action: float
    log_prob_mean: float

    def scalars(self, prefix: str = "grpo") -> dict[str, float]:
        return {f"{prefix}/{name}": value for name, value in vars(self).items()}


def group_advantages(
    rewards: Tensor,
    config: GRPOConfig,
) -> Tensor:
    """``[P, G]`` rewards -> ``[P, G]`` advantages, centred within each group.

    Split out from the step so the arithmetic can be tested against hand-worked
    numbers without a model in the way.
    """
    centred = rewards - rewards.mean(dim=1, keepdim=True)
    if not config.normalize_advantages:
        return centred
    spread = rewards.std(dim=1, keepdim=True)
    return centred / (spread + config.advantage_epsilon)


def grpo_step(
    model: DagnabbitAutoEncoder,
    proposer: torch.nn.Module,
    specifications: Tensor,
    reward_fn: RewardFunction,
    config: GRPOConfig,
) -> tuple[Tensor, GRPOStats]:
    """One on-policy GRPO update's loss, plus what happened.

    ``specifications`` is ``[P, ...]`` -- P prompts in whatever form the
    proposer reads. Each is repeated ``group_size`` times, so the batch that
    reaches the decoder is ``P * G``.

    Returns the loss to call ``backward`` on; the caller owns the optimizer.

    Single update per batch of samples, so the policy that generated the
    samples *is* the policy being updated: the PPO importance ratio is
    identically 1 and the clipped surrogate reduces to an advantage-weighted
    log-probability. It is omitted rather than computed as a no-op. Doing more
    than one inner epoch would require reinstating it.
    """
    num_prompts = specifications.shape[0]
    group_size = config.group_size

    repeated = specifications.repeat_interleave(group_size, dim=0)
    prompt_indices = torch.arange(
        num_prompts,
        device=specifications.device,
    ).repeat_interleave(group_size)

    sampled = sample_graphs_from_latent(
        model,
        proposer(repeated),
        temperature=config.temperature,
    )
    rewards = reward_fn(sampled.graphs, prompt_indices).to(
        device=sampled.log_probs.device,
        dtype=sampled.log_probs.dtype,
    )
    if rewards.shape != (num_prompts * group_size,):
        raise ValueError(
            f"reward_fn returned {tuple(rewards.shape)}, expected "
            f"({num_prompts * group_size},)"
        )

    grouped = rewards.view(num_prompts, group_size)
    advantages = group_advantages(grouped, config).flatten().detach()

    policy_loss = -(advantages * sampled.log_probs).mean()
    entropy_per_action = (sampled.entropies / sampled.num_actions).mean()
    loss = policy_loss - config.entropy_weight * entropy_per_action

    stats = GRPOStats(
        loss=loss.item(),
        policy_loss=policy_loss.item(),
        reward_mean=grouped.mean().item(),
        reward_best=grouped.max().item(),
        reward_group_std=grouped.std(dim=1).mean().item(),
        advantage_abs_mean=advantages.abs().mean().item(),
        entropy_per_action=entropy_per_action.item(),
        log_prob_mean=sampled.log_probs.mean().item(),
    )
    return loss, stats
