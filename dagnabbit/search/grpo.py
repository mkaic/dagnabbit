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

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from dagnabbit.dag.policy import PolicySample

# Given the drawn sample (graphs in choice-tensor form) and, for each row, the
# index of the prompt it was drawn for, return the rewards: [B], or [B, C] to
# score C parts of the specification separately (see GRPOConfig.factored_credit).
RewardFunction = Callable[
    [PolicySample, Tensor],
    Tensor,
]

# Given [B, K, D] proposed latents, draw one graph per row and report the
# log-probability of having drawn it. Which source of randomness that uses is
# not this module's business; see :mod:`dagnabbit.dag.policy`.
PolicySampler = Callable[[Tensor], PolicySample]


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 32
    # Entropy bonus. Only bites for a policy whose spread is learnable: a
    # fixed-sigma latent policy has constant entropy, so this contributes no
    # gradient and exploration is governed entirely by that sigma.
    entropy_weight: float = 0.0
    # Guards the group-std division when a group happens to score identically.
    advantage_epsilon: float = 1e-4
    # Divide advantages by the group standard deviation. Standard GRPO does;
    # turning it off gives the unnormalized ("Dr. GRPO") variant, which drops
    # the bias that scaling by std introduces toward low-variance groups.
    normalize_advantages: bool = True
    # Pay each part of a [B, C] reward to the latent token positionally
    # responsible for it, instead of averaging the parts into one scalar that
    # multiplies the whole log-probability.
    #
    # This is a bias-for-variance trade, and worth being explicit about. The
    # unbiased estimator is ``(sum_c A_c) * sum_k grad log p(z_k)``; factoring
    # keeps only the ``c == k`` terms. Those cross terms are genuinely nonzero
    # here -- the decoder attends over the whole latent, so every output plane
    # depends on every token -- so this is a heuristic, not an identity. What
    # it buys is a per-plane learning signal instead of one averaged scalar,
    # which matters most when planes differ sharply in difficulty and an
    # improvement on a hard one is diluted by seven unchanged ones.
    #
    # Requires the sampler to populate ``PolicySample.token_log_probs`` with a
    # token axis matching the reward's C.
    factored_credit: bool = True


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
    # Fraction of (prompt, output-plane) cells whose group is flat enough that
    # the epsilon guard zeroes its advantages -- signal that isn't there. With
    # a correlation reward this is expected to be substantial: a constant
    # target plane scores 0 for every sample in the group, by construction.
    dead_plane_fraction: float

    def scalars(self, prefix: str = "grpo") -> dict[str, float]:
        return {f"{prefix}/{name}": value for name, value in vars(self).items()}


def group_advantages(
    rewards: Tensor,
    config: GRPOConfig,
) -> Tensor:
    """Rewards -> advantages, centred within each group.

    Takes ``[P, G]``, or ``[P, G, C]`` to centre and scale each of C reward
    parts against its own group statistics. Reducing over axis 1 either way,
    the group axis, so the C parts never mix: a plane every sample finds hard
    does not drag down the ranking on a plane they differ on.

    Split out from the step so the arithmetic can be tested against hand-worked
    numbers without a model in the way.
    """
    centred = rewards - rewards.mean(dim=1, keepdim=True)
    if not config.normalize_advantages:
        return centred
    spread = rewards.std(dim=1, keepdim=True)
    return centred / (spread + config.advantage_epsilon)


def grpo_step(
    proposer: torch.nn.Module,
    specifications: Tensor,
    sampler: PolicySampler,
    reward_fn: RewardFunction,
    config: GRPOConfig,
) -> tuple[Tensor, GRPOStats]:
    """One on-policy GRPO update's loss, plus what happened.

    ``specifications`` is ``[P, ...]`` -- P prompts in whatever form the
    proposer reads. The batch reaching the decoder is ``P * G``.

    ``reward_fn`` may score the specification as a whole (``[B]``) or in C
    parts (``[B, C]``). Parts are centred and scaled against their own group
    statistics either way; ``GRPOConfig.factored_credit`` decides whether each
    part's advantage is then paid to its own latent token or averaged into one
    scalar for the whole log-probability.

    The proposer runs on the P *distinct* specifications and its latents are
    repeated, rather than repeating the specifications and encoding each G
    times. The proposer is deterministic, so both give identical latents and
    identical gradients (autograd sums the G contributions back through the one
    forward pass) -- but a specification here is a 2 MB truth-table image and a
    latent is a few KB, so the difference decides whether a large group fits in
    memory at all.

    Returns the loss to call ``backward`` on; the caller owns the optimizer.

    Single update per batch of samples, so the policy that generated the
    samples *is* the policy being updated: the PPO importance ratio is
    identically 1 and the clipped surrogate reduces to an advantage-weighted
    log-probability. It is omitted rather than computed as a no-op. Doing more
    than one inner epoch would require reinstating it.
    """
    num_prompts = specifications.shape[0]
    group_size = config.group_size

    prompt_indices = torch.arange(
        num_prompts,
        device=specifications.device,
    ).repeat_interleave(group_size)

    sampled = sampler(proposer(specifications).repeat_interleave(group_size, dim=0))
    rewards = reward_fn(sampled, prompt_indices).to(
        device=sampled.log_probs.device,
        dtype=sampled.log_probs.dtype,
    )
    batch_size = num_prompts * group_size
    # A [B] reward is the C == 1 case; carrying one shape from here keeps the
    # two credit-assignment modes from needing separate arithmetic.
    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(-1)
    if rewards.ndim != 2 or rewards.shape[0] != batch_size:
        raise ValueError(
            f"reward_fn returned {tuple(rewards.shape)}, expected "
            f"({batch_size},) or ({batch_size}, C)"
        )
    num_parts = rewards.shape[1]

    grouped = rewards.view(num_prompts, group_size, num_parts)
    advantages = group_advantages(grouped, config).view(batch_size, num_parts).detach()

    if config.factored_credit:
        token_log_probs = sampled.token_log_probs
        if token_log_probs is None:
            raise ValueError(
                "factored_credit needs PolicySample.token_log_probs, which "
                "this sampler does not provide; set factored_credit=False"
            )
        if token_log_probs.shape != (batch_size, num_parts):
            raise ValueError(
                f"token_log_probs is {tuple(token_log_probs.shape)}, but the "
                f"reward has {num_parts} parts to pay to {batch_size} rows; "
                "the latent must carry one token per scored part"
            )
        weighted = (advantages * token_log_probs).sum(dim=-1)
    else:
        # Averaged rather than summed so the loss keeps the scale it had when
        # the reward was a single plane-averaged scalar.
        weighted = advantages.mean(dim=-1) * sampled.log_probs

    policy_loss = -weighted.mean()
    entropy_per_action = (sampled.entropies / sampled.num_actions).mean()
    loss = policy_loss - config.entropy_weight * entropy_per_action

    # Per-sample scalars for reporting: a graph's reward is its plane average,
    # whatever credit assignment the update itself used.
    per_sample = grouped.mean(dim=-1)
    stats = GRPOStats(
        loss=loss.item(),
        policy_loss=policy_loss.item(),
        reward_mean=per_sample.mean().item(),
        reward_best=per_sample.max().item(),
        reward_group_std=per_sample.std(dim=1).mean().item(),
        advantage_abs_mean=advantages.abs().mean().item(),
        entropy_per_action=entropy_per_action.item(),
        log_prob_mean=sampled.log_probs.mean().item(),
        dead_plane_fraction=(
            grouped.std(dim=1) < config.advantage_epsilon
        ).to(torch.float32).mean().item(),
    )
    return loss, stats
