"""Scoring a circuit against a behaviour it was asked to reproduce.

Scoring against a *fixed* objective barely discriminates: measured against the
adder, random circuits are pinned at 0.4996 +/- 0.0019. So every graph is
treated as its own goal instead -- evaluate it, call what it computed the
target, and score a proposal for matching *that*. Labels are exact, unlimited,
and need no search: hindsight relabelling.

The wager is that a model which learns to invert arbitrary circuit behaviours
has learned to invert circuit behaviours in general, and that the adder is then
one more query. Whether that transfers across the gap between random-circuit
behaviour and *structured* behaviour is the open question; it is what the
reference-circuit evaluation measures.

Why the score is a correlation and not an accuracy
--------------------------------------------------
Bit accuracy has a degenerate global optimum, and anything optimizing or
selecting against it finds that optimum. Random-circuit output planes are
heavily biased (about a fifth are outright constant), so the highest-scoring
*simple* answer is to read the target's per-plane majority bit and emit a
constant matching it. That earns ``E[max(p, 1-p)]`` -- roughly 0.67 against
relabelled random targets. Measured on policy-gradient runs of 2026-07-28:
reward_mean climbed 0.52 -> 0.67 while ``constant_output_fraction`` climbed
0.13 -> 0.98, which is that hack and nothing else.

The hack pays nothing on the targets actually worth solving. Every output plane
of the adder is exactly balanced -- for fixed ``a``, ``b`` sweeps all 256
values, so each sum bit is 50/50 -- and so is XOR's. A constant predictor
therefore scores exactly 0.5 there, unmoved.

:func:`behaviour_correlation` (the Matthews correlation coefficient, per output
plane) removes the attractor at the root: its numerator is identically zero
whenever the *prediction* is constant, whatever the target's bias, so majority
matching earns nothing. It is also zero whenever the *target* plane is
constant, which correctly marks those planes as carrying no learnable signal
rather than as free score. On a balanced target it reduces to ``2 * accuracy
- 1``, so on the references it is the same measurement rescaled.

This is why in-distribution best-of-N selection must rank by correlation:
against relabelled random targets, ranking by accuracy would preferentially
pick the constant-output candidates. On the references either statistic ranks
the same way.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from dagnabbit.dag.description import FixedInDegreeDAGDescription
from dagnabbit.tasks.logic_gates.evaluate import (
    BitpackedTask,
    evaluate_choices,
    evaluate_graphs,
    popcount,
)

# Circuits per evaluation chunk. The evaluator holds a [chunk, num_nodes,
# num_words] uint8 buffer -- ~1.2 MB per circuit for the adder table -- so a
# whole best-of-N candidate set at once would allocate hundreds of megabytes.
DEFAULT_EVAL_CHUNK = 128


def packed_behaviours(
    graphs: Sequence[FixedInDegreeDAGDescription],
    task: BitpackedTask,
    chunk_size: int = DEFAULT_EVAL_CHUNK,
) -> Tensor:
    """What each graph computes on ``task``'s inputs, as ``[B, C, W]`` packed.

    Only the task's *input* columns are used; its targets are ignored. Chunked
    so batch size and evaluator memory stay decoupled.
    """
    graphs = list(graphs)
    if not graphs:
        raise ValueError("no graphs to evaluate")
    return torch.cat(
        [
            evaluate_graphs(graphs[start : start + chunk_size], task)
            for start in range(0, len(graphs), chunk_size)
        ]
    )


def behaviour_confusion(
    predicted: Tensor,
    targets: Tensor,
    task: BitpackedTask,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-output-plane confusion counts of ``[B, C, W]`` packed behaviours.

    Returns ``(true_positives, false_positives, false_negatives,
    true_negatives)``, each ``[B, C]`` int64 on the CPU. Three popcounts carry
    it: the intersection plus each side's own set bits determine the rest.

    Counted exactly as int64 and returned on the CPU for the same reasons
    :func:`~dagnabbit.tasks.logic_gates.evaluate.bit_accuracy` gives -- the
    count tensor is tiny next to the evaluation, and the ratios downstream want
    a float64 MPS does not have.
    """
    if predicted.shape != targets.shape:
        raise ValueError(
            f"predicted {tuple(predicted.shape)} and targets "
            f"{tuple(targets.shape)} must match"
        )
    mask = task.valid_bit_mask
    true_positives = popcount(predicted & targets & mask).sum(dim=-1, dtype=torch.int64)
    predicted_ones = popcount(predicted & mask).sum(dim=-1, dtype=torch.int64)
    target_ones = popcount(targets & mask).sum(dim=-1, dtype=torch.int64)

    true_positives = true_positives.cpu()
    false_positives = predicted_ones.cpu() - true_positives
    false_negatives = target_ones.cpu() - true_positives
    true_negatives = task.num_rows - true_positives - false_positives - false_negatives
    return true_positives, false_positives, false_negatives, true_negatives


def behaviour_correlation(
    predicted: Tensor,
    targets: Tensor,
    task: BitpackedTask,
) -> Tensor:
    """Per-output-plane Matthews correlation of ``[B, C, W]`` behaviours.

    Returns ``[B, C]`` in [-1, 1]: 1 for an exactly reproduced plane, 0 for a
    plane no better than a constant, negative for an inverted one. This is the
    training reward -- see the module docstring for why accuracy is not.

    Degenerate cases fall out of the arithmetic rather than needing a branch.
    A constant prediction has ``TP*TN - FP*FN == 0`` (one of each product's
    factors is zero), and its denominator is zero too; a constant target
    likewise. Counts are integers, so a non-zero denominator is at least 1, and
    clamping there turns exactly the 0/0 cases into 0.
    """
    true_positives, false_positives, false_negatives, true_negatives = (
        count.double() for count in behaviour_confusion(predicted, targets, task)
    )
    numerator = true_positives * true_negatives - false_positives * false_negatives
    # Paired rather than multiplied out: each pair's product stays under 2^53
    # for tables up to 2^26 rows, so float64 carries it exactly.
    denominator = torch.sqrt(
        (true_positives + false_positives) * (true_positives + false_negatives)
    ) * torch.sqrt(
        (true_negatives + false_positives) * (true_negatives + false_negatives)
    )
    return numerator / denominator.clamp(min=1.0)


def behaviour_accuracy_per_output(
    predicted: Tensor,
    targets: Tensor,
    task: BitpackedTask,
) -> Tensor:
    """Per-output-plane bit accuracy of ``[B, C, W]`` behaviours -> ``[B, C]``.

    Retained alongside :func:`behaviour_correlation` for reporting: it is the
    statistic every run before 2026-07-28 optimized, so logging it keeps new
    runs comparable with those curves. It is *not* a good training signal; the
    module docstring explains what it rewards instead.
    """
    if predicted.shape != targets.shape:
        raise ValueError(
            f"predicted {tuple(predicted.shape)} and targets "
            f"{tuple(targets.shape)} must match"
        )
    mismatched = (predicted ^ targets) & task.valid_bit_mask
    mismatches = popcount(mismatched).sum(dim=-1, dtype=torch.int64).cpu()
    return 1.0 - mismatches.double() / task.num_rows


def behaviour_accuracy(
    predicted: Tensor,
    targets: Tensor,
    task: BitpackedTask,
) -> Tensor:
    """Per-graph bit accuracy of ``[B, C, W]`` outputs against ``[B, C, W]`` goals.

    The per-graph counterpart of
    :func:`~dagnabbit.tasks.logic_gates.evaluate.bit_accuracy`, which scores a
    batch against one shared target. Here every graph has its own. The plane
    average of :func:`behaviour_accuracy_per_output`.
    """
    return behaviour_accuracy_per_output(predicted, targets, task).mean(dim=-1)


# Selectable per-output statistics by name. Each maps ``[B, C, W]`` predictions
# and goals to a ``[B, C]`` score. "correlation" is the one to rank by; see the
# module docstring for why "accuracy" is reporting-only.
BEHAVIOUR_STATISTICS = {
    "correlation": behaviour_correlation,
    "accuracy": behaviour_accuracy_per_output,
}


def behaviour_match_score(
    trunk_types: Tensor,
    parent_choices: Tensor,
    target_indices: Tensor,
    targets: Tensor,
    task: BitpackedTask,
    trunk_node_in_degrees: Sequence[int],
    statistic: str = "correlation",
) -> Tensor:
    """Score each candidate graph for matching its own target behaviour.

    ``trunk_types`` is ``[B, T]`` and ``parent_choices`` is ``[B, N, S]`` -- the
    choice-tensor pair
    :meth:`~dagnabbit.dag.autoencoder.DagnabbitAutoEncoder.descriptions_from_choices`
    consumes. ``targets`` is ``[P, C, W]``, one goal behaviour per
    specification; ``target_indices`` is ``[B]`` saying which specification each
    candidate was drawn for, so a best-of-N candidate set for P specifications
    scores in one call. Returns ``[B, C]``, one score per output plane, left
    undecomposed so callers can weight or average planes themselves.

    Evaluation runs straight off the choice tensors
    (:func:`~dagnabbit.tasks.logic_gates.evaluate.evaluate_choices`), never
    materializing description objects: at best-of-N candidate counts that Python
    detour dominates everything else.
    """
    if statistic not in BEHAVIOUR_STATISTICS:
        raise ValueError(
            f"unknown behaviour statistic {statistic!r}; expected one of "
            f"{sorted(BEHAVIOUR_STATISTICS)}"
        )
    predicted = evaluate_choices(
        trunk_types,
        parent_choices,
        task,
        trunk_node_in_degrees,
    )
    goals = targets.to(predicted.device)[target_indices.to(predicted.device)]
    return BEHAVIOUR_STATISTICS[statistic](predicted, goals, task)


def constant_output_fraction(behaviours: Tensor, task: BitpackedTask) -> float:
    """Fraction of output planes that are constant across the whole table.

    A guard against the obvious reward hack. Nearly a fifth of random graphs'
    output planes are already constant, so a policy can farm real reward by
    learning to emit constants and matching those. If this climbs toward 1
    while reward improves, the reward is being gamed rather than solved. Under
    an accuracy reward it did exactly that, reaching 0.98 -- which is what
    :func:`behaviour_correlation` exists to prevent.

    Counted on the CPU: the fraction is a float64 mean and MPS has no float64.
    """
    ones = popcount(behaviours & task.valid_bit_mask).sum(dim=-1, dtype=torch.int64)
    dead = (ones == 0) | (ones == task.num_rows)
    return float(dead.cpu().double().mean())
