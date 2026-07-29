"""Rewarding a circuit for reproducing a behaviour it was asked to reproduce.

The reward that makes GRPO work here is **not** fitness against a fixed
objective. Scored against the adder, random circuits are pinned at 0.4996 +/-
0.0019, and a group of them carries no usable signal.

Instead, every random graph is treated as its own goal: evaluate it, call what
it computed the target, and reward a proposal for matching *that*. Labels are
exact, unlimited, and need no search -- hindsight relabelling. The reward then
spreads over roughly 0.47 +/- 0.11 within a group, which is what the
group-relative advantage needs.

The wager is that a model which learns to invert arbitrary circuit behaviours
has learned to invert circuit behaviours in general, and that the adder is then
one more query. Whether that transfers across the gap between random-circuit
behaviour and *structured* behaviour is the open question; it is what the
reference-circuit evaluation measures.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from dagnabbit.dag.description import FixedInDegreeDAGDescription
from dagnabbit.tasks.logic_gates.evaluate import (
    BitpackedTask,
    evaluate_graphs,
    popcount,
)

# Circuits per evaluation chunk. The evaluator holds a [chunk, num_nodes,
# num_words] uint8 buffer -- ~1.2 MB per circuit for the adder table -- so a
# whole GRPO batch at once would allocate hundreds of megabytes.
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


def behaviour_accuracy(
    predicted: Tensor,
    targets: Tensor,
    task: BitpackedTask,
) -> Tensor:
    """Per-graph bit accuracy of ``[B, C, W]`` outputs against ``[B, C, W]`` goals.

    The per-graph counterpart of
    :func:`~dagnabbit.tasks.logic_gates.evaluate.bit_accuracy`, which scores a
    batch against one shared target. Here every graph has its own.

    Counted exactly as int64, then divided in float64 on the CPU, matching how
    ``bit_accuracy`` keeps the score exact for tables larger than float32's
    integer range.
    """
    if predicted.shape != targets.shape:
        raise ValueError(
            f"predicted {tuple(predicted.shape)} and targets "
            f"{tuple(targets.shape)} must match"
        )
    mismatched = (predicted ^ targets) & task.valid_bit_mask
    mismatches = popcount(mismatched).sum(dim=(-2, -1), dtype=torch.int64).cpu()
    total_bits = task.num_rows * predicted.shape[1]
    return 1.0 - mismatches.double() / total_bits


def behaviour_match_reward(
    graphs: Sequence[FixedInDegreeDAGDescription],
    prompt_indices: Tensor,
    targets: Tensor,
    task: BitpackedTask,
    chunk_size: int = DEFAULT_EVAL_CHUNK,
) -> Tensor:
    """Reward each sampled graph for matching its prompt's target behaviour.

    ``targets`` is ``[P, C, W]``, one goal behaviour per prompt;
    ``prompt_indices`` is ``[B]`` saying which prompt each graph was drawn for.
    Shaped to be bound into a
    :data:`~dagnabbit.search.grpo.RewardFunction` with ``functools.partial``.
    """
    predicted = packed_behaviours(graphs, task, chunk_size)
    goals = targets.to(predicted.device)[prompt_indices.to(predicted.device)]
    return behaviour_accuracy(predicted, goals, task)


def constant_output_fraction(behaviours: Tensor, task: BitpackedTask) -> float:
    """Fraction of output planes that are constant across the whole table.

    A guard against the obvious reward hack. Nearly a fifth of random graphs'
    output planes are already constant, so a policy can farm real reward by
    learning to emit constants and matching those. If this climbs toward 1
    while reward improves, the reward is being gamed rather than solved.
    """
    ones = popcount(behaviours & task.valid_bit_mask).sum(dim=-1, dtype=torch.int64)
    dead = (ones == 0) | (ones == task.num_rows)
    return float(dead.double().mean())
