"""Scoring predicted truth-table bits against the exact ones.

Bit accuracy alone cannot tell "read the circuit" apart from "predicted the
majority bit". A random NAND/NOR circuit's output is often lopsided -- sometimes
outright constant -- and a model that learns the marginal and ignores the graph
scores well above 0.5 on exactly the outputs it understands least. Matthews
correlation is the check: it is 0 for any constant predictor regardless of class
balance, so it only moves when a prediction tracks the *specific* circuit.

Counting is done exactly, as int64, on whatever device the batch lives on; the
ratio is then taken in float64 on the CPU. The count tensor is only
``[B, num_output_nodes]``, so the transfer is free next to the forward pass, and
it keeps the score exact for tables well past float32's 2^24 integer range while
sidestepping MPS, which has no float64 at all. This mirrors what
:func:`~dagnabbit.tasks.logic_gates.evaluate.bit_accuracy` does.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConfusionCounts:
    """Per-(graph, output) confusion counts, all ``[B, C]`` int64."""

    true_positives: Tensor
    false_positives: Tensor
    false_negatives: Tensor
    true_negatives: Tensor

    @property
    def total(self) -> Tensor:
        return (
            self.true_positives
            + self.false_positives
            + self.false_negatives
            + self.true_negatives
        )

    def cpu_double(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return tuple(
            count.cpu().double()
            for count in (
                self.true_positives,
                self.false_positives,
                self.false_negatives,
                self.true_negatives,
            )
        )


def confusion_counts(logits: Tensor, targets: Tensor) -> ConfusionCounts:
    """``[B, K, C, rows]`` logits and 0/1 targets -> per-(graph, output) counts.

    Patches and rows are both summed over, so a count is over every bit scored
    for that output node this step.
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} and targets {tuple(targets.shape)} "
            "must have the same shape"
        )
    if logits.ndim != 4:
        raise ValueError(f"expected [B, K, C, rows]; got {tuple(logits.shape)}")

    predicted = logits > 0
    actual = targets > 0.5
    over = (1, 3)

    def count(mask: Tensor) -> Tensor:
        return mask.sum(dim=over, dtype=torch.int64)

    return ConfusionCounts(
        true_positives=count(predicted & actual),
        false_positives=count(predicted & ~actual),
        false_negatives=count(~predicted & actual),
        true_negatives=count(~predicted & ~actual),
    )


def matthews_correlation(counts: ConfusionCounts) -> tuple[Tensor, Tensor]:
    """Returns ``(mcc, defined)``, both ``[B, C]`` on the CPU.

    ``mcc`` is in [-1, 1]; ``defined`` is false where the *target* was constant
    over the bits scored, which leaves nothing to correlate against. Those
    entries are set to 0 and should be excluded from any average rather than
    counted as failures -- predicting a constant output correctly is the right
    answer, and scoring it 0 would understate a model that got it right.

    A constant *prediction* against a varying target is a different matter: that
    is a real failure and correctly scores 0, so it stays in the average.

    The denominator is built from four square roots rather than the square root
    of a fourfold product, which keeps every intermediate near the scale of the
    counts themselves.
    """
    tp, fp, fn, tn = counts.cpu_double()

    actual_positives = tp + fn
    actual_negatives = tn + fp
    predicted_positives = tp + fp
    predicted_negatives = tn + fn

    defined = (actual_positives > 0) & (actual_negatives > 0)
    # A constant prediction leaves the denominator at zero even when the target
    # varies; that case is a legitimate score of 0, not an undefined one.
    nonzero = defined & (predicted_positives > 0) & (predicted_negatives > 0)

    denominator = (
        predicted_positives.sqrt()
        * actual_positives.sqrt()
        * actual_negatives.sqrt()
        * predicted_negatives.sqrt()
    )
    mcc = torch.where(
        nonzero,
        (tp * tn - fp * fn) / denominator.clamp(min=1.0),
        torch.zeros_like(denominator),
    )
    return mcc, defined


def bucket_by_rank(
    values: Tensor, ranks: Tensor, valid: Tensor | None = None
) -> tuple[dict[int, float], dict[int, int]]:
    """Average ``values`` into buckets keyed by integer ``ranks``.

    All three are flat and the same length. Returns ``(means, counts)`` over the
    entries that are valid, with empty buckets omitted entirely rather than
    reported as zero.
    """
    if values.shape != ranks.shape:
        raise ValueError("values and ranks must be the same shape")
    values = values.detach().cpu().double()
    ranks = ranks.detach().cpu().long()
    weights = (
        torch.ones_like(values)
        if valid is None
        else valid.detach().cpu().double().reshape(values.shape)
    )

    num_buckets = int(ranks.max().item()) + 1 if ranks.numel() else 0
    totals = torch.zeros(num_buckets, dtype=torch.float64)
    counts = torch.zeros(num_buckets, dtype=torch.float64)
    totals.scatter_add_(0, ranks, values * weights)
    counts.scatter_add_(0, ranks, weights)

    return (
        {
            rank: float(totals[rank] / counts[rank])
            for rank in range(num_buckets)
            if counts[rank] > 0
        },
        {rank: int(counts[rank]) for rank in range(num_buckets) if counts[rank] > 0},
    )
