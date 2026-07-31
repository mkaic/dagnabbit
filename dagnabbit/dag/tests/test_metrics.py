"""Tests for :mod:`dagnabbit.dag.metrics`.

Matthews correlation exists here to catch a model that has learned the marginal
bit rather than the circuit, so the cases that matter are the degenerate ones:
constant predictions, constant targets, and heavy class imbalance. Those are
exactly where a hand-rolled formula goes wrong, and where accuracy misleads.
"""

import math

import pytest
import torch

from dagnabbit.dag.metrics import (
    bucket_by_rank,
    confusion_counts,
    matthews_correlation,
)


def from_bits(predicted: list[int], actual: list[int]) -> tuple[float, bool]:
    """MCC of one output node from explicit bit lists. Returns (mcc, defined)."""
    # [B=1, K=1, C=1, rows]
    logits = torch.tensor(predicted, dtype=torch.float32).reshape(1, 1, 1, -1) - 0.5
    targets = torch.tensor(actual, dtype=torch.float32).reshape(1, 1, 1, -1)
    mcc, defined = matthews_correlation(confusion_counts(logits, targets))
    return float(mcc[0, 0]), bool(defined[0, 0])


def reference_mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    """The textbook formula, written out independently."""
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return (tp * tn - fp * fn) / denominator


# --------------------------------------------------------------------------
# Confusion counts
# --------------------------------------------------------------------------


def test_confusion_counts_sum_over_patches_and_rows():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 4, 16)
    targets = (torch.rand(3, 5, 4, 16) > 0.5).float()
    counts = confusion_counts(logits, targets)

    assert counts.true_positives.shape == (3, 4)
    # Every scored bit lands in exactly one cell.
    assert torch.equal(counts.total, torch.full((3, 4), 5 * 16))

    predicted = logits > 0
    actual = targets > 0.5
    assert torch.equal(
        counts.true_positives, (predicted & actual).sum(dim=(1, 3), dtype=torch.int64)
    )
    assert torch.equal(
        counts.false_negatives,
        (~predicted & actual).sum(dim=(1, 3), dtype=torch.int64),
    )


def test_confusion_counts_reject_bad_shapes():
    with pytest.raises(ValueError, match="same shape"):
        confusion_counts(torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 5))
    with pytest.raises(ValueError, match=r"\[B, K, C, rows\]"):
        confusion_counts(torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))


# --------------------------------------------------------------------------
# Matthews correlation
# --------------------------------------------------------------------------


def test_perfect_and_inverted_predictions():
    actual = [0, 1, 1, 0, 1, 0, 0, 1]
    assert from_bits(actual, actual) == (pytest.approx(1.0), True)
    inverted = [1 - bit for bit in actual]
    assert from_bits(inverted, actual) == (pytest.approx(-1.0), True)


def test_constant_prediction_against_a_varying_target_scores_zero():
    """The failure this metric exists to catch: predicting the majority bit.

    Accuracy rewards it in proportion to the imbalance; MCC must not.
    """
    # 7 of 8 bits are 1, so always-1 scores 0.875 accuracy and zero correlation.
    actual = [1, 1, 1, 1, 1, 1, 1, 0]
    mcc, defined = from_bits([1] * 8, actual)
    assert defined, "the target varies, so MCC is defined"
    assert mcc == pytest.approx(0.0)

    accuracy = sum(p == a for p, a in zip([1] * 8, actual)) / 8
    assert accuracy == pytest.approx(0.875)


def test_constant_target_is_undefined_not_zero():
    """Nothing to correlate against; the entry must be excluded, not failed."""
    for constant in (0, 1):
        actual = [constant] * 8
        mcc, defined = from_bits(actual, actual)  # a perfect prediction, no less
        assert not defined, "a constant target leaves MCC undefined"
        assert mcc == pytest.approx(0.0)


@pytest.mark.parametrize(
    "predicted,actual",
    [
        ([1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1, 1, 0]),
        ([1, 1, 1, 1, 0, 0, 1, 0], [0, 1, 1, 1, 0, 1, 1, 0]),
        ([0, 0, 1, 0, 0, 0, 0, 1], [1, 0, 1, 1, 0, 0, 0, 1]),
    ],
)
def test_matches_the_textbook_formula(predicted, actual):
    tp = sum(p == 1 and a == 1 for p, a in zip(predicted, actual))
    fp = sum(p == 1 and a == 0 for p, a in zip(predicted, actual))
    fn = sum(p == 0 and a == 1 for p, a in zip(predicted, actual))
    tn = sum(p == 0 and a == 0 for p, a in zip(predicted, actual))
    mcc, defined = from_bits(predicted, actual)
    assert defined
    assert mcc == pytest.approx(reference_mcc(tp, fp, fn, tn))


def test_stays_in_range_over_random_batches():
    torch.manual_seed(0)
    logits = torch.randn(16, 8, 4, 64)
    targets = (torch.rand(16, 8, 4, 64) > 0.3).float()
    mcc, defined = matthews_correlation(confusion_counts(logits, targets))
    assert mcc.shape == (16, 4)
    assert bool(((mcc >= -1.0) & (mcc <= 1.0)).all())
    # Independent logits and targets: correlation should sit near zero.
    assert abs(float(mcc[defined].mean())) < 0.05


def test_exact_for_counts_past_float32_integer_range():
    """Counting is int64 and the ratio float64, so a huge table stays exact."""
    rows = 1 << 22  # 4.2M bits per output, well past 2^24 when squared
    targets = torch.zeros(1, 1, 1, rows)
    targets[..., : rows // 2] = 1.0
    # Predict perfectly except for a single flipped bit.
    logits = targets.clone() - 0.5
    logits[..., 0] = -0.5

    tp, fp, fn, tn = rows // 2 - 1, 0, 1, rows // 2
    mcc, defined = matthews_correlation(confusion_counts(logits, targets))
    assert defined[0, 0]
    assert float(mcc[0, 0]) == pytest.approx(reference_mcc(tp, fp, fn, tn), abs=1e-12)
    assert float(mcc[0, 0]) < 1.0, "one wrong bit must not round to a perfect score"


# --------------------------------------------------------------------------
# Rank bucketing
# --------------------------------------------------------------------------


def test_bucket_by_rank_averages_and_omits_empty_buckets():
    values = torch.tensor([1.0, 3.0, 10.0, 20.0, 30.0])
    ranks = torch.tensor([0, 0, 4, 4, 4])
    means, counts = bucket_by_rank(values, ranks)
    assert means == {0: pytest.approx(2.0), 4: pytest.approx(20.0)}
    assert counts == {0: 2, 4: 3}
    # Ranks 1-3 had nothing in them and must be absent, not reported as zero.
    assert set(means) == {0, 4}


def test_bucket_by_rank_honours_the_validity_mask():
    values = torch.tensor([1.0, 99.0, 10.0, 20.0])
    ranks = torch.tensor([0, 0, 2, 2])
    valid = torch.tensor([True, False, True, True])
    means, counts = bucket_by_rank(values, ranks, valid=valid)
    assert means[0] == pytest.approx(1.0), "the invalid entry must not be averaged in"
    assert counts == {0: 1, 2: 2}


def test_bucket_by_rank_drops_a_fully_invalid_bucket():
    values = torch.tensor([5.0, 6.0, 7.0])
    ranks = torch.tensor([1, 1, 3])
    valid = torch.tensor([False, False, True])
    means, counts = bucket_by_rank(values, ranks, valid=valid)
    assert set(means) == {3}
    assert counts == {3: 1}


def test_bucket_by_rank_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        bucket_by_rank(torch.zeros(3), torch.zeros(4, dtype=torch.long))
