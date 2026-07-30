"""Tests for behaviour scoring, weighted towards the degenerate cases.

The statistics here exist because bit accuracy has a global optimum that is not
the thing anyone wants -- see the module docstring of
:mod:`dagnabbit.tasks.logic_gates.rewards`. So the interesting assertions are
about what a *constant* predictor earns, and about the chunking that keeps a
best-of-N candidate set from asking for gigabytes in one allocation.
"""

import pytest
import torch

from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.tasks.logic_gates.evaluate import adder_task, evaluate_choices
from dagnabbit.tasks.logic_gates.rewards import (
    behaviour_accuracy_per_output,
    behaviour_correlation,
    behaviours_from_choices,
    constant_output_fraction,
    score_behaviours,
)

NUM_ROOT_NODES = 16
NUM_TRUNK_NODES = 24
NUM_OUTPUT_NODES = 8
NUM_TRUNK_NODE_TYPES = 2
IN_DEGREES = [2, 2]


def random_choices(count: int, seed: int = 0):
    """Choice tensors for ``count`` random graphs, in canonical position space."""
    torch.manual_seed(seed)
    graphs = [
        make_random_graph_description(
            num_root_nodes=NUM_ROOT_NODES,
            num_trunk_nodes=NUM_TRUNK_NODES,
            num_output_nodes=NUM_OUTPUT_NODES,
            trunk_node_in_degrees=IN_DEGREES,
            num_trunk_node_types=NUM_TRUNK_NODE_TYPES,
        )
        for _ in range(count)
    ]
    num_nodes = NUM_ROOT_NODES + NUM_TRUNK_NODES + NUM_OUTPUT_NODES
    trunk_types = torch.stack(
        [
            graph.canonical_node_types[
                NUM_ROOT_NODES : NUM_ROOT_NODES + NUM_TRUNK_NODES
            ]
            for graph in graphs
        ]
    )
    parent_choices = torch.stack([graph.canonical_parent_positions for graph in graphs])
    assert parent_choices.shape == (count, num_nodes, max(IN_DEGREES))
    return trunk_types, parent_choices


def test_chunking_does_not_change_the_behaviours() -> None:
    """The whole point of the wrapper: same answer, bounded peak allocation."""
    task = adder_task("cpu")
    trunk_types, parent_choices = random_choices(7)

    unchunked = evaluate_choices(trunk_types, parent_choices, task, IN_DEGREES)
    for chunk_size in (1, 2, 3, 7, 64):
        chunked = behaviours_from_choices(
            trunk_types, parent_choices, task, IN_DEGREES, chunk_size=chunk_size
        )
        assert torch.equal(chunked, unchunked), f"chunk_size={chunk_size} diverged"


def test_chunking_rejects_a_nonpositive_size() -> None:
    task = adder_task("cpu")
    trunk_types, parent_choices = random_choices(2)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        behaviours_from_choices(
            trunk_types, parent_choices, task, IN_DEGREES, chunk_size=0
        )


def test_scoring_evaluates_nothing_and_accepts_several_statistics() -> None:
    """score_behaviours takes behaviours, so a caller can evaluate once."""
    task = adder_task("cpu")
    trunk_types, parent_choices = random_choices(4)
    behaviours = behaviours_from_choices(trunk_types, parent_choices, task, IN_DEGREES)
    # Each graph is its own goal: hindsight relabelling, so every score is
    # a perfect match against itself.
    target_indices = torch.arange(4)
    scores = score_behaviours(behaviours, target_indices, behaviours, task)

    assert set(scores) == {"correlation", "accuracy"}
    for name, value in scores.items():
        assert value.shape == (4, NUM_OUTPUT_NODES), name
    torch.testing.assert_close(
        scores["accuracy"],
        torch.ones(4, NUM_OUTPUT_NODES, dtype=torch.float64),
    )


def test_scoring_rejects_an_unknown_statistic() -> None:
    task = adder_task("cpu")
    trunk_types, parent_choices = random_choices(2)
    behaviours = behaviours_from_choices(trunk_types, parent_choices, task, IN_DEGREES)
    with pytest.raises(ValueError, match="unknown behaviour statistic"):
        score_behaviours(
            behaviours, torch.arange(2), behaviours, task, statistics=("nonsense",)
        )


def test_a_constant_prediction_earns_zero_correlation_but_real_accuracy() -> None:
    """The degeneracy the correlation statistic exists to close.

    Against a biased target, always answering with the majority bit scores well
    on accuracy and exactly nothing on correlation. This is why best-of-N
    selection ranks by correlation: ranking by accuracy would actively pick the
    constant-output candidate out of the pool.
    """
    task = adder_task("cpu")
    num_words = task.target_values.shape[1]

    # A target biased 3:1 towards ones, and an all-ones prediction.
    target = torch.zeros(1, 1, num_words, dtype=torch.uint8)
    target[..., : (num_words * 3) // 4] = 0xFF
    constant = torch.full((1, 1, num_words), 0xFF, dtype=torch.uint8)

    accuracy = behaviour_accuracy_per_output(constant, target, task)
    correlation = behaviour_correlation(constant, target, task)

    assert float(accuracy) > 0.7, "majority matching should look good on accuracy"
    assert float(correlation) == 0.0, "and earn exactly nothing on correlation"


def test_constant_output_fraction_spots_a_degenerate_batch() -> None:
    task = adder_task("cpu")
    num_words = task.target_values.shape[1]
    constant = torch.zeros(3, NUM_OUTPUT_NODES, num_words, dtype=torch.uint8)
    assert constant_output_fraction(constant, task) == 1.0

    varied = torch.zeros(1, 1, num_words, dtype=torch.uint8)
    varied[..., : num_words // 2] = 0xFF
    assert constant_output_fraction(varied, task) == 0.0
