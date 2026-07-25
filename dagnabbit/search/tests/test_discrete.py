"""Tests for graph-space evolutionary search.

The properties that matter: mutation never produces an invalid or acyclic-
violating graph, mutation rate means what it claims, elitism never regresses,
and the search actually solves a small problem rather than merely running.
"""

import random

import numpy as np
import pytest
import torch

from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.search.common import score_population
from dagnabbit.search.discrete import evolve_discrete, mutate_graph
from dagnabbit.tasks.logic_gates.evaluate import (
    BitpackedTask,
    adder_task,
    make_valid_bit_mask,
)
from dagnabbit.tasks.logic_gates.reference_circuits import nand_ripple_carry_adder

TRUNK_IN_DEGREES = [2, 2]
NUM_TRUNK_NODE_TYPES = 2


def tiny_graph(seed: int = 0):
    # The generator seeds itself from torch, so torch is what makes this
    # reproducible -- random.seed() has no effect on it.
    torch.manual_seed(seed)
    return make_random_graph_description(
        num_root_nodes=4,
        num_trunk_nodes=16,
        num_output_nodes=2,
        trunk_node_in_degrees=TRUNK_IN_DEGREES,
        num_trunk_node_types=NUM_TRUNK_NODE_TYPES,
    )


def xor_task() -> BitpackedTask:
    """4 inputs, 16 rows; outputs are a XOR b and a AND b of the first two bits.

    Small enough that a (1+16) ES should solve it in seconds, which is what
    makes it usable as a "the search actually works" test.
    """
    num_rows = 16
    bits = np.array(
        [[(row >> shift) & 1 for row in range(num_rows)] for shift in (3, 2, 1, 0)],
        dtype=np.uint8,
    )
    targets = np.stack([bits[0] ^ bits[1], bits[0] & bits[1]])
    root_values = torch.from_numpy(np.packbits(bits, axis=-1))
    return BitpackedTask(
        root_values=root_values,
        target_values=torch.from_numpy(np.packbits(targets, axis=-1)),
        num_rows=num_rows,
        valid_bit_mask=make_valid_bit_mask(num_rows, root_values.shape[1]),
    )


def assert_structurally_valid(graph):
    """Independent re-check of the DAG invariants, not trusting the constructor."""
    output_start = graph.num_root_nodes + graph.num_trunk_nodes
    assert len(graph.node_types) == graph.num_nodes
    assert len(graph.node_inputs_indices) == graph.num_nodes

    for node in range(graph.num_root_nodes):
        assert graph.node_inputs_indices[node] == []

    for node in range(graph.num_root_nodes, graph.num_nodes):
        parents = graph.node_inputs_indices[node]
        if node < output_start:
            node_type = graph.node_types[node]
            assert 0 <= node_type < graph.num_trunk_node_types
            assert len(parents) == graph.trunk_node_in_degrees[node_type]
        else:
            assert len(parents) == 1
        for parent in parents:
            # Acyclicity and the "outputs are leaves" rule.
            assert 0 <= parent < node, f"node {node} parent {parent} not earlier"
            assert parent < output_start, f"node {node} points at output {parent}"


# --------------------------------------------------------------------------
# Mutation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mutation_rate", [0.0, 0.01, 0.1, 0.5, 1.0])
def test_mutation_always_produces_a_valid_graph(mutation_rate):
    rng = random.Random(0)
    graph = tiny_graph()
    for _ in range(50):
        graph = mutate_graph(graph, rng, mutation_rate)
        assert_structurally_valid(graph)


def test_mutation_rate_controls_how_much_changes():
    graph = tiny_graph()

    def changed_genes(rate, trials=40):
        rng = random.Random(1)
        total = 0
        for _ in range(trials):
            mutant = mutate_graph(graph, rng, rate, ensure_mutation=False)
            total += sum(
                original != new
                for original, new in zip(
                    graph.node_inputs_indices, mutant.node_inputs_indices
                )
            )
            total += sum(
                original != new
                for original, new in zip(graph.node_types, mutant.node_types)
            )
        return total / trials

    low, high = changed_genes(0.02), changed_genes(0.4)
    assert low < high
    assert changed_genes(0.0, trials=5) == 0


def test_ensure_mutation_prevents_clones():
    rng = random.Random(0)
    graph = tiny_graph()
    for _ in range(20):
        mutant = mutate_graph(graph, rng, mutation_rate=0.0, ensure_mutation=True)
        assert (
            mutant.node_inputs_indices != graph.node_inputs_indices
            or mutant.node_types != graph.node_types
        )


def test_mutation_can_change_type_and_leaves_arity_consistent():
    """With differing in-degrees, a type change must resize the slot list."""
    rng = random.Random(3)
    torch.manual_seed(3)
    graph = make_random_graph_description(
        num_root_nodes=4,
        num_trunk_nodes=16,
        num_output_nodes=2,
        trunk_node_in_degrees=[2, 3],
        num_trunk_node_types=2,
    )
    for _ in range(50):
        graph = mutate_graph(graph, rng, mutation_rate=0.3)
        assert_structurally_valid(graph)


def test_mutation_rejects_a_bad_rate():
    with pytest.raises(ValueError):
        mutate_graph(tiny_graph(), random.Random(0), mutation_rate=1.5)


def test_mutation_does_not_touch_the_parent():
    rng = random.Random(0)
    graph = tiny_graph()
    before_inputs = [list(parents) for parents in graph.node_inputs_indices]
    before_types = list(graph.node_types)
    mutate_graph(graph, rng, mutation_rate=0.5)
    assert graph.node_inputs_indices == before_inputs
    assert graph.node_types == before_types


# --------------------------------------------------------------------------
# The search loop
# --------------------------------------------------------------------------


def test_best_fitness_never_regresses():
    history = []
    evolve_discrete(
        xor_task(),
        num_generations=30,
        num_offspring=8,
        mutation_rate=0.05,
        seed=0,
        num_root_nodes=4,
        num_trunk_nodes=16,
        num_output_nodes=2,
        on_generation=history.append,
    )
    best = [record.best_fitness for record in history]
    assert best == sorted(best), "elitism broken: best fitness went down"


def test_search_improves_on_random_and_solves_a_small_task():
    task = xor_task()
    result = evolve_discrete(
        task,
        num_generations=400,
        num_offspring=16,
        mutation_rate=0.04,
        seed=0,
        num_root_nodes=4,
        num_trunk_nodes=16,
        num_output_nodes=2,
    )
    assert_structurally_valid(result.best_graph)

    torch.manual_seed(99)
    baseline = score_population(
        [
            make_random_graph_description(
                num_root_nodes=4,
                num_trunk_nodes=16,
                num_output_nodes=2,
                trunk_node_in_degrees=TRUNK_IN_DEGREES,
                num_trunk_node_types=NUM_TRUNK_NODE_TYPES,
            )
            for _ in range(result.evaluations)
        ],
        task,
    )
    assert result.best_fitness > float(baseline.max()), (
        "evolution did no better than drawing the same number of random graphs"
    )
    assert result.best_fitness == 1.0, "should solve XOR/AND on 4 inputs"


def test_stops_early_on_reaching_the_target():
    result = evolve_discrete(
        xor_task(),
        num_generations=10_000,
        num_offspring=16,
        mutation_rate=0.04,
        seed=0,
        num_root_nodes=4,
        num_trunk_nodes=16,
        num_output_nodes=2,
    )
    assert result.best_fitness == 1.0
    assert result.generations < 10_000, "did not stop when the target was reached"


def test_warm_start_from_a_known_circuit_is_not_lost():
    """Seeding with the perfect adder must terminate immediately at 1.0."""
    result = evolve_discrete(
        adder_task(),
        num_generations=5,
        num_offspring=4,
        initial_graphs=[nand_ripple_carry_adder()],
        seed=0,
    )
    assert result.best_fitness == 1.0
    assert result.evaluations == 1, "should stop before mutating a perfect solution"


def test_reproducible_from_seed():
    def run(seed):
        return evolve_discrete(
            xor_task(),
            num_generations=25,
            num_offspring=8,
            mutation_rate=0.05,
            seed=seed,
            num_root_nodes=4,
            num_trunk_nodes=16,
            num_output_nodes=2,
        ).best_fitness

    assert run(0) == run(0)


def test_rejects_bad_configuration():
    with pytest.raises(ValueError):
        evolve_discrete(xor_task(), num_generations=1, num_offspring=0)
    with pytest.raises(ValueError):
        evolve_discrete(xor_task(), num_generations=1, num_parents=0)
