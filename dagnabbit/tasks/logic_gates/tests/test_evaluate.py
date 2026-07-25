"""Correctness tests for bitpacked circuit evaluation.

The load-bearing test is :func:`test_matches_unpacked_reference`, which checks
the packed evaluator against a deliberately naive per-row boolean
implementation. Everything else pins down a specific way the packing can be
subtly wrong: bit order, root/output slot mapping, padding contamination,
cross-graph leakage in the batch, and popcount.
"""


import numpy as np
import pytest
import torch

from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_random_graph_description,
)
from dagnabbit.tasks.logic_gates.bitarrays import get_8bit_adder_truth_table
from dagnabbit.tasks.logic_gates.evaluate import (
    BITS_PER_WORD,
    BitpackedTask,
    adder_task,
    bit_accuracy,
    evaluate_graphs,
    make_valid_bit_mask,
    popcount,
)
from dagnabbit.tasks.logic_gates.operators import GATE_NAMES
from dagnabbit.tasks.logic_gates.reference_circuits import nand_ripple_carry_adder

TRUNK_IN_DEGREES = [2, 2]
NUM_TRUNK_NODE_TYPES = 2
NAND, NOR = 0, 1


# --------------------------------------------------------------------------
# Reference implementation: no packing, no batching, one row at a time.
# --------------------------------------------------------------------------


def evaluate_reference(
    graph: FixedInDegreeDAGDescription,
    root_bits: np.ndarray,
) -> np.ndarray:
    """Evaluate one circuit on unpacked bits. ``root_bits`` is [R, num_rows] bool."""
    values: list[np.ndarray | None] = [None] * graph.num_nodes
    for root in range(graph.num_root_nodes):
        values[root] = root_bits[root].astype(bool)

    output_start = graph.num_root_nodes + graph.num_trunk_nodes
    for node in range(graph.num_root_nodes, graph.num_nodes):
        parents = [values[p] for p in graph.node_inputs_indices[node]]
        node_type = graph.node_types[node]
        if node < output_start:
            name = GATE_NAMES[node_type]
            accumulator = parents[0]
            for parent in parents[1:]:
                if name == "NAND":
                    accumulator = accumulator & parent
                elif name == "NOR":
                    accumulator = accumulator | parent
                else:
                    raise AssertionError(f"unhandled gate {name}")
            values[node] = ~accumulator
        else:
            values[node] = parents[0]

    return np.stack([values[node] for node in range(output_start, graph.num_nodes)])


def pack_rows(bits: np.ndarray) -> torch.Tensor:
    """[K, num_rows] bool -> [K, num_words] uint8, matching bitarrays' bit order."""
    return torch.from_numpy(np.packbits(bits.astype(np.uint8), axis=-1))


def unpack_rows(words: torch.Tensor, num_rows: int) -> np.ndarray:
    """[K, num_words] uint8 -> [K, num_rows] bool."""
    return np.unpackbits(words.cpu().numpy(), axis=-1)[..., :num_rows].astype(bool)


def random_task(
    num_root_nodes: int,
    num_output_nodes: int,
    num_rows: int,
    seed: int,
) -> tuple[BitpackedTask, np.ndarray]:
    """A task with random inputs and targets, plus its unpacked input bits."""
    generator = np.random.default_rng(seed)
    root_bits = generator.integers(
        0, 2, size=(num_root_nodes, num_rows), dtype=np.uint8
    )
    target_bits = generator.integers(
        0, 2, size=(num_output_nodes, num_rows), dtype=np.uint8
    )
    root_values = pack_rows(root_bits.astype(bool))
    task = BitpackedTask(
        root_values=root_values,
        target_values=pack_rows(target_bits.astype(bool)),
        num_rows=num_rows,
        valid_bit_mask=make_valid_bit_mask(num_rows, root_values.shape[1]),
    )
    return task, root_bits.astype(bool)


def make_graph(
    node_inputs_indices: list[list[int]],
    node_types: list[int],
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
) -> FixedInDegreeDAGDescription:
    return FixedInDegreeDAGDescription(
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        num_trunk_node_types=NUM_TRUNK_NODE_TYPES,
        trunk_node_in_degrees=TRUNK_IN_DEGREES,
        node_inputs_indices=node_inputs_indices,
        node_types=node_types,
    )


def random_graph(seed: int, num_trunk_nodes: int = 32) -> FixedInDegreeDAGDescription:
    """A reproducible random graph.

    make_random_graph_description seeds its own random.Random from
    torch.randint, so torch is the channel that controls it -- random.seed()
    would have no effect.
    """
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        return make_random_graph_description(
            num_root_nodes=4,
            num_trunk_nodes=num_trunk_nodes,
            num_output_nodes=3,
            trunk_node_in_degrees=TRUNK_IN_DEGREES,
            num_trunk_node_types=NUM_TRUNK_NODE_TYPES,
        )
    finally:
        torch.random.set_rng_state(state)


# --------------------------------------------------------------------------
# popcount
# --------------------------------------------------------------------------


def test_popcount_is_exhaustively_correct():
    words = torch.arange(256, dtype=torch.uint8)
    expected = torch.tensor([bin(value).count("1") for value in range(256)])
    assert torch.equal(popcount(words).to(torch.int64), expected)


def test_popcount_rejects_wrong_dtype():
    with pytest.raises(TypeError):
        popcount(torch.zeros(4, dtype=torch.int64))


# --------------------------------------------------------------------------
# The main equivalence test
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_rows", [8, 65, 100, 256, 1000])
def test_matches_unpacked_reference(num_rows):
    """Packed evaluation must agree bit-for-bit with the naive implementation.

    Row counts deliberately include values that are not multiples of 8, so the
    final word carries padding.
    """
    graphs = [random_graph(seed) for seed in range(6)]
    task, root_bits = random_task(
        num_root_nodes=graphs[0].num_root_nodes,
        num_output_nodes=graphs[0].num_output_nodes,
        num_rows=num_rows,
        seed=num_rows,
    )

    packed = evaluate_graphs(graphs, task)
    actual = unpack_rows(packed, num_rows)

    for index, graph in enumerate(graphs):
        expected = evaluate_reference(graph, root_bits)
        assert np.array_equal(actual[index], expected), f"graph {index} mismatch"


def test_batch_matches_individual_evaluation():
    """A graph's result must not depend on what else is in the batch."""
    graphs = [random_graph(seed) for seed in range(5)]
    task, _ = random_task(4, 3, num_rows=333, seed=7)

    batched = evaluate_graphs(graphs, task)
    for index, graph in enumerate(graphs):
        alone = evaluate_graphs([graph], task)
        assert torch.equal(batched[index : index + 1], alone)


def test_batch_with_differing_rank_depths():
    """Graphs of different depth in one batch: shallow ones must not be re-run."""
    # Same layout in both -- only the wiring depth differs. Every gate here
    # reads the roots directly, so all four sit at rank 1.
    shallow = make_graph(
        node_inputs_indices=[[], [], [0, 1], [0, 1], [1, 0], [1, 0], [2], [3]],
        node_types=[2, 3, NAND, NAND, NOR, NOR, 4, 4],
        num_root_nodes=2,
        num_trunk_nodes=4,
        num_output_nodes=2,
    )
    # A chain, so each gate sits at its own rank.
    deep = make_graph(
        node_inputs_indices=[[], [], [0, 1], [2, 0], [3, 1], [4, 0], [5], [4]],
        node_types=[2, 3, NAND, NAND, NOR, NAND, 4, 4],
        num_root_nodes=2,
        num_trunk_nodes=4,
        num_output_nodes=2,
    )
    assert max(shallow.node_ranks) < max(deep.node_ranks)

    task, root_bits = random_task(2, 2, num_rows=64, seed=11)
    packed = evaluate_graphs([shallow, deep], task)
    actual = unpack_rows(packed, 64)

    assert np.array_equal(actual[0], evaluate_reference(shallow, root_bits))
    assert np.array_equal(actual[1], evaluate_reference(deep, root_bits))


# --------------------------------------------------------------------------
# Hand-built circuits with known semantics
# --------------------------------------------------------------------------


def test_nand_built_xor():
    """The textbook 4-NAND XOR, to pin down gate semantics independently."""
    graph = make_graph(
        node_inputs_indices=[
            [],  # 0: root a
            [],  # 1: root b
            [0, 1],  # 2: NAND(a, b)
            [0, 2],  # 3: NAND(a, n2)
            [1, 2],  # 4: NAND(b, n2)
            [3, 4],  # 5: NAND(n3, n4) == a XOR b
            [5],  # 6: output
        ],
        node_types=[2, 3, NAND, NAND, NAND, NAND, 4],
        num_root_nodes=2,
        num_trunk_nodes=4,
        num_output_nodes=1,
    )

    a = np.array([0, 0, 1, 1], dtype=bool)
    b = np.array([0, 1, 0, 1], dtype=bool)
    root_values = pack_rows(np.stack([a, b]))
    task = BitpackedTask(
        root_values=root_values,
        target_values=pack_rows((a ^ b)[None, :]),
        num_rows=4,
        valid_bit_mask=make_valid_bit_mask(4, root_values.shape[1]),
    )

    packed = evaluate_graphs([graph], task)
    assert np.array_equal(unpack_rows(packed, 4)[0, 0], a ^ b)

    overall, per_output = bit_accuracy(packed, task)
    assert overall.item() == pytest.approx(1.0)
    assert per_output[0, 0].item() == pytest.approx(1.0)


def test_nor_built_or():
    """NOR(NOR(a, b), NOR(a, b)) == a OR b."""
    graph = make_graph(
        node_inputs_indices=[[], [], [0, 1], [2, 2], [3]],
        node_types=[2, 3, NOR, NOR, 4],
        num_root_nodes=2,
        num_trunk_nodes=2,
        num_output_nodes=1,
    )

    a = np.array([0, 0, 1, 1], dtype=bool)
    b = np.array([0, 1, 0, 1], dtype=bool)
    root_values = pack_rows(np.stack([a, b]))
    task = BitpackedTask(
        root_values=root_values,
        target_values=pack_rows((a | b)[None, :]),
        num_rows=4,
        valid_bit_mask=make_valid_bit_mask(4, root_values.shape[1]),
    )
    assert np.array_equal(unpack_rows(evaluate_graphs([graph], task), 4)[0, 0], a | b)


def test_gate_types_are_not_swapped():
    """Trunk type 0 must be NAND and type 1 NOR, on inputs that distinguish them."""
    a = np.array([0, 0, 1, 1], dtype=bool)
    b = np.array([0, 1, 0, 1], dtype=bool)
    root_values = pack_rows(np.stack([a, b]))
    task = BitpackedTask(
        root_values=root_values,
        target_values=pack_rows(np.zeros((1, 4), dtype=bool)),
        num_rows=4,
        valid_bit_mask=make_valid_bit_mask(4, root_values.shape[1]),
    )

    for gate_type, expected in ((NAND, ~(a & b)), (NOR, ~(a | b))):
        graph = make_graph(
            node_inputs_indices=[[], [], [0, 1], [2]],
            node_types=[2, 3, gate_type, 4],
            num_root_nodes=2,
            num_trunk_nodes=1,
            num_output_nodes=1,
        )
        result = unpack_rows(evaluate_graphs([graph], task), 4)[0, 0]
        assert np.array_equal(result, expected), f"gate type {gate_type} is wrong"


# --------------------------------------------------------------------------
# Padding, scoring, and slot mapping
# --------------------------------------------------------------------------


def test_padding_bits_cannot_affect_the_score():
    """A NOT-heavy circuit fills padding with ones; the score must ignore them."""
    num_rows = 3  # 5 padding bits in the single word
    # An inverter chain: output = NAND(a, a) == NOT a. With all-zero inputs the
    # padding bits become 1 after the NAND, so a missing mask would be visible.
    graph = make_graph(
        node_inputs_indices=[[], [], [0, 1], [2]],
        node_types=[2, 3, NAND, 4],
        num_root_nodes=2,
        num_trunk_nodes=1,
        num_output_nodes=1,
    )
    root_bits = np.zeros((2, num_rows), dtype=bool)
    root_values = pack_rows(root_bits)
    # NAND(0, 0) == 1, so the correct output is all ones on the real rows.
    task = BitpackedTask(
        root_values=root_values,
        target_values=pack_rows(np.ones((1, num_rows), dtype=bool)),
        num_rows=num_rows,
        valid_bit_mask=make_valid_bit_mask(num_rows, root_values.shape[1]),
    )

    packed = evaluate_graphs([graph], task)
    # The padding really is contaminated -- this is what the mask is for.
    assert packed[0, 0, 0].item() == 0xFF
    overall, _ = bit_accuracy(packed, task)
    assert overall.item() == pytest.approx(1.0)


def test_valid_bit_mask_shape_and_content():
    mask = make_valid_bit_mask(num_rows=11, num_words=2)
    assert mask.shape == (2,)
    bits = np.unpackbits(mask.numpy())
    assert bits[:11].all() and not bits[11:].any()

    with pytest.raises(ValueError):
        make_valid_bit_mask(num_rows=17, num_words=2)


def test_bit_accuracy_counts_mismatches_exactly():
    num_rows = 20
    task, _ = random_task(2, 2, num_rows=num_rows, seed=3)
    # Predict the target exactly, then flip 3 known bits in output 0.
    predicted = task.target_values.clone()[None, ...]
    flipped = np.zeros((task.num_output_nodes, num_rows), dtype=bool)
    flipped[0, [0, 5, 19]] = True
    predicted = predicted ^ pack_rows(flipped)[None, ...]

    overall, per_output = bit_accuracy(predicted, task)
    assert per_output[0, 0].item() == pytest.approx(1.0 - 3 / num_rows)
    assert per_output[0, 1].item() == pytest.approx(1.0)
    assert overall.item() == pytest.approx(1.0 - 3 / (num_rows * 2))


def test_adder_task_slot_mapping():
    """Wiring output j straight to root j must score exactly as numpy predicts.

    This is the end-to-end check on the root and output slot ordering: if roots
    were reversed, or outputs were in the wrong order, or the packed bit order
    disagreed with the truth table's, this number would move.
    """
    task = adder_task()
    num_root_nodes, num_output_nodes = 16, 8
    num_trunk_nodes = 4

    node_inputs_indices: list[list[int]] = [[] for _ in range(num_root_nodes)]
    node_types = [NUM_TRUNK_NODE_TYPES + i for i in range(num_root_nodes)]
    # Dead filler trunk gates: the description requires exactly num_trunk_nodes.
    for _ in range(num_trunk_nodes):
        node_inputs_indices.append([0, 1])
        node_types.append(NAND)
    for output_slot in range(num_output_nodes):
        node_inputs_indices.append([output_slot])
        node_types.append(NUM_TRUNK_NODE_TYPES + num_root_nodes)

    graph = make_graph(
        node_inputs_indices=node_inputs_indices,
        node_types=node_types,
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
    )

    _, per_output = bit_accuracy(evaluate_graphs([graph], task), task)

    packed_inputs, packed_sums = get_8bit_adder_truth_table()
    input_bits = np.unpackbits(packed_inputs, axis=-1).astype(bool)
    sum_bits = np.unpackbits(packed_sums, axis=-1).astype(bool)
    for output_slot in range(num_output_nodes):
        expected = (input_bits[output_slot] == sum_bits[output_slot]).mean()
        assert per_output[0, output_slot].item() == pytest.approx(expected)


def test_adder_task_shapes():
    task = adder_task()
    assert task.num_rows == 65536
    assert task.num_words == 65536 // BITS_PER_WORD
    assert task.root_values.shape == (16, task.num_words)
    assert task.target_values.shape == (8, task.num_words)
    assert bool(task.valid_bit_mask.all())


def test_reference_adder_is_exactly_correct():
    """A hand-wired 68-NAND adder must score 1.0 on every bit of every row.

    The strongest end-to-end check available: it exercises root slot mapping,
    gate semantics, 19 ranks of topological evaluation, output slot mapping and
    scoring simultaneously, against a circuit whose behaviour is known from its
    construction rather than from this code.
    """
    graph = nand_ripple_carry_adder()
    task = adder_task()

    outputs = evaluate_graphs([graph], task)
    assert torch.equal(outputs[0], task.target_values)

    overall, per_output = bit_accuracy(outputs, task)
    assert overall.item() == 1.0
    assert bool((per_output == 1.0).all())


def test_reference_adder_rejects_an_impossible_width():
    with pytest.raises(ValueError):
        nand_ripple_carry_adder(num_root_nodes=16, num_output_nodes=4)
    with pytest.raises(ValueError, match="trunk nodes"):
        nand_ripple_carry_adder(num_trunk_nodes=32)


def test_random_circuits_score_near_half():
    """Sanity floor: random circuits on the adder land close to chance."""
    graphs = [
        make_random_graph_description(
            num_root_nodes=16,
            num_trunk_nodes=64,
            num_output_nodes=8,
            trunk_node_in_degrees=TRUNK_IN_DEGREES,
            num_trunk_node_types=NUM_TRUNK_NODE_TYPES,
        )
        for _ in range(8)
    ]
    task = adder_task()
    overall, _ = bit_accuracy(evaluate_graphs(graphs, task), task)
    assert 0.2 < overall.mean().item() < 0.8


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_rejects_mismatched_root_count():
    graph = random_graph(0)  # 4 roots, 3 outputs
    task = adder_task()  # 16 inputs, 8 outputs
    with pytest.raises(ValueError, match="roots"):
        evaluate_graphs([graph], task)


def test_rejects_heterogeneous_batch():
    task, _ = random_task(4, 3, num_rows=32, seed=1)
    mismatched = make_random_graph_description(
        num_root_nodes=4,
        num_trunk_nodes=8,
        num_output_nodes=3,
        trunk_node_in_degrees=TRUNK_IN_DEGREES,
        num_trunk_node_types=NUM_TRUNK_NODE_TYPES,
    )
    with pytest.raises(ValueError, match="homogeneous"):
        evaluate_graphs([random_graph(0, num_trunk_nodes=32), mismatched], task)


def test_rejects_empty_batch():
    with pytest.raises(ValueError):
        evaluate_graphs([], adder_task())
