"""Correctness tests for bitpacked circuit evaluation.

The load-bearing test is :func:`test_matches_unpacked_reference`, which checks
the packed evaluator against a deliberately naive per-row boolean implementation
written against the *ragged node-index* form. Since the evaluator consumes
node indices, that comparison covers the node layout end to end as well.
Everything else pins down a specific way the packing can be subtly wrong: bit
order, root/output slot mapping, padding contamination, cross-graph leakage in
the batch, and popcount.
"""

import numpy as np
import pytest
import torch

from dagnabbit.dag.graphs import Geometry, from_lists, ranks_from_lists
from dagnabbit.dag.generate import generate_arrays
from dagnabbit.tasks.logic_gates.evaluate import (
    BITS_PER_WORD,
    BitpackedTask,
    adder_task,
    bit_accuracy,
    evaluate_choices,
    evaluate_graphs,
    exhaustive_root_values,
    make_valid_bit_mask,
    popcount,
    unpack_bits,
)
from dagnabbit.tasks.logic_gates.operators import gate_set, nand, nor
from dagnabbit.tasks.logic_gates.reference_circuits import nand_ripple_carry_adder

# These tests exercise NAND/NOR semantics and slot mapping, so they pin that
# pair explicitly rather than following whatever GATE_OPERATORS currently is
# (NAND, XOR, XNOR -- see operators.py for why).
GATE_NAMES = ("NAND", "NOR")
OPERATORS = (nand, nor)

NAND, NOR = 0, 1
# The gate set these geometries are written against. The reference builders
# resolve gates by name through this, so "NAND" means type 0 here and would
# still mean the right type if the pair were ordered the other way.
TEST_GATES = gate_set("NAND", "NOR")
ADDER_GEOMETRY = Geometry(16, 128, 8, 2, (2, 2))
RANDOM_GEOMETRY = Geometry(4, 32, 3, 2, (2, 2))


# --------------------------------------------------------------------------
# Reference implementation: no packing, no batching, one row at a time.
# --------------------------------------------------------------------------


def evaluate_reference(
    node_inputs_indices: list[list[int]],
    node_types: list[int],
    geometry: Geometry,
    root_bits: np.ndarray,
) -> np.ndarray:
    """Evaluate one circuit on unpacked bits. ``root_bits`` is [R, num_rows] bool."""
    values: list[np.ndarray | None] = [None] * geometry.num_nodes
    for root in range(geometry.num_root_nodes):
        values[root] = root_bits[root].astype(bool)

    for node in range(geometry.num_root_nodes, geometry.num_nodes):
        parents = [values[p] for p in node_inputs_indices[node]]
        if node < geometry.output_start:
            name = GATE_NAMES[node_types[node]]
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

    return np.stack(
        [values[node] for node in range(geometry.output_start, geometry.num_nodes)]
    )


def pack_rows(bits: np.ndarray) -> torch.Tensor:
    """[K, num_rows] bool -> [K, num_words] uint8, big-endian within each word."""
    return torch.from_numpy(np.packbits(bits.astype(np.uint8), axis=-1))


def unpack_rows(words: torch.Tensor, num_rows: int) -> np.ndarray:
    """[..., num_words] uint8 -> [..., num_rows] bool."""
    return np.unpackbits(words.cpu().numpy(), axis=-1)[..., :num_rows].astype(bool)


def random_root_bits(
    num_root_nodes: int, num_rows: int, seed: int
) -> tuple[torch.Tensor, np.ndarray]:
    """Random packed input columns, plus the unpacked bits behind them."""
    generator = np.random.default_rng(seed)
    bits = generator.integers(
        0, 2, size=(num_root_nodes, num_rows), dtype=np.uint8
    ).astype(bool)
    return pack_rows(bits), bits


def random_task(
    num_root_nodes: int, num_output_nodes: int, num_rows: int, seed: int
) -> tuple[BitpackedTask, np.ndarray]:
    """A task with random inputs and targets, plus its unpacked input bits."""
    root_values, root_bits = random_root_bits(num_root_nodes, num_rows, seed)
    generator = np.random.default_rng(seed + 1)
    target_bits = generator.integers(
        0, 2, size=(num_output_nodes, num_rows), dtype=np.uint8
    ).astype(bool)
    task = BitpackedTask(
        root_values=root_values,
        target_values=pack_rows(target_bits),
        num_rows=num_rows,
        valid_bit_mask=make_valid_bit_mask(num_rows, root_values.shape[1]),
    )
    return task, root_bits


def circuit(
    node_inputs_indices: list[list[int]],
    node_types: list[int],
    geometry: Geometry,
):
    """A hand-built circuit as both ragged lists and a node-index batch of one."""
    return (
        node_inputs_indices,
        node_types,
        from_lists(node_inputs_indices, node_types, geometry),
    )


def random_circuit(seed: int, geometry: Geometry = RANDOM_GEOMETRY):
    """A reproducible random circuit, in both forms."""
    node_types, in_degrees, parents, _ = generate_arrays(
        1,
        geometry.num_root_nodes,
        geometry.num_trunk_nodes,  # minimum == maximum: every position a gate
        geometry.num_trunk_nodes,
        geometry.num_output_nodes,
        geometry.num_trunk_node_types,
        0.0,  # uniform gate mixture, so these circuits stay reproducible
        geometry.mask_type,
        np.asarray(geometry.trunk_node_in_degrees, dtype=np.int64),
        geometry.maximum_indegree,
        seed,
    )
    inputs = [
        parents[0, node, : in_degrees[0, node]].tolist()
        for node in range(geometry.num_nodes)
    ]
    return circuit(inputs, node_types[0].tolist(), geometry)


def two_root_geometry(num_trunk_nodes: int, num_output_nodes: int) -> Geometry:
    return Geometry(2, num_trunk_nodes, num_output_nodes, 2, (2, 2))


def root_types(geometry: Geometry) -> list[int]:
    return list(range(geometry.root_type_start, geometry.output_type))


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


def test_unpack_bits_round_trips_packbits():
    generator = np.random.default_rng(0)
    words = generator.integers(0, 256, size=(3, 7), dtype=np.uint8)
    expected = np.unpackbits(words, axis=-1)
    assert np.array_equal(unpack_bits(torch.from_numpy(words)).numpy(), expected)


# --------------------------------------------------------------------------
# The main equivalence test
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_rows", [8, 65, 100, 256, 1000])
def test_matches_unpacked_reference(num_rows):
    """Packed evaluation must agree bit-for-bit with the naive implementation.

    Row counts deliberately include values that are not multiples of 8, so the
    final word carries padding.
    """
    circuits = [random_circuit(seed) for seed in range(6)]
    root_values, root_bits = random_root_bits(
        RANDOM_GEOMETRY.num_root_nodes, num_rows, num_rows
    )

    for index, (inputs, types, graphs) in enumerate(circuits):
        packed = evaluate_graphs(graphs, root_values, OPERATORS)
        actual = unpack_rows(packed, num_rows)[0]
        expected = evaluate_reference(inputs, types, RANDOM_GEOMETRY, root_bits)
        assert np.array_equal(actual, expected), f"circuit {index} mismatch"


def test_batch_matches_individual_evaluation():
    """A graph's result must not depend on what else is in the batch."""
    circuits = [random_circuit(seed) for seed in range(5)]
    root_values, _ = random_root_bits(RANDOM_GEOMETRY.num_root_nodes, 333, 7)

    stacked_types = torch.cat([graphs.trunk_types for _, _, graphs in circuits])
    stacked_positions = torch.cat([graphs.parent_indices for _, _, graphs in circuits])
    batched = evaluate_choices(
        stacked_types,
        stacked_positions,
        root_values,
        RANDOM_GEOMETRY.num_output_nodes,
        RANDOM_GEOMETRY.trunk_node_in_degrees,
        OPERATORS,
    )
    for index, (_, _, graphs) in enumerate(circuits):
        alone = evaluate_graphs(graphs, root_values, OPERATORS)
        assert torch.equal(batched[index : index + 1], alone)


def test_batch_with_differing_depths():
    """Graphs of different depth in one batch must both be evaluated fully."""
    geometry = two_root_geometry(num_trunk_nodes=4, num_output_nodes=2)
    # Every gate reads the roots directly, so all four sit at rank 1.
    shallow = circuit(
        [[], [], [0, 1], [0, 1], [1, 0], [1, 0], [2], [3]],
        [
            *root_types(geometry),
            NAND,
            NAND,
            NOR,
            NOR,
            geometry.output_type,
            geometry.output_type,
        ],
        geometry,
    )
    # A chain, so each gate sits at its own rank.
    deep = circuit(
        [[], [], [0, 1], [2, 0], [3, 1], [4, 0], [5], [4]],
        [
            *root_types(geometry),
            NAND,
            NAND,
            NOR,
            NAND,
            geometry.output_type,
            geometry.output_type,
        ],
        geometry,
    )
    assert max(ranks_from_lists(shallow[0], geometry)) < max(
        ranks_from_lists(deep[0], geometry)
    )

    root_values, root_bits = random_root_bits(2, 64, 11)
    stacked_types = torch.cat([shallow[2].trunk_types, deep[2].trunk_types])
    stacked_positions = torch.cat([shallow[2].parent_indices, deep[2].parent_indices])
    packed = evaluate_choices(
        stacked_types,
        stacked_positions,
        root_values,
        geometry.num_output_nodes,
        geometry.trunk_node_in_degrees,
        OPERATORS,
    )
    actual = unpack_rows(packed, 64)

    for index, (inputs, types, _) in enumerate((shallow, deep)):
        expected = evaluate_reference(inputs, types, geometry, root_bits)
        assert np.array_equal(actual[index], expected)


# --------------------------------------------------------------------------
# Hand-built circuits with known semantics
# --------------------------------------------------------------------------


def test_nand_built_xor():
    """The textbook 4-NAND XOR, to pin down gate semantics independently."""
    geometry = two_root_geometry(num_trunk_nodes=4, num_output_nodes=1)
    _, _, graphs = circuit(
        [
            [],  # 0: root a
            [],  # 1: root b
            [0, 1],  # 2: NAND(a, b)
            [0, 2],  # 3: NAND(a, n2)
            [1, 2],  # 4: NAND(b, n2)
            [3, 4],  # 5: NAND(n3, n4) == a XOR b
            [5],  # 6: output
        ],
        [*root_types(geometry), NAND, NAND, NAND, NAND, geometry.output_type],
        geometry,
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

    packed = evaluate_graphs(graphs, root_values, OPERATORS)
    assert np.array_equal(unpack_rows(packed, 4)[0, 0], a ^ b)

    overall, per_output = bit_accuracy(packed, task)
    assert overall.item() == pytest.approx(1.0)
    assert per_output[0, 0].item() == pytest.approx(1.0)


def test_nor_built_or():
    """NOR(NOR(a, b), NOR(a, b)) == a OR b."""
    geometry = two_root_geometry(num_trunk_nodes=2, num_output_nodes=1)
    _, _, graphs = circuit(
        [[], [], [0, 1], [2, 2], [3]],
        [*root_types(geometry), NOR, NOR, geometry.output_type],
        geometry,
    )
    a = np.array([0, 0, 1, 1], dtype=bool)
    b = np.array([0, 1, 0, 1], dtype=bool)
    root_values = pack_rows(np.stack([a, b]))
    packed = evaluate_graphs(graphs, root_values, OPERATORS)
    assert np.array_equal(unpack_rows(packed, 4)[0, 0], a | b)


def test_gate_types_are_not_swapped():
    """Trunk type 0 must be NAND and type 1 NOR, on inputs that distinguish them."""
    geometry = two_root_geometry(num_trunk_nodes=1, num_output_nodes=1)
    a = np.array([0, 0, 1, 1], dtype=bool)
    b = np.array([0, 1, 0, 1], dtype=bool)
    root_values = pack_rows(np.stack([a, b]))

    for gate_type, expected in ((NAND, ~(a & b)), (NOR, ~(a | b))):
        _, _, graphs = circuit(
            [[], [], [0, 1], [2]],
            [*root_types(geometry), gate_type, geometry.output_type],
            geometry,
        )
        result = unpack_rows(evaluate_graphs(graphs, root_values, OPERATORS), 4)[0, 0]
        assert np.array_equal(result, expected), f"gate type {gate_type} is wrong"


# --------------------------------------------------------------------------
# Padding, scoring, and slot mapping
# --------------------------------------------------------------------------


def test_padding_bits_cannot_affect_the_score():
    """A NOT-heavy circuit fills padding with ones; the score must ignore them."""
    num_rows = 3  # 5 padding bits in the single word
    geometry = two_root_geometry(num_trunk_nodes=1, num_output_nodes=1)
    # An inverter: output = NAND(a, a) == NOT a. With all-zero inputs the padding
    # bits become 1 after the NAND, so a missing mask would be visible.
    _, _, graphs = circuit(
        [[], [], [0, 1], [2]],
        [*root_types(geometry), NAND, geometry.output_type],
        geometry,
    )
    root_values = pack_rows(np.zeros((2, num_rows), dtype=bool))
    # NAND(0, 0) == 1, so the correct output is all ones on the real rows.
    task = BitpackedTask(
        root_values=root_values,
        target_values=pack_rows(np.ones((1, num_rows), dtype=bool)),
        num_rows=num_rows,
        valid_bit_mask=make_valid_bit_mask(num_rows, root_values.shape[1]),
    )

    packed = evaluate_graphs(graphs, root_values, OPERATORS)
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
    flipped = np.zeros((task.num_output_nodes, num_rows), dtype=bool)
    flipped[0, [0, 5, 19]] = True
    predicted = task.target_values.clone()[None, ...] ^ pack_rows(flipped)[None, ...]

    overall, per_output = bit_accuracy(predicted, task)
    assert per_output[0, 0].item() == pytest.approx(1.0 - 3 / num_rows)
    assert per_output[0, 1].item() == pytest.approx(1.0)
    assert overall.item() == pytest.approx(1.0 - 3 / (num_rows * 2))


# --------------------------------------------------------------------------
# The exhaustive enumeration and the adder task
# --------------------------------------------------------------------------


def test_exhaustive_root_values_enumerate_the_row_index():
    """Row r must present the integer r, root 0 holding the most significant bit."""
    for num_root_nodes in (2, 4, 8):
        num_rows = 1 << num_root_nodes
        packed = exhaustive_root_values(num_root_nodes)
        assert packed.shape == (num_root_nodes, max(1, num_rows // BITS_PER_WORD))
        # Trailing bits of the final word are padding when the row count is
        # under one word; only the real rows carry the enumeration.
        bits = unpack_bits(packed).numpy()[:, :num_rows]
        weights = 1 << np.arange(num_root_nodes - 1, -1, -1)
        assert np.array_equal(
            (bits * weights[:, None]).sum(axis=0), np.arange(num_rows)
        )


def test_adder_task_shapes():
    task = adder_task()
    assert task.num_rows == 65536
    assert task.num_words == 65536 // BITS_PER_WORD
    assert task.root_values.shape == (16, task.num_words)
    assert task.target_values.shape == (8, task.num_words)
    assert bool(task.valid_bit_mask.all())


def test_adder_task_targets_are_the_sum():
    """Row r holds a = r >> 8 on roots 0-7, b = r & 255 on 8-15, and (a+b) % 256."""
    task = adder_task()
    input_bits = unpack_bits(task.root_values).numpy()
    sum_bits = unpack_bits(task.target_values).numpy()
    weights = 1 << np.arange(7, -1, -1)

    a = (input_bits[:8] * weights[:, None]).sum(axis=0)
    b = (input_bits[8:] * weights[:, None]).sum(axis=0)
    total = (sum_bits * weights[:, None]).sum(axis=0)
    rows = np.arange(task.num_rows)

    assert np.array_equal(a, rows >> 8)
    assert np.array_equal(b, rows & 0xFF)
    assert np.array_equal(total, (a + b) % 256)


def test_adder_task_slot_mapping():
    """Wiring output j straight to root j must score exactly as numpy predicts.

    The end-to-end check on root and output slot ordering: if roots were
    reversed, or outputs were in the wrong order, or the packed bit order
    disagreed with the truth table's, this number would move.
    """
    task = adder_task()
    geometry = Geometry(16, 4, 8, 2, (2, 2))
    inputs: list[list[int]] = [[] for _ in range(geometry.num_root_nodes)]
    types = root_types(geometry)
    # Dead filler gates: the layout requires exactly num_trunk_nodes of them.
    for _ in range(geometry.num_trunk_nodes):
        inputs.append([0, 1])
        types.append(NAND)
    for output_slot in range(geometry.num_output_nodes):
        inputs.append([output_slot])
        types.append(geometry.output_type)

    _, _, graphs = circuit(inputs, types, geometry)
    _, per_output = bit_accuracy(
        evaluate_graphs(graphs, task.root_values, OPERATORS), task
    )

    input_bits = unpack_bits(task.root_values).numpy().astype(bool)
    sum_bits = unpack_bits(task.target_values).numpy().astype(bool)
    for output_slot in range(geometry.num_output_nodes):
        expected = (input_bits[output_slot] == sum_bits[output_slot]).mean()
        assert per_output[0, output_slot].item() == pytest.approx(expected)


def test_reference_adder_is_exactly_correct():
    """A hand-wired NAND adder must score 1.0 on every bit of every row.

    The strongest end-to-end check available: root slot mapping, gate semantics,
    23 ranks of topological evaluation, the node layout, output slot mapping and
    scoring all at once, against a circuit whose behaviour is known from its
    construction rather than from this code.
    """
    graphs = nand_ripple_carry_adder(ADDER_GEOMETRY, TEST_GATES)
    task = adder_task()

    outputs = evaluate_graphs(graphs, task.root_values, OPERATORS)
    assert torch.equal(outputs[0], task.target_values)

    overall, per_output = bit_accuracy(outputs, task)
    assert overall.item() == 1.0
    assert bool((per_output == 1.0).all())


def test_both_adder_vocabularies_compute_the_adder():
    """The NAND and NAND+XOR builds must agree on behaviour and differ on gates.

    The pair only isolates gate vocabulary if everything else is held roughly
    fixed, so this also pins that they stay comparable in depth. If one drifts
    far deeper than the other, the probe stops being a controlled comparison and
    starts measuring depth again.
    """
    from dagnabbit.tasks.logic_gates.operators import DEFAULT_GATE_SET
    from dagnabbit.tasks.logic_gates.reference_circuits import (
        mixed_ripple_carry_adder,
        nand_ripple_carry_adder,
    )

    gates = DEFAULT_GATE_SET
    geometry = Geometry(16, 128, 8, gates.num_types, gates.in_degrees)
    task = adder_task()
    depths = {}
    for name, build in (
        ("nand", nand_ripple_carry_adder),
        ("mixed", mixed_ripple_carry_adder),
    ):
        graphs = build(geometry, gates)
        outputs = evaluate_graphs(graphs, task.root_values, gates.operators)
        assert torch.equal(outputs[0], task.target_values), name
        assert float(bit_accuracy(outputs, task)[0][0]) == 1.0, name
        depths[name] = int(graphs.ranks.max())

        used = graphs.trunk_types[0].unique().tolist()
        if name == "nand":
            assert used == [gates.index_of("NAND")], (
                "the all-NAND build must use only NAND"
            )
        else:
            assert gates.index_of("XOR") in used, (
                "the mixed build must actually spend XOR gates"
            )

    assert abs(depths["nand"] - depths["mixed"]) <= 6, (
        f"the two builds have drifted apart in depth: {depths}"
    )


def test_reference_adder_rejects_an_impossible_width():
    with pytest.raises(ValueError):
        nand_ripple_carry_adder(Geometry(16, 128, 4, 2, (2, 2)), TEST_GATES)
    with pytest.raises(ValueError, match="trunk nodes"):
        nand_ripple_carry_adder(Geometry(16, 32, 8, 2, (2, 2)), TEST_GATES)


def test_random_circuits_score_near_half():
    """Sanity floor: random circuits on the adder land close to chance."""
    from dagnabbit.dag.graphs import sample

    torch.manual_seed(0)
    graphs = sample(8, Geometry(16, 64, 8, 2, (2, 2)))
    task = adder_task()
    overall, _ = bit_accuracy(
        evaluate_graphs(graphs, task.root_values, OPERATORS), task
    )
    assert 0.2 < overall.mean().item() < 0.8


def test_masked_trunk_positions_cannot_affect_the_result():
    """A ``<MASK>`` position is swept over but nothing reads it.

    The evaluator has no notion of masking: it runs *every* trunk position,
    including the ones holding no gate, and those compute whatever their
    all-zero parent slots imply. That is only safe because no live node or
    output references them, which the sampler guarantees. Scribbling arbitrary
    types and wiring into the masked block and getting bit-identical outputs is
    the direct test of that claim -- if it ever fails, masked positions are
    leaking into the circuit.
    """
    from dagnabbit.dag.graphs import SamplingConfig, sample

    geometry = Geometry(8, 32, 4, 2, (2, 2))
    torch.manual_seed(0)
    graphs = sample(16, geometry, sampling=SamplingConfig(minimum_trunk_nodes=8))
    assert bool(graphs.trunk_is_masked.any()), "nothing was masked"

    roots = exhaustive_root_values(geometry.num_root_nodes)
    expected = evaluate_graphs(graphs, roots, OPERATORS)

    masked = graphs.trunk_is_masked
    types = graphs.trunk_types.clone()
    positions = graphs.parent_indices.clone()
    trunk = slice(geometry.num_root_nodes, geometry.output_start)
    types[masked] = NOR
    # Any strictly-earlier index is legal wiring; roots carry real values, so
    # these gates now compute something rather than nothing.
    scribble = torch.randint(0, geometry.num_root_nodes, positions[:, trunk].shape)
    positions[:, trunk] = torch.where(
        masked.unsqueeze(-1), scribble, positions[:, trunk]
    )

    perturbed = evaluate_choices(
        types,
        positions,
        roots,
        geometry.num_output_nodes,
        geometry.trunk_node_in_degrees,
        OPERATORS,
    )
    assert torch.equal(perturbed, expected)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_rejects_a_forward_reference():
    """A parent at or after its consumer would silently alias a real node."""
    _, _, graphs = random_circuit(0)
    positions = graphs.parent_indices.clone()
    consumer = RANDOM_GEOMETRY.num_root_nodes + 1
    positions[0, consumer, 0] = consumer  # points at itself
    with pytest.raises(ValueError, match="at or after its consumer"):
        evaluate_choices(
            graphs.trunk_types,
            positions,
            exhaustive_root_values(RANDOM_GEOMETRY.num_root_nodes),
            RANDOM_GEOMETRY.num_output_nodes,
            RANDOM_GEOMETRY.trunk_node_in_degrees,
            OPERATORS,
        )


def test_rejects_a_reference_into_the_output_block():
    _, _, graphs = random_circuit(0)
    positions = graphs.parent_indices.clone()
    positions[0, -1, 0] = RANDOM_GEOMETRY.output_start  # an output reading an output
    with pytest.raises(ValueError, match="output block"):
        evaluate_choices(
            graphs.trunk_types,
            positions,
            exhaustive_root_values(RANDOM_GEOMETRY.num_root_nodes),
            RANDOM_GEOMETRY.num_output_nodes,
            RANDOM_GEOMETRY.trunk_node_in_degrees,
            OPERATORS,
        )


def test_rejects_mismatched_node_count():
    _, _, graphs = random_circuit(0)
    with pytest.raises(ValueError, match="expected"):
        evaluate_choices(
            graphs.trunk_types,
            graphs.parent_indices[:, :-1],
            exhaustive_root_values(RANDOM_GEOMETRY.num_root_nodes),
            RANDOM_GEOMETRY.num_output_nodes,
            RANDOM_GEOMETRY.trunk_node_in_degrees,
            OPERATORS,
        )


# --------------------------------------------------------------------------
# The gate set as configuration
# --------------------------------------------------------------------------


def test_gate_set_validates_its_contents():
    from dagnabbit.tasks.logic_gates.operators import GateSet, gate_set

    assert gate_set("NAND", "XOR").in_degrees == (2, 2)
    assert gate_set("NAND", in_degrees=(3,)).in_degrees == (3,)

    with pytest.raises(ValueError, match="unknown gate"):
        gate_set("NAND", "IMPLIES")
    with pytest.raises(ValueError, match="duplicate gate"):
        gate_set("NAND", "NAND")
    with pytest.raises(ValueError, match="at least one gate"):
        gate_set()
    with pytest.raises(ValueError, match="in-degrees for"):
        GateSet(names=("NAND", "XOR"), in_degrees=(2,))


def test_reference_circuits_follow_the_gate_set_order():
    """Reordering the gate set must move the adder's type ids with it.

    This is the trap the GateSet exists to close: the builders used to hardcode
    "NAND is type 0, XOR is type 1", so a configuration that ordered them the
    other way would have silently emitted XOR wherever the adder wanted NAND and
    still passed every shape check. Both orders must build a real adder, and
    each must use the type its own set assigns.
    """
    from dagnabbit.tasks.logic_gates.operators import gate_set
    from dagnabbit.tasks.logic_gates.reference_circuits import (
        mixed_ripple_carry_adder,
        nand_ripple_carry_adder,
    )

    task = adder_task()
    for gates in (gate_set("NAND", "XOR", "XNOR"), gate_set("XNOR", "XOR", "NAND")):
        geometry = Geometry(16, 128, 8, gates.num_types, gates.in_degrees)

        plain = nand_ripple_carry_adder(geometry, gates)
        assert plain.trunk_types[0].unique().tolist() == [gates.index_of("NAND")]
        outputs = evaluate_graphs(plain, task.root_values, gates.operators)
        assert float(bit_accuracy(outputs, task)[0][0]) == 1.0, f"nand, {gates.names}"

        mixed = mixed_ripple_carry_adder(geometry, gates)
        assert gates.index_of("XOR") in mixed.trunk_types[0].unique().tolist()
        outputs = evaluate_graphs(mixed, task.root_values, gates.operators)
        assert float(bit_accuracy(outputs, task)[0][0]) == 1.0, f"mixed, {gates.names}"


def test_a_gate_set_without_nand_says_so():
    """The adder is written in NAND; a set lacking it cannot build one."""
    from dagnabbit.tasks.logic_gates.operators import gate_set
    from dagnabbit.tasks.logic_gates.reference_circuits import (
        mixed_ripple_carry_adder,
        nand_ripple_carry_adder,
    )

    no_nand = gate_set("AND", "OR", "XOR")
    geometry = Geometry(16, 128, 8, no_nand.num_types, no_nand.in_degrees)
    with pytest.raises(ValueError, match="no NAND gate"):
        nand_ripple_carry_adder(geometry, no_nand)

    no_xor = gate_set("NAND", "NOR")
    geometry = Geometry(16, 128, 8, no_xor.num_types, no_xor.in_degrees)
    with pytest.raises(ValueError, match="no XOR gate"):
        mixed_ripple_carry_adder(geometry, no_xor)


def test_a_geometry_that_disagrees_with_the_gate_set_is_rejected():
    from dagnabbit.tasks.logic_gates.operators import gate_set
    from dagnabbit.tasks.logic_gates.reference_circuits import nand_ripple_carry_adder

    two = gate_set("NAND", "XOR")
    with pytest.raises(ValueError, match="trunk types but the gate set"):
        nand_ripple_carry_adder(Geometry(16, 128, 8, 3, (2, 2, 2)), two)


def test_the_configured_gate_set_drives_the_geometry():
    """config.GEOMETRY must not restate what config.GATES already says."""
    from dagnabbit.scripts import config as cfg

    assert cfg.GEOMETRY.num_trunk_node_types == cfg.GATES.num_types
    assert cfg.GEOMETRY.trunk_node_in_degrees == cfg.GATES.in_degrees
    assert len(cfg.GATES.operators) == cfg.GATES.num_types


def test_probes_are_skipped_when_the_gate_set_cannot_express_them():
    """A gate set chosen for training must not abort the run over a probe.

    ``probe_references`` used to build both adders unconditionally, so a
    NAND-only configuration raised at the very first eval -- step 0, before a
    single optimizer step had been taken.
    """
    from dagnabbit.scripts.train_simulator import available_probes
    from dagnabbit.tasks.logic_gates.operators import gate_set

    def names(gates):
        return [name for name, _ in available_probes(gates)]

    assert names(gate_set("NAND", "XOR", "XNOR")) == ["nand", "mixed"]
    assert names(gate_set("XNOR", "XOR", "NAND")) == ["nand", "mixed"]
    # The mixed adder needs XOR; the all-NAND one does not.
    assert names(gate_set("NAND")) == ["nand"]
    assert names(gate_set("NAND", "NOR")) == ["nand"]
    # Neither is expressible without NAND, and that is not an error.
    assert names(gate_set("AND", "OR", "XOR")) == []


def test_every_declared_probe_builds_under_a_gate_set_that_satisfies_it():
    """The declared requirements must be the real ones, not a stale guess."""
    from dagnabbit.dag.graphs import Geometry
    from dagnabbit.scripts.train_simulator import REFERENCE_PROBES
    from dagnabbit.tasks.logic_gates.evaluate import adder_task, bit_accuracy
    from dagnabbit.tasks.logic_gates.evaluate import evaluate_graphs
    from dagnabbit.tasks.logic_gates.operators import gate_set

    task = adder_task()
    for name, build, required in REFERENCE_PROBES:
        gates = gate_set(*required)
        geometry = Geometry(16, 128, 8, gates.num_types, gates.in_degrees)
        graphs = build(geometry, gates)
        outputs = evaluate_graphs(graphs, task.root_values, gates.operators)
        assert float(bit_accuracy(outputs, task)[0][0]) == 1.0, name
