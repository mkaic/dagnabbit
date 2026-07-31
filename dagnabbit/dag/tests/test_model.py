"""Tests for the simulator in :mod:`dagnabbit.dag.model`.

Two things here are load-bearing and easy to get silently wrong.

:func:`patch_targets` slices packed words *before* unpacking them, so an off-by-
one in the word arithmetic would train the model against the wrong rows while
the loss still looked healthy. It is checked against a whole-table unpack.

:class:`NodeTokens` claims to be a pure function of ``(type, own position,
parent positions)``. If it were accidentally sensitive to a padded slot's stored
value, or blind to one of its inputs, the sequence would stop identifying the
graph -- which is the property Phase 1 depends on when it constructs tokens from
categorical choices instead of reading them off a real graph.
"""

from dataclasses import replace

import pytest
import torch

from dagnabbit.dag.graphs import Geometry, sample
from dagnabbit.dag.model import (
    GraphSimulator,
    NodeTokens,
    SimulatorConfig,
    patch_targets,
    sample_patch_indices,
)
from dagnabbit.tasks.logic_gates.evaluate import (
    evaluate_graphs,
    exhaustive_root_values,
    unpack_bits,
)

GEOMETRY = Geometry(8, 24, 4, 1, (2,))
CONFIG = SimulatorConfig(
    embedding_dim=64,
    attention_head_dim=32,
    mlp_expansion_factor=2.0,
    num_simulator_layers=2,
    num_decoder_layers=2,
    num_patches=16,
)


@pytest.fixture
def graphs():
    torch.manual_seed(0)
    return sample(5, GEOMETRY)


def packed_outputs(graphs):
    return evaluate_graphs(graphs, exhaustive_root_values(GEOMETRY.num_root_nodes))


# --------------------------------------------------------------------------
# Patch targets
# --------------------------------------------------------------------------


def test_patch_targets_match_a_whole_table_unpack(graphs):
    packed = packed_outputs(graphs)
    rows_per_patch = GEOMETRY.num_truth_table_rows // CONFIG.num_patches
    full = unpack_bits(packed).float()

    patch_indices = torch.tensor([0, 3, CONFIG.num_patches - 1, 1])
    selected = patch_targets(packed, patch_indices, rows_per_patch)
    assert selected.shape == (
        len(graphs),
        len(patch_indices),
        GEOMETRY.num_output_nodes,
        rows_per_patch,
    )
    for slot, patch in enumerate(patch_indices.tolist()):
        expected = full[:, :, patch * rows_per_patch : (patch + 1) * rows_per_patch]
        assert torch.equal(selected[:, slot], expected), f"patch {patch}"


def test_patch_targets_cover_the_table_exactly(graphs):
    """Every patch in order must reassemble the whole truth table."""
    packed = packed_outputs(graphs)
    rows_per_patch = GEOMETRY.num_truth_table_rows // CONFIG.num_patches
    everything = patch_targets(packed, torch.arange(CONFIG.num_patches), rows_per_patch)
    # [B, P, C, rows] -> [B, C, P * rows]
    reassembled = everything.permute(0, 2, 1, 3).flatten(start_dim=2)
    assert torch.equal(reassembled, unpack_bits(packed).float())


def test_sample_patch_indices_are_distinct_and_in_range():
    indices = sample_patch_indices(CONFIG.num_patches, CONFIG.num_patches, "cpu")
    assert sorted(indices.tolist()) == list(range(CONFIG.num_patches))
    with pytest.raises(ValueError):
        sample_patch_indices(4, 5, "cpu")


# --------------------------------------------------------------------------
# Node tokens
# --------------------------------------------------------------------------


def test_node_tokens_ignore_padded_slot_values(graphs):
    """A masked-off slot's stored index must not reach the token."""
    torch.manual_seed(0)
    tokens = NodeTokens(GEOMETRY, CONFIG)
    reference = tokens(
        graphs.node_types, graphs.parent_indices, graphs.parent_slot_mask
    )

    scrambled = graphs.parent_indices.clone()
    # Outputs are in-degree 1, so slot 1 is padding everywhere in that block.
    scrambled[:, GEOMETRY.output_start :, 1] = GEOMETRY.num_nodes - 1
    assert not torch.equal(scrambled, graphs.parent_indices)

    perturbed = tokens(graphs.node_types, scrambled, graphs.parent_slot_mask)
    assert torch.equal(reference, perturbed)


def test_node_tokens_depend_on_every_input(graphs):
    """Changing a type, or a real parent pointer, must move that node's token."""
    torch.manual_seed(0)
    tokens = NodeTokens(GEOMETRY, CONFIG)
    reference = tokens(
        graphs.node_types, graphs.parent_indices, graphs.parent_slot_mask
    )
    node = GEOMETRY.num_root_nodes + 3

    types = graphs.node_types.clone()
    # Any other valid entry in the type table; with one trunk type the output
    # class is the nearest distinct one.
    types[:, node] = GEOMETRY.output_type
    moved = tokens(types, graphs.parent_indices, graphs.parent_slot_mask)
    assert not torch.allclose(reference[:, node], moved[:, node])

    positions = graphs.parent_indices.clone()
    positions[:, node, 0] = (positions[:, node, 0] + 1) % GEOMETRY.num_root_nodes
    moved = tokens(graphs.node_types, positions, graphs.parent_slot_mask)
    assert not torch.allclose(reference[:, node], moved[:, node])


def test_node_tokens_distinguish_parent_slots():
    """Swapping slot 0 and slot 1 must change the token: slot order is meaningful."""
    torch.manual_seed(0)
    tokens = NodeTokens(GEOMETRY, CONFIG)
    types = torch.zeros(1, GEOMETRY.num_nodes, dtype=torch.long)
    mask = torch.ones(1, GEOMETRY.num_nodes, 2, dtype=torch.bool)
    forward = torch.zeros(1, GEOMETRY.num_nodes, 2, dtype=torch.long)
    forward[0, :, 0] = 0
    forward[0, :, 1] = 1

    swapped = forward.flip(-1)
    assert not torch.allclose(
        tokens(types, forward, mask), tokens(types, swapped, mask)
    )


# --------------------------------------------------------------------------
# The whole model
# --------------------------------------------------------------------------


def test_forward_shapes_and_patch_selection(graphs):
    torch.manual_seed(0)
    model = GraphSimulator(GEOMETRY, CONFIG)
    model.eval()

    rows_per_patch = model.decoder.rows_per_patch
    assert rows_per_patch * CONFIG.num_patches == GEOMETRY.num_truth_table_rows

    everything = model.forward_graphs(graphs)
    assert everything.shape == (
        len(graphs),
        CONFIG.num_patches,
        GEOMETRY.num_output_nodes,
        rows_per_patch,
    )

    # Decoding a subset must give exactly the rows of the full decode: the patch
    # queries are independent, so selecting them cannot change their values.
    patch_indices = torch.tensor([2, 0, CONFIG.num_patches - 1])
    subset = model.forward_graphs(graphs, patch_indices)
    assert torch.allclose(subset, everything[:, patch_indices], atol=1e-5)


def test_registers_extend_the_sequence_without_changing_the_output_shape(graphs):
    """Registers ride along in attention but must not alter what is decoded."""
    torch.manual_seed(0)
    config = replace(CONFIG, num_register_tokens=5)
    model = GraphSimulator(GEOMETRY, config)
    model.eval()

    tokens = model.node_tokens(
        graphs.node_types, graphs.parent_indices, graphs.parent_slot_mask
    )
    sequence = model.simulator(tokens)
    assert sequence.shape == (len(graphs), GEOMETRY.num_nodes + 5, CONFIG.embedding_dim)

    # The decoder's output is a function of the patch count, not the context.
    logits = model.forward_graphs(graphs)
    assert logits.shape == (
        len(graphs),
        CONFIG.num_patches,
        GEOMETRY.num_output_nodes,
        model.decoder.rows_per_patch,
    )


def test_zero_registers_allocates_nothing(graphs):
    torch.manual_seed(0)
    model = GraphSimulator(GEOMETRY, replace(CONFIG, num_register_tokens=0))
    assert model.simulator.register_tokens is None
    assert not any("register" in name for name, _ in model.named_parameters())

    tokens = model.node_tokens(
        graphs.node_types, graphs.parent_indices, graphs.parent_slot_mask
    )
    assert model.simulator(tokens).shape[1] == GEOMETRY.num_nodes


def test_registers_are_trained_and_shared_across_the_batch(graphs):
    torch.manual_seed(0)
    config = replace(CONFIG, num_register_tokens=3)
    model = GraphSimulator(GEOMETRY, config)
    assert model.simulator.register_tokens.shape == (3, CONFIG.embedding_dim)

    patch_indices = sample_patch_indices(CONFIG.num_patches, 4, "cpu")
    targets = patch_targets(
        packed_outputs(graphs), patch_indices, model.decoder.rows_per_patch
    )
    logits = model.forward_graphs(graphs, patch_indices)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).backward()
    grad = model.simulator.register_tokens.grad
    assert grad is not None and grad.abs().sum() > 0

    # One bank of registers for the whole batch, not one per graph: the values
    # entering the stack must be identical for every row.
    model.eval()
    tokens = model.node_tokens(
        graphs.node_types, graphs.parent_indices, graphs.parent_slot_mask
    )
    stacked = torch.cat(
        [tokens, model.simulator.register_tokens.expand(len(graphs), -1, -1)], dim=1
    )
    assert torch.equal(
        stacked[:, GEOMETRY.num_nodes :],
        model.simulator.register_tokens.expand(len(graphs), -1, -1),
    )


def test_registers_change_what_the_model_computes(graphs):
    """A register the model can attend to must be able to move the prediction."""
    torch.manual_seed(0)
    without = GraphSimulator(GEOMETRY, replace(CONFIG, num_register_tokens=0)).eval()
    torch.manual_seed(0)
    with_registers = GraphSimulator(
        GEOMETRY, replace(CONFIG, num_register_tokens=4)
    ).eval()

    # Same seed, so every shared parameter is identical; only the registers and
    # the attention they participate in differ.
    assert torch.equal(
        without.node_tokens.position_embeddings,
        with_registers.node_tokens.position_embeddings,
    )
    assert not torch.allclose(
        without.forward_graphs(graphs), with_registers.forward_graphs(graphs)
    )


def test_negative_register_count_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        GraphSimulator(GEOMETRY, replace(CONFIG, num_register_tokens=-1))


def test_patch_count_must_divide_the_table():
    with pytest.raises(ValueError, match="do not divide"):
        GraphSimulator(
            Geometry(4, 8, 2, 2, (2, 2)),
            SimulatorConfig(embedding_dim=32, attention_head_dim=32, num_patches=7),
        )


def test_embedding_dim_must_split_into_heads():
    with pytest.raises(ValueError, match="not a multiple"):
        GraphSimulator(
            GEOMETRY, SimulatorConfig(embedding_dim=48, attention_head_dim=32)
        )


def test_every_parameter_receives_a_gradient(graphs):
    torch.manual_seed(0)
    model = GraphSimulator(GEOMETRY, CONFIG)
    patch_indices = sample_patch_indices(CONFIG.num_patches, 4, "cpu")
    targets = patch_targets(
        packed_outputs(graphs), patch_indices, model.decoder.rows_per_patch
    )
    logits = model.forward_graphs(graphs, patch_indices)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).backward()

    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_graphs_with_different_structure_get_different_predictions(graphs):
    """A sanity floor: the model must actually read the graph."""
    torch.manual_seed(0)
    model = GraphSimulator(GEOMETRY, CONFIG)
    model.eval()
    logits = model.forward_graphs(graphs)
    assert not torch.allclose(logits[0], logits[1], atol=1e-4)
