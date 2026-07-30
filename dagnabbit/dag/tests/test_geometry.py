"""Tests for :class:`~dagnabbit.dag.geometry.GraphGeometry`.

This used to also cover a ``GraphBatchLoader`` that generated batches in spawned
worker processes. That was removed once generation got fast enough that there was
nothing worth hiding behind the optimizer step, so what remains is the geometry
value object and the batch-sampling call. The pickling tests are kept: nothing
ships descriptions across a process boundary today, but they are cheap and they
pin the property that made the flat rank partition worth having.
"""

import pickle

import pytest
import torch

from dagnabbit.dag.description import FixedInDegreeDAGDescription
from dagnabbit.dag.geometry import GraphGeometry

GEOMETRY = GraphGeometry(
    num_root_nodes=4,
    num_trunk_nodes=12,
    num_output_nodes=2,
    num_trunk_node_types=2,
    trunk_node_in_degrees=(2, 2),
)

RANK_TENSOR_NAMES = ("node_indices", "parent_indices", "valid_parent_mask", "subtypes")
PARTITION_TENSOR_NAMES = (*RANK_TENSOR_NAMES, "rank_of_row")
CANONICAL_TENSOR_NAMES = (
    "node_types_tensor",
    "canonical_positions_tensor",
    "canonical_order_tensor",
    "canonical_node_types",
    "canonical_parent_positions",
    "canonical_parent_slot_mask",
)


def assert_descriptions_identical(left, right) -> None:
    """Every tensor a consumer reads must survive a round trip untouched."""
    assert left.node_types == right.node_types
    assert left.node_inputs_indices == right.node_inputs_indices
    assert left.num_nodes == right.num_nodes

    for name in CANONICAL_TENSOR_NAMES:
        assert torch.equal(getattr(left, name), getattr(right, name)), name

    for name in PARTITION_TENSOR_NAMES:
        assert torch.equal(
            getattr(left.rank_partition, name), getattr(right.rank_partition, name)
        ), name
    assert left.rank_partition.counts == right.rank_partition.counts
    assert (
        left.rank_partition.has_valid_parents == right.rank_partition.has_valid_parents
    )


# --- pickling ---------------------------------------------------------------


def test_a_pickled_description_is_unchanged() -> None:
    graph = GEOMETRY.sample_batch(1)[0]
    assert_descriptions_identical(graph, pickle.loads(pickle.dumps(graph)))


def test_pickling_does_not_duplicate_rank_storage() -> None:
    """The historical failure this representation was chosen to avoid.

    ``rank_batches`` holds *slices* of the flat ``rank_partition`` tensors, and
    pickling a slice serializes its whole underlying storage -- so shipping the
    per-rank views wrote each base tensor once per rank. Measured at the real
    training geometry that was 111 KB per graph against ~13 KB of actual data, an
    8.7x blowup, which is what made background workers a 0.7x regression against
    inline generation back when there were background workers.

    The views are materialized here on purpose, since the live risk is a future
    change putting them back into the pickled state.

    Deliberately measured at the *training* geometry rather than this module's
    tiny one: with 18 nodes and a handful of ranks, per-object pickle framing
    dominates the buffers and the fixed and broken cases are only a factor apart.
    At 152 nodes and ~15 ranks they differ by nearly an order of magnitude, which
    is what makes the bound below meaningful.
    """
    from dagnabbit.scripts import config as cfg

    graph = GraphGeometry.from_config(cfg).sample_batch(1)[0]
    num_ranks = len(graph.rank_batches)
    payload = len(pickle.dumps(graph))

    live_bytes = sum(
        getattr(graph, name).numel() * getattr(graph, name).element_size()
        for name in CANONICAL_TENSOR_NAMES
    )
    # The flat rank bases, counted once each, which is what should travel.
    live_bytes += sum(
        getattr(graph.rank_partition, name).numel()
        * getattr(graph.rank_partition, name).element_size()
        for name in PARTITION_TENSOR_NAMES
    )
    assert num_ranks >= 5, "geometry too shallow for this test to mean anything"
    # Duplication would multiply the rank bases by num_ranks; staying under 3x
    # total sits far below that and comfortably above the framing overhead.
    assert payload < live_bytes * 3, (
        f"pickle is {payload} bytes against {live_bytes} of live tensor data "
        f"across {num_ranks} ranks; rank slices are probably duplicating their "
        "base storage again"
    )


def test_pickling_ships_arrays_and_drops_the_views() -> None:
    """The arrays are the primitive state; the ragged views derive from them.

    Dropping an array instead would be a live bug rather than a size regression:
    for a description built by ``from_arrays`` the ragged views derive from the
    arrays, so losing one leaves the pair mutually recursive and any access
    blows the stack.
    """
    graph = GEOMETRY.sample_batch(1)[0]
    # Force every lazy view to materialize, so the pickle has something to drop.
    assert len(graph.rank_batches) > 1
    assert graph.node_inputs_indices is not None
    assert graph.node_ranks is not None
    assert graph.canonical_order is not None
    assert graph.leaf_node_indices is not None

    restored = pickle.loads(pickle.dumps(graph))

    for name in FixedInDegreeDAGDescription._PICKLED_ARRAYS:
        assert name in restored.__dict__, f"{name} must survive; views need it"
    for name in FixedInDegreeDAGDescription._DROPPED_ON_PICKLE:
        assert name not in restored.__dict__, f"{name} is derivable and should go"

    # Every dropped view is still reachable, just rebuilt on demand.
    assert restored.node_inputs_indices == graph.node_inputs_indices
    assert restored.node_types == graph.node_types
    assert restored.node_ranks == graph.node_ranks
    assert restored.canonical_order == graph.canonical_order
    assert restored.canonical_positions == graph.canonical_positions
    assert restored.leaf_node_indices == graph.leaf_node_indices
    assert len(restored.rank_batches) == len(graph.rank_batches)


def test_geometry_is_picklable_and_frozen() -> None:
    assert pickle.loads(pickle.dumps(GEOMETRY)) == GEOMETRY
    with pytest.raises(Exception):
        GEOMETRY.num_root_nodes = 99  # type: ignore[misc]


# --- geometry ---------------------------------------------------------------


def test_geometry_from_config_matches_a_directly_sampled_graph() -> None:
    from dagnabbit.scripts import config as cfg

    geometry = GraphGeometry.from_config(cfg)
    graph = geometry.sample_batch(1)[0]
    assert graph.num_root_nodes == cfg.NUM_ROOT_NODES
    assert graph.num_trunk_nodes == cfg.NUM_TRUNK_NODES
    assert graph.num_output_nodes == cfg.NUM_OUTPUT_NODES


def test_geometry_expands_a_scalar_in_degree() -> None:
    """config.py allows an int; the geometry normalizes to per-type."""

    class FakeConfig:
        NUM_ROOT_NODES = 4
        NUM_TRUNK_NODES = 8
        NUM_OUTPUT_NODES = 2
        NUM_TRUNK_NODE_TYPES = 3
        TRUNK_NODE_TYPE_IN_DEGREES = 2

    geometry = GraphGeometry.from_config(FakeConfig)
    assert geometry.trunk_node_in_degrees == (2, 2, 2)


def test_sample_batch_returns_the_requested_count_of_distinct_graphs() -> None:
    batch = GEOMETRY.sample_batch(8)
    assert len(batch) == 8
    signatures = {tuple(graph.node_types) for graph in batch}
    assert len(signatures) > 1, "sampling should not return identical graphs"
