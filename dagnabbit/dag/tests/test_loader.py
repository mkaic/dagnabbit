"""Tests for background graph-batch generation.

Every interesting failure mode here is silent. Identically-seeded workers still
produce batches at full speed while destroying the uncorrelated-data property
that in-loop generation exists to provide; a lossy pickle still trains, just on
subtly wrong graphs. So the assertions are about *equivalence* and *distinctness*
rather than about throughput, which belongs in
:mod:`dagnabbit.scripts.profile_batch_loader` instead.

The multi-process tests are deliberately small -- spawning a worker re-imports
torch and costs seconds -- and use one or two workers with tiny graphs.
"""

import pickle

import pytest
import torch

from dagnabbit.dag.description import graphs_match, make_random_graph_description
from dagnabbit.dag.loader import GraphBatchLoader, GraphGeometry, default_num_workers

GEOMETRY = GraphGeometry(
    num_root_nodes=4,
    num_trunk_nodes=12,
    num_output_nodes=2,
    num_trunk_node_types=2,
    trunk_node_in_degrees=(2, 2),
)

RANK_TENSOR_NAMES = ("node_indices", "parent_indices", "valid_parent_mask", "subtypes")
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

    assert len(left.rank_batches) == len(right.rank_batches)
    for index, (rank_left, rank_right) in enumerate(
        zip(left.rank_batches, right.rank_batches)
    ):
        for name in RANK_TENSOR_NAMES:
            assert torch.equal(getattr(rank_left, name), getattr(rank_right, name)), (
                f"rank {index} {name}"
            )
        assert rank_left.has_valid_parents == rank_right.has_valid_parents


# --- pickling ---------------------------------------------------------------


def test_a_pickled_description_is_unchanged() -> None:
    """The custom __getstate__ rebuilds rank slices; they must match exactly."""
    graph = GEOMETRY.sample_batch(1)[0]
    assert_descriptions_identical(graph, pickle.loads(pickle.dumps(graph)))


def test_pickling_does_not_duplicate_rank_storage() -> None:
    """Why __getstate__ exists at all.

    ``rank_batches`` holds *slices* of four flat tensors. Pickling a slice
    serializes its whole underlying storage, so the naive pickle wrote each base
    tensor once per rank. Measured at the real training geometry that was 111 KB
    per graph against ~13 KB of actual data -- an 8.7x blowup that made
    background workers a 0.7x regression against inline generation.

    Deliberately measured at the *training* geometry rather than this module's
    tiny one: with 18 nodes and a handful of ranks, per-object pickle framing
    dominates the buffers and the fixed and broken cases are only a factor apart.
    At 152 nodes and ~15 ranks they differ by nearly an order of magnitude, which
    is what makes the bound below meaningful.
    """
    from dagnabbit.scripts import config as cfg

    graph = GraphGeometry.from_config(cfg).sample_batch(1)[0]
    payload = len(pickle.dumps(graph))

    live_bytes = sum(
        getattr(graph, name).numel() * getattr(graph, name).element_size()
        for name in CANONICAL_TENSOR_NAMES
    )
    # The four flat rank bases, counted once each, which is what should travel.
    live_bytes += sum(
        getattr(rank, name).numel() * getattr(rank, name).element_size()
        for rank in graph.rank_batches
        for name in RANK_TENSOR_NAMES
    )
    num_ranks = len(graph.rank_batches)
    assert num_ranks >= 5, "geometry too shallow for this test to mean anything"
    # Duplication would multiply the rank bases by num_ranks; staying under 3x
    # total sits far below that and comfortably above the framing overhead.
    assert payload < live_bytes * 3, (
        f"pickle is {payload} bytes against {live_bytes} of live tensor data "
        f"across {num_ranks} ranks; rank slices are probably duplicating their "
        "base storage again"
    )


def test_pickling_drops_the_rank_groups_cache() -> None:
    """A materialized cached_property is pure IPC weight; nothing hot reads it."""
    graph = GEOMETRY.sample_batch(1)[0]
    assert graph.rank_groups is not None  # force the cached_property
    assert "rank_groups" in graph.__dict__
    restored = pickle.loads(pickle.dumps(graph))
    assert "rank_groups" not in restored.__dict__
    # Still reachable, just rebuilt on demand.
    assert len(restored.rank_groups) == len(graph.rank_groups)


def test_geometry_is_picklable_and_frozen() -> None:
    """It crosses the spawn boundary, so it has to survive pickling."""
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
    """config.py allows an int; workers need the expanded per-type list."""

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


# --- the synchronous path ---------------------------------------------------


def test_zero_workers_generates_inline() -> None:
    with GraphBatchLoader(GEOMETRY, batch_size=5, num_workers=0) as loader:
        assert len(loader.next_batch()) == 5
        assert "synchronous" in loader.describe()


def test_loader_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        GraphBatchLoader(GEOMETRY, batch_size=0)
    with pytest.raises(ValueError, match="num_workers must be non-negative"):
        GraphBatchLoader(GEOMETRY, batch_size=4, num_workers=-1)
    with pytest.raises(ValueError, match="prefetch_batches must be positive"):
        GraphBatchLoader(GEOMETRY, batch_size=4, num_workers=0, prefetch_batches=0)


def test_default_num_workers_leaves_the_trainer_a_core() -> None:
    import os

    workers = default_num_workers()
    assert 1 <= workers <= 8
    assert workers < (os.cpu_count() or 2) or (os.cpu_count() or 2) <= 2


# --- the background path ----------------------------------------------------


@pytest.mark.slow
def test_background_batches_are_well_formed_graphs() -> None:
    """A worker-built graph must be indistinguishable from a locally built one.

    Compared structurally rather than by identity: the point is that a graph
    which crossed a process boundary is still a valid, complete description that
    ``graphs_match`` can reason about.
    """
    with GraphBatchLoader(GEOMETRY, batch_size=4, num_workers=1, seed=3) as loader:
        batch = loader.next_batch()

    assert len(batch) == 4
    local = make_random_graph_description(
        num_root_nodes=GEOMETRY.num_root_nodes,
        num_trunk_nodes=GEOMETRY.num_trunk_nodes,
        num_output_nodes=GEOMETRY.num_output_nodes,
        trunk_node_in_degrees=list(GEOMETRY.trunk_node_in_degrees),
        num_trunk_node_types=GEOMETRY.num_trunk_node_types,
    )
    for graph in batch:
        assert graph.num_nodes == local.num_nodes
        assert len(graph.rank_batches) >= 1
        # graphs_match is the real structural comparator; it must not choke on a
        # round-tripped description.
        assert isinstance(graphs_match(graph, local), bool)
        assert graphs_match(graph, graph)


@pytest.mark.slow
def test_workers_do_not_generate_correlated_streams() -> None:
    """The silent failure that would void the whole point of in-loop generation.

    Identically-seeded workers would emit identical graph streams at full
    throughput. With two workers and several batches, near-total distinctness is
    the expectation.
    """
    with GraphBatchLoader(GEOMETRY, batch_size=4, num_workers=2, seed=5) as loader:
        graphs = [graph for _ in range(6) for graph in loader.next_batch()]

    signatures = {tuple(graph.node_types) for graph in graphs}
    assert len(signatures) > len(graphs) // 2, (
        f"only {len(signatures)} distinct graphs out of {len(graphs)}; workers "
        "are probably sharing a seed"
    )


@pytest.mark.slow
def test_closing_is_idempotent_and_blocks_further_use() -> None:
    loader = GraphBatchLoader(GEOMETRY, batch_size=2, num_workers=1, seed=7)
    loader.next_batch()
    loader.close()
    loader.close()  # must not raise
    with pytest.raises(RuntimeError, match="closed"):
        loader.next_batch()


@pytest.mark.slow
def test_workers_are_reaped_on_context_exit() -> None:
    """A leaked worker keeps generating graphs nobody reads."""
    with GraphBatchLoader(GEOMETRY, batch_size=2, num_workers=2, seed=9) as loader:
        loader.next_batch()
        processes = list(loader._processes)
        assert all(process.is_alive() for process in processes)

    for process in processes:
        process.join(timeout=10)
        assert not process.is_alive(), "worker survived loader shutdown"
