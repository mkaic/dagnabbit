"""Tests for the compiled generator in :mod:`dagnabbit.dag.generate`.

The compiled kernels reimplement the sampler *and* two derived overlays that the
Python implementation in :mod:`dagnabbit.dag.description` also computes, so the
risk here is silent divergence rather than a crash. Three things are checked:

1. **Exactness of the overlays.** Given a structure, the compiled ``ranks`` and
   ``canonical_order`` must be bit-identical to :meth:`compute_node_ranks` and
   :meth:`compute_canonical_order`. This is not a nicety: the canonical order
   defines the sequence model's targets, and the two implementations order the
   Kahn frontier with independently written comparisons.
2. **Equivalence of the two construction paths.** A description built by
   ``from_arrays`` must be indistinguishable from one built from ragged lists,
   including the lazily derived views.
3. **The sampled distribution**, statistically, plus the constraints every
   sampled graph must satisfy.
"""

import numpy as np
import pytest
import torch

from dagnabbit.dag.description import FixedInDegreeDAGDescription
from dagnabbit.dag.generate import generate_arrays, sample_graph_batch

# (roots, trunk, outputs, num_types, in_degrees), all satisfying the two
# generation preconditions: roots >= max in-degree, and
# roots <= trunk * (min in-degree - 1) + outputs.
GEOMETRIES = [
    pytest.param(1, 4, 2, 1, 1, id="in-degree-1"),
    pytest.param(4, 8, 2, 1, 4, id="in-degree-equals-roots-forces-rejections"),
    pytest.param(4, 6, 2, 3, [2, 3, 4], id="mixed-in-degrees-ragged"),
    pytest.param(16, 128, 8, 2, 2, id="training-geometry"),
    pytest.param(3, 1, 3, 1, 2, id="single-trunk-node"),
    pytest.param(2, 2, 4, 2, [1, 2], id="mixed-with-in-degree-1"),
]


def normalize(num_types, in_degrees):
    return [in_degrees] * num_types if isinstance(in_degrees, int) else list(in_degrees)


def raw_arrays(count, roots, trunk, outputs, num_types, in_degrees, seed):
    degrees = normalize(num_types, in_degrees)
    return generate_arrays(
        count,
        roots,
        trunk,
        outputs,
        num_types,
        np.asarray(degrees, dtype=np.int64),
        max([1, *degrees]),
        seed,
    )


@pytest.mark.parametrize("roots,trunk,outputs,num_types,in_degrees", GEOMETRIES)
def test_compiled_overlays_match_the_python_implementation(
    roots, trunk, outputs, num_types, in_degrees
) -> None:
    """Ranks and canonical order must agree exactly, not approximately.

    The comparison is done by handing the *compiled* structure to the Python
    class and letting it derive the overlays itself, so any disagreement is a
    disagreement between the two algorithms rather than between two samplers.
    """
    degrees = normalize(num_types, in_degrees)
    count = 40
    node_types, in_degs, parents, ranks, order = raw_arrays(
        count, roots, trunk, outputs, num_types, in_degrees, 17
    )
    num_nodes = roots + trunk + outputs

    for index in range(count):
        inputs = [
            parents[index, node, : in_degs[index, node]].tolist()
            for node in range(num_nodes)
        ]
        reference = FixedInDegreeDAGDescription(
            num_root_nodes=roots,
            num_trunk_nodes=trunk,
            num_output_nodes=outputs,
            num_trunk_node_types=num_types,
            trunk_node_in_degrees=degrees,
            node_inputs_indices=inputs,
            node_types=node_types[index].tolist(),
        )
        assert reference.node_ranks == ranks[index].tolist(), f"ranks, graph {index}"
        assert reference.canonical_order == order[index].tolist(), (
            f"canonical_order, graph {index}"
        )


@pytest.mark.parametrize("roots,trunk,outputs,num_types,in_degrees", GEOMETRIES)
def test_the_two_construction_paths_agree(
    roots, trunk, outputs, num_types, in_degrees
) -> None:
    """``from_arrays`` and ``__init__`` must produce indistinguishable objects."""
    degrees = normalize(num_types, in_degrees)
    torch.manual_seed(4)
    compiled = sample_graph_batch(
        6,
        num_root_nodes=roots,
        num_trunk_nodes=trunk,
        num_output_nodes=outputs,
        num_trunk_node_types=num_types,
        trunk_node_in_degrees=in_degrees,
    )

    for graph in compiled:
        reference = FixedInDegreeDAGDescription(
            num_root_nodes=roots,
            num_trunk_nodes=trunk,
            num_output_nodes=outputs,
            num_trunk_node_types=num_types,
            trunk_node_in_degrees=degrees,
            node_inputs_indices=graph.node_inputs_indices,
            node_types=graph.node_types,
        )

        # The ragged views, which the compiled path derives rather than receives.
        assert graph.node_inputs_indices == reference.node_inputs_indices
        assert graph.node_types == reference.node_types
        assert graph.node_ranks == reference.node_ranks
        assert graph.canonical_order == reference.canonical_order
        assert graph.canonical_positions == reference.canonical_positions
        assert graph.leaf_node_indices == reference.leaf_node_indices

        for name in (
            "node_types_tensor",
            "leaf_node_indices_tensor",
            "canonical_positions_tensor",
            "canonical_order_tensor",
            "canonical_node_types",
            "canonical_parent_positions",
            "canonical_parent_slot_mask",
        ):
            assert torch.equal(getattr(graph, name), getattr(reference, name)), name

        for name in (
            "node_indices",
            "parent_indices",
            "valid_parent_mask",
            "subtypes",
            "rank_of_row",
        ):
            assert torch.equal(
                getattr(graph.rank_partition, name),
                getattr(reference.rank_partition, name),
            ), name
        assert graph.rank_partition.counts == reference.rank_partition.counts
        assert (
            graph.rank_partition.has_valid_parents
            == reference.rank_partition.has_valid_parents
        )


@pytest.mark.parametrize("roots,trunk,outputs,num_types,in_degrees", GEOMETRIES)
def test_sampled_graphs_satisfy_every_constraint(
    roots, trunk, outputs, num_types, in_degrees
) -> None:
    degrees = normalize(num_types, in_degrees)
    torch.manual_seed(0)
    output_start = roots + trunk

    for graph in sample_graph_batch(
        40,
        num_root_nodes=roots,
        num_trunk_nodes=trunk,
        num_output_nodes=outputs,
        num_trunk_node_types=num_types,
        trunk_node_in_degrees=in_degrees,
    ):
        assert graph.num_nodes == roots + trunk + outputs
        for node in range(roots):
            assert graph.node_inputs_indices[node] == []
            assert graph.node_types[node] == num_types + node
        for node in range(roots, output_start):
            parents = graph.node_inputs_indices[node]
            assert len(parents) == degrees[graph.node_types[node]]
            assert all(parent < node for parent in parents)
            assert len(set(parents)) == len(parents)
        for node in range(output_start, graph.num_nodes):
            parents = graph.node_inputs_indices[node]
            assert len(parents) == 1
            assert parents[0] < output_start
            assert graph.node_types[node] == num_types + roots

        # The coverage pass exists to make this true: no producer is orphaned.
        referenced = {
            parent for parents in graph.node_inputs_indices for parent in parents
        }
        assert referenced >= set(range(output_start))


def test_generation_is_reproducible_and_uncorrelated() -> None:
    """numba keeps its own RNG state, so seeding is easy to get wrong.

    Identical batches from step to step would silently destroy the premise of
    the whole setup -- that no example is ever seen twice -- while throughput
    looked fine.
    """
    geometry = dict(
        num_root_nodes=8,
        num_trunk_nodes=32,
        num_output_nodes=4,
        num_trunk_node_types=2,
        trunk_node_in_degrees=2,
    )

    torch.manual_seed(1234)
    first = sample_graph_batch(16, **geometry)
    torch.manual_seed(1234)
    replay = sample_graph_batch(16, **geometry)
    for left, right in zip(first, replay):
        assert left.node_types == right.node_types
        assert left.node_inputs_indices == right.node_inputs_indices

    # Consecutive batches under one seed must differ from each other, and from
    # the first batch: the generator is reseeded per call from torch.
    torch.manual_seed(99)
    batches = [sample_graph_batch(8, **geometry) for _ in range(4)]
    signatures = {
        (tuple(graph.node_types), tuple(map(tuple, graph.node_inputs_indices)))
        for batch in batches
        for graph in batch
    }
    assert len(signatures) == 32, (
        f"only {len(signatures)} distinct graphs out of 32; the compiled "
        "generator is probably being reseeded identically each call"
    )


def test_sampled_distribution_is_unbiased() -> None:
    """Cheap statistics over the sampler's prior, at the training geometry.

    ``slot0_earlier`` is the one that catches a broken shuffle: both wiring
    passes append, so without the Fisher-Yates pass slot 0 would always hold the
    coverage parent and the share would sit far from a half.
    """
    torch.manual_seed(5)
    roots, trunk, outputs = 16, 128, 8
    graphs = sample_graph_batch(
        600,
        num_root_nodes=roots,
        num_trunk_nodes=trunk,
        num_output_nodes=outputs,
        num_trunk_node_types=2,
        trunk_node_in_degrees=2,
    )

    out_degrees = np.zeros(roots + trunk, dtype=np.int64)
    types = np.zeros(2, dtype=np.int64)
    ascending = 0
    pairs = 0
    for graph in graphs:
        for parents in graph.node_inputs_indices:
            for parent in parents:
                out_degrees[parent] += 1
            if len(parents) == 2:
                ascending += parents[0] < parents[1]
                pairs += 1
        types += np.bincount(graph.node_types[roots : roots + trunk], minlength=2)

    # Every slot is filled exactly once, so the mean out-degree is fixed by the
    # counts: (trunk * 2 + outputs) slots spread over (roots + trunk) producers.
    expected_mean = (trunk * 2 + outputs) / (roots + trunk)
    assert abs(out_degrees.mean() / len(graphs) - expected_mean) < 1e-9
    assert (out_degrees > 0).all(), "some producer was never used in any graph"

    type_share = types / types.sum()
    assert abs(type_share[0] - 0.5) < 0.02, (
        f"trunk type draw looks biased: {type_share}"
    )

    share = ascending / pairs
    assert abs(share - 0.5) < 0.02, f"slot order looks biased: {share:.4f} ascending"
