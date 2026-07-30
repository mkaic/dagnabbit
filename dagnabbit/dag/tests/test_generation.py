"""Tests for the derived-array overlays on a graph description.

Every overlay the model reads -- the padded parent view, the leaf set, the rank
partition, the canonical sequence tensors -- is a vectorized gather rather than a
Python loop over nodes, and the batch collator regroups a whole batch by rank
with a single stable argsort. Both are easy to get subtly wrong in ways that
still train, just on quietly wrong graphs, so these compare against the ragged
Python lists and against the nested loop the collator replaced.

Sampling itself is tested in ``test_generate.py``, against the compiled
generator that now performs it.
"""

import pytest
import torch

from dagnabbit.dag.description import (
    collate_rank_partitions,
    make_random_graph_description,
)

# (roots, trunk, outputs, num_types, in_degrees) -- see the module docstring for
# what each one is here to catch. All satisfy the two generation preconditions:
# roots >= max in-degree, and roots <= trunk * (min in-degree - 1) + outputs,
# which is why an in-degree of 1 forces a small root count.
GEOMETRIES = [
    pytest.param(1, 4, 2, 1, 1, id="in-degree-1-no-shuffle-draws"),
    pytest.param(4, 8, 2, 1, 4, id="in-degree-equals-roots-forces-rejections"),
    pytest.param(4, 6, 2, 3, [2, 3, 4], id="mixed-in-degrees-ragged"),
    pytest.param(16, 128, 8, 2, 2, id="training-geometry"),
    pytest.param(3, 1, 3, 1, 2, id="single-trunk-node"),
    pytest.param(2, 2, 4, 2, [1, 2], id="mixed-with-in-degree-1"),
]


def build(roots, trunk, outputs, num_types, in_degrees):
    return make_random_graph_description(
        num_root_nodes=roots,
        num_trunk_nodes=trunk,
        num_output_nodes=outputs,
        num_trunk_node_types=num_types,
        trunk_node_in_degrees=in_degrees,
    )


@pytest.mark.parametrize("roots,trunk,outputs,num_types,in_degrees", GEOMETRIES)
def test_derived_arrays_agree_with_the_ragged_source(
    roots, trunk, outputs, num_types, in_degrees
) -> None:
    """Every vectorized overlay must still say what the Python lists say.

    ``parent_arrays`` is the single padded numpy view the rank batches, the leaf
    set and the canonical tensors are all gathered from, so a padding or masking
    error there corrupts the model's inputs silently.
    """
    torch.manual_seed(1)
    graph = build(roots, trunk, outputs, num_types, in_degrees)
    padded, slot_mask = graph.parent_arrays

    assert padded.shape == (graph.num_nodes, graph.maximum_indegree)
    assert slot_mask.shape == padded.shape
    for node_idx, parents in enumerate(graph.node_inputs_indices):
        in_degree = len(parents)
        assert padded[node_idx, :in_degree].tolist() == parents
        assert slot_mask[node_idx].tolist() == [
            slot < in_degree for slot in range(graph.maximum_indegree)
        ]
        # Padding must not look like a real edge to anything reading values.
        assert not padded[node_idx, in_degree:].any()

    # Leaves: nodes nobody references.
    referenced = {parent for parents in graph.node_inputs_indices for parent in parents}
    expected_leaves = [n for n in range(graph.num_nodes) if n not in referenced]
    assert graph.leaf_node_indices == expected_leaves

    # Rank batches partition the nodes, each slice ascending, grouped by rank.
    seen: list[int] = []
    for rank, batch in enumerate(graph.rank_batches):
        nodes = batch.node_indices.tolist()
        assert nodes == sorted(nodes)
        assert all(graph.node_ranks[node] == rank for node in nodes)
        assert batch.has_valid_parents == any(
            graph.node_inputs_indices[node] for node in nodes
        )
        for row, node in enumerate(nodes):
            parents = graph.node_inputs_indices[node]
            assert batch.parent_indices[row, : len(parents)].tolist() == parents
            assert batch.valid_parent_mask[row].tolist() == [
                slot < len(parents) for slot in range(graph.maximum_indegree)
            ]
            assert batch.subtypes[row].item() == graph.node_types[node]
        seen.extend(nodes)
    assert sorted(seen) == list(range(graph.num_nodes))

    # Canonical tensors, in sequence order rather than storage order.
    assert graph.canonical_order_tensor.tolist() == graph.canonical_order
    assert graph.canonical_node_types.tolist() == [
        graph.node_types[node] for node in graph.canonical_order
    ]
    for position, node in enumerate(graph.canonical_order):
        parents = graph.node_inputs_indices[node]
        expected = [graph.canonical_positions[parent] for parent in parents]
        assert graph.canonical_parent_positions[position, : len(parents)].tolist() == (
            expected
        )
        assert graph.canonical_parent_slot_mask[position].tolist() == [
            slot < len(parents) for slot in range(graph.maximum_indegree)
        ]


def test_tensor_dtypes_survive_the_numpy_route() -> None:
    """Built via ``torch.from_numpy`` now, which takes dtype from the array."""
    torch.manual_seed(2)
    graph = build(16, 32, 4, 2, 2)
    for name in (
        "node_types_tensor",
        "leaf_node_indices_tensor",
        "canonical_positions_tensor",
        "canonical_order_tensor",
        "canonical_node_types",
        "canonical_parent_positions",
    ):
        assert getattr(graph, name).dtype == torch.long, name
    assert graph.canonical_parent_slot_mask.dtype == torch.bool
    for batch in graph.rank_batches:
        assert batch.node_indices.dtype == torch.long
        assert batch.parent_indices.dtype == torch.long
        assert batch.subtypes.dtype == torch.long
        assert batch.valid_parent_mask.dtype == torch.bool


def reference_collate(graphs, device):
    """The nested loop ``collate_rank_partitions`` replaced, kept as the oracle.

    One ``torch.cat`` per field per rank over every graph's slice of that rank.
    Straightforward, and 3x slower than one stable argsort over the batch.
    """
    max_ranks = max(len(graph.rank_batches) for graph in graphs)
    per_rank = []
    for rank in range(max_ranks):
        fields = ([], [], [], [], [])
        for batch_index, graph in enumerate(graphs):
            if rank >= len(graph.rank_batches):
                continue
            batch = graph.rank_batches[rank]
            rows = batch.node_indices.shape[0]
            if not rows:
                continue
            fields[0].append(torch.full((rows,), batch_index, dtype=torch.long))
            fields[1].append(batch.node_indices)
            fields[2].append(batch.parent_indices)
            fields[3].append(batch.valid_parent_mask)
            fields[4].append(batch.subtypes)
        per_rank.append(tuple(torch.cat(f).to(device) for f in fields))
    return per_rank


@pytest.mark.parametrize("roots,trunk,outputs,num_types,in_degrees", GEOMETRIES)
def test_collate_matches_the_nested_loop_it_replaced(
    roots, trunk, outputs, num_types, in_degrees
) -> None:
    """Row *ordering* has to match, not just row content.

    Every field is indexed by the same row position, so a permutation applied to
    one and not another silently pairs a node with another node's parents. The
    stable argsort is supposed to reproduce the old loop exactly: rank-major,
    then graph order, then ascending node index.

    A batch of graphs with *different* depths is the interesting case, since then
    the deepest ranks draw rows from only some of the graphs.
    """
    torch.manual_seed(5)
    graphs = [build(roots, trunk, outputs, num_types, in_degrees) for _ in range(7)]
    device = torch.device("cpu")

    collated = collate_rank_partitions(graphs, device)
    expected = reference_collate(graphs, device)

    assert collated.num_ranks == len(expected)
    assert sum(collated.counts) == sum(graph.num_nodes for graph in graphs), (
        "every node of every graph must appear exactly once"
    )

    for rank, reference in enumerate(expected):
        rows = collated.rank_slice(rank)
        assert collated.counts[rank] == reference[0].shape[0], f"rank {rank} row count"
        for name, actual, want in zip(
            (
                "batch_indices",
                "node_indices",
                "parent_indices",
                "valid_parent_mask",
                "subtypes",
            ),
            (
                collated.batch_indices[rows],
                collated.node_indices[rows],
                collated.parent_indices[rows],
                collated.valid_parent_mask[rows],
                collated.subtypes[rows],
            ),
            reference,
        ):
            assert torch.equal(actual, want), f"rank {rank} {name}"

        # And the flag, which the old loop OR-ed across graphs.
        assert collated.has_valid_parents[rank] == any(
            graph.rank_batches[rank].has_valid_parents
            for graph in graphs
            if rank < len(graph.rank_batches)
        )


def test_collate_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_rank_partitions([], torch.device("cpu"))


def test_rank_batches_is_a_view_of_the_flat_partition() -> None:
    """The per-rank list is now a lazily built convenience over ``rank_partition``.

    It must agree with the flat form exactly, and it must not be part of the
    pickled state -- shipping a slice drags its whole base storage along, which is
    what made background workers a regression.
    """
    import pickle

    torch.manual_seed(6)
    graph = build(16, 128, 8, 2, 2)
    partition = graph.rank_partition

    assert "rank_batches" not in graph.__dict__, "should not be built eagerly"
    offset = 0
    for rank, batch in enumerate(graph.rank_batches):
        rows = slice(offset, offset + partition.counts[rank])
        offset += partition.counts[rank]
        assert torch.equal(batch.node_indices, partition.node_indices[rows])
        assert torch.equal(batch.parent_indices, partition.parent_indices[rows])
        assert torch.equal(batch.valid_parent_mask, partition.valid_parent_mask[rows])
        assert torch.equal(batch.subtypes, partition.subtypes[rows])
        assert batch.has_valid_parents == partition.has_valid_parents[rank]
    assert offset == graph.num_nodes

    assert "rank_batches" in graph.__dict__  # now materialized
    restored = pickle.loads(pickle.dumps(graph))
    assert "rank_batches" not in restored.__dict__
    assert torch.equal(restored.rank_partition.node_indices, partition.node_indices)
    assert restored.rank_partition.counts == partition.counts
