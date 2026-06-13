"""Regression checks for edge-splitting random DAG generation.

Run directly:

    uv run python -m dagnabbit.dag.tests.test_edge_split_generation
"""

import random
from collections import Counter, deque

import torch

from dagnabbit.dag.description import (
    _EdgeSplitGraph,
    _EdgeSplitRole,
    _assemble_edge_split_description,
    _build_interleaved_edge_split_graph,
    _description_from_flat_parent_arrays,
    _mcmc_rewire,
    _native,
    make_random_graph_description,
)


def _out_degrees(graph) -> list[int]:
    out_degree = [0] * graph.num_nodes
    for parents in graph.node_inputs_indices:
        for parent in parents:
            out_degree[parent] += 1
    return out_degree


def _assert_edge_split_invariants(graph) -> None:
    assert graph.num_nodes == (
        graph.num_root_nodes + graph.num_trunk_nodes + graph.num_output_nodes
    )

    out_degree = _out_degrees(graph)
    output_start = graph.num_root_nodes + graph.num_trunk_nodes

    for node_idx, parents in enumerate(graph.node_inputs_indices):
        for parent in parents:
            assert 0 <= parent < node_idx, (node_idx, parent)

    for node_idx in range(graph.num_root_nodes):
        assert graph.node_inputs_indices[node_idx] == []

    for node_idx in range(graph.num_root_nodes, output_start):
        trunk_type = graph.node_types[node_idx]
        expected_in_degree = graph.trunk_node_in_degrees[trunk_type]
        assert len(graph.node_inputs_indices[node_idx]) == expected_in_degree
        assert out_degree[node_idx] >= 1

    for node_idx in range(output_start, graph.num_nodes):
        assert len(graph.node_inputs_indices[node_idx]) == 1
        assert out_degree[node_idx] == 0
        assert node_idx in graph.leaf_node_indices


def _raw_parent_signature(raw: _EdgeSplitGraph) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(parents) for parents in raw.parents)


def _assert_raw_parent_child_multisets(raw: _EdgeSplitGraph) -> None:
    from_children: list[Counter[int]] = [Counter() for _ in raw.parents]
    for parent, children in enumerate(raw.children):
        for child in children:
            from_children[child][parent] += 1

    for node, parents in enumerate(raw.parents):
        assert Counter(parents) == from_children[node]


def _assert_raw_acyclic(raw: _EdgeSplitGraph) -> None:
    in_degrees = [len(parents) for parents in raw.parents]
    ready = deque(node for node, in_degree in enumerate(in_degrees) if in_degree == 0)
    visited = 0

    while ready:
        node = ready.popleft()
        visited += 1
        for child in raw.children[node]:
            in_degrees[child] -= 1
            if in_degrees[child] == 0:
                ready.append(child)

    assert visited == len(raw.parents)


def _build_unmixed_interleaved_graph(**params) -> _EdgeSplitGraph:
    return _build_interleaved_edge_split_graph(
        **params,
        mcmc_passes_per_split_round=0,
    )


def test_exact_counts_and_degrees() -> None:
    parameter_sets = [
        dict(
            num_root_nodes=1,
            num_trunk_nodes=0,
            num_output_nodes=1,
            trunk_node_in_degrees=[2, 2],
            num_trunk_node_types=2,
        ),
        dict(
            num_root_nodes=4,
            num_trunk_nodes=40,
            num_output_nodes=3,
            trunk_node_in_degrees=[1, 3],
            num_trunk_node_types=2,
        ),
        dict(
            num_root_nodes=8,
            num_trunk_nodes=128,
            num_output_nodes=4,
            trunk_node_in_degrees=[2, 2],
            num_trunk_node_types=2,
        ),
    ]
    for params in parameter_sets:
        for seed in range(20):
            torch.manual_seed(seed)
            graph = make_random_graph_description(**params)
            _assert_edge_split_invariants(graph)


def test_invariants_survive_interleaved_mcmc_mixing() -> None:
    parameter_sets = [
        dict(
            num_root_nodes=2,
            num_trunk_nodes=8,
            num_output_nodes=2,
            trunk_node_in_degrees=[1, 2],
            num_trunk_node_types=2,
        ),
        dict(
            num_root_nodes=4,
            num_trunk_nodes=30,
            num_output_nodes=5,
            trunk_node_in_degrees=[2, 3],
            num_trunk_node_types=2,
        ),
        dict(
            num_root_nodes=6,
            num_trunk_nodes=50,
            num_output_nodes=4,
            trunk_node_in_degrees=[1, 2, 3],
            num_trunk_node_types=3,
        ),
    ]
    for params in parameter_sets:
        for seed in range(10):
            torch.manual_seed(seed)
            graph = make_random_graph_description(
                **params,
                mcmc_passes_per_split_round=2,
            )
            _assert_edge_split_invariants(graph)


def test_mcmc_rewire_preserves_raw_in_degrees_roles_and_acyclicity() -> None:
    params = dict(
        num_root_nodes=4,
        num_trunk_nodes=40,
        num_output_nodes=3,
        trunk_node_in_degrees=[1, 3],
        num_trunk_node_types=2,
    )
    rng = random.Random(0)
    raw = _build_unmixed_interleaved_graph(**params, rng=rng)
    in_degrees = [len(parents) for parents in raw.parents]
    roles = list(raw.roles)

    _mcmc_rewire(raw, num_steps=1_000, rng=rng)

    assert [len(parents) for parents in raw.parents] == in_degrees
    assert raw.roles == roles
    for node, role in enumerate(raw.roles):
        if role is _EdgeSplitRole.LEAF:
            assert raw.children[node] == []
    _assert_raw_parent_child_multisets(raw)
    _assert_raw_acyclic(raw)


class _FirstChoiceRng:
    def randrange(self, stop: int) -> int:
        assert stop > 0
        return 0

    def choice(self, seq: list[int]) -> int:
        assert seq
        return seq[0]

    def shuffle(self, seq: list[int]) -> None:
        return


class _ReverseShuffleRng(_FirstChoiceRng):
    def shuffle(self, seq: list[int]) -> None:
        seq.reverse()


def test_interleaved_builder_round_robins_frozen_edges() -> None:
    raw = _build_unmixed_interleaved_graph(
        num_root_nodes=1,
        num_trunk_nodes=4,
        num_output_nodes=3,
        num_trunk_node_types=1,
        trunk_node_in_degrees=[1],
        rng=_FirstChoiceRng(),  # type: ignore[arg-type]
    )

    # The first round sees the three initial root->output edges and splits each
    # once; the fourth split starts a refreshed round at the first updated edge.
    assert [raw.parents[leaf][0] for leaf in (1, 2, 3)] == [4, 5, 6]
    assert raw.parents[4] == [7]
    assert raw.parents[5] == [0]
    assert raw.parents[6] == [0]
    assert raw.parents[7] == [0]
    _assert_raw_parent_child_multisets(raw)
    _assert_raw_acyclic(raw)


def test_interleaved_builder_shuffles_each_split_round() -> None:
    raw = _build_unmixed_interleaved_graph(
        num_root_nodes=1,
        num_trunk_nodes=3,
        num_output_nodes=3,
        num_trunk_node_types=1,
        trunk_node_in_degrees=[1],
        rng=_ReverseShuffleRng(),  # type: ignore[arg-type]
    )

    assert [raw.parents[leaf][0] for leaf in (1, 2, 3)] == [6, 5, 4]
    _assert_raw_parent_child_multisets(raw)
    _assert_raw_acyclic(raw)


def test_duplicate_parents_are_accepted() -> None:
    torch.manual_seed(0)
    graph = make_random_graph_description(
        num_root_nodes=1,
        num_trunk_nodes=1,
        num_output_nodes=1,
        trunk_node_in_degrees=[2],
        num_trunk_node_types=1,
    )
    assert graph.node_inputs_indices[1] == [0, 0]
    _assert_edge_split_invariants(graph)


class _DuplicateParentProposalRng:
    def __init__(self) -> None:
        self.randrange_calls = 0

    def randrange(self, stop: int) -> int:
        self.randrange_calls += 1
        if self.randrange_calls == 1:
            assert stop == 2
            return 0
        assert stop == 3
        return 1

    def shuffle(self, seq: list[int]) -> None:
        assert seq == [0, 1, 2, 3, 4]
        seq[:] = [2, 0, 1, 3, 4]


def test_mcmc_rewire_allows_duplicate_parent_proposals() -> None:
    raw = _EdgeSplitGraph(
        parents=[[], [], [0, 1], [2], [0]],
        children=[[2, 4], [2], [3], [], []],
        roles=[
            _EdgeSplitRole.ROOT,
            _EdgeSplitRole.ROOT,
            _EdgeSplitRole.TRUNK,
            _EdgeSplitRole.LEAF,
            _EdgeSplitRole.LEAF,
        ],
        trunk_types={2: 0},
    )

    _mcmc_rewire(raw, num_steps=1, rng=_DuplicateParentProposalRng())  # type: ignore[arg-type]

    assert raw.parents[2] == [1, 1]
    _assert_raw_parent_child_multisets(raw)
    _assert_raw_acyclic(raw)


class _McmcRoundRobinRng:
    def __init__(self) -> None:
        self.new_parents = iter([3, 2, 1, 3])
        self.shuffle_calls = 0

    def randrange(self, stop: int) -> int:
        if stop == 1:
            return 0
        assert stop == 4
        return next(self.new_parents)

    def shuffle(self, seq: list[int]) -> None:
        self.shuffle_calls += 1
        if self.shuffle_calls == 1:
            seq.reverse()
        else:
            seq.sort()


def test_mcmc_rewire_round_robins_target_nodes() -> None:
    raw = _EdgeSplitGraph(
        parents=[[], [], [], [], [0], [0], [0], [0]],
        children=[[4, 5, 6, 7], [], [], [], [], [], [], []],
        roles=[
            _EdgeSplitRole.ROOT,
            _EdgeSplitRole.ROOT,
            _EdgeSplitRole.ROOT,
            _EdgeSplitRole.ROOT,
            _EdgeSplitRole.LEAF,
            _EdgeSplitRole.LEAF,
            _EdgeSplitRole.LEAF,
            _EdgeSplitRole.LEAF,
        ],
        trunk_types={},
    )
    rng = _McmcRoundRobinRng()

    _mcmc_rewire(raw, num_steps=9, rng=rng)  # type: ignore[arg-type]

    assert [raw.parents[leaf][0] for leaf in (4, 5, 6, 7)] == [0, 1, 2, 3]
    assert rng.shuffle_calls == 2
    _assert_raw_parent_child_multisets(raw)
    _assert_raw_acyclic(raw)


class _ActiveNodeMcmcRng:
    def randrange(self, stop: int) -> int:
        if stop == 1:
            return 0
        assert stop == 3
        return 1

    def shuffle(self, seq: list[int]) -> None:
        assert seq == [0, 1, 2, 3]
        seq[:] = [3, 0, 1, 2]


def test_mcmc_rewire_excludes_inactive_future_trunks() -> None:
    raw = _EdgeSplitGraph(
        parents=[[], [], [0], [0], []],
        children=[[2, 3], [], [], [], []],
        roles=[
            _EdgeSplitRole.ROOT,
            _EdgeSplitRole.ROOT,
            _EdgeSplitRole.LEAF,
            _EdgeSplitRole.TRUNK,
            _EdgeSplitRole.TRUNK,
        ],
        trunk_types={3: 0},
    )

    _mcmc_rewire(
        raw,
        num_steps=1,
        rng=_ActiveNodeMcmcRng(),  # type: ignore[arg-type]
        active_nodes=[0, 1, 2, 3],
    )

    assert raw.parents[3] == [1]
    assert raw.children[4] == []
    _assert_raw_parent_child_multisets(raw)
    _assert_raw_acyclic(raw)


class _ShuffleRecordingRng(_FirstChoiceRng):
    def __init__(self) -> None:
        self.node_order_shuffle_lengths: list[int] = []

    def shuffle(self, seq) -> None:
        if seq and isinstance(seq[0], int):
            self.node_order_shuffle_lengths.append(len(seq))


def test_interleaved_builder_runs_configured_final_full_size_mcmc_passes() -> None:
    rng = _ShuffleRecordingRng()

    _build_interleaved_edge_split_graph(
        num_root_nodes=1,
        num_trunk_nodes=2,
        num_output_nodes=1,
        num_trunk_node_types=1,
        trunk_node_in_degrees=[1],
        mcmc_passes_per_split_round=2,
        rng=rng,  # type: ignore[arg-type]
    )

    assert rng.node_order_shuffle_lengths == [3, 3, 4, 4]


def test_torch_seed_determinism() -> None:
    params = dict(
        num_root_nodes=4,
        num_trunk_nodes=40,
        num_output_nodes=3,
        trunk_node_in_degrees=[1, 3],
        num_trunk_node_types=2,
    )
    torch.manual_seed(123)
    first = make_random_graph_description(**params)
    torch.manual_seed(123)
    second = make_random_graph_description(**params)
    torch.manual_seed(124)
    different = make_random_graph_description(**params)

    assert first.node_inputs_indices == second.node_inputs_indices
    assert first.node_types == second.node_types
    assert (first.node_inputs_indices, first.node_types) != (
        different.node_inputs_indices,
        different.node_types,
    )


def test_mcmc_torch_seed_determinism() -> None:
    params = dict(
        num_root_nodes=4,
        num_trunk_nodes=40,
        num_output_nodes=3,
        trunk_node_in_degrees=[1, 3],
        num_trunk_node_types=2,
        mcmc_passes_per_split_round=3,
    )
    torch.manual_seed(123)
    first = make_random_graph_description(**params)
    torch.manual_seed(123)
    second = make_random_graph_description(**params)
    torch.manual_seed(124)
    different = make_random_graph_description(**params)

    assert first.node_inputs_indices == second.node_inputs_indices
    assert first.node_types == second.node_types
    assert (first.node_inputs_indices, first.node_types) != (
        different.node_inputs_indices,
        different.node_types,
    )


def test_mcmc_chain_moves_from_edge_split_seed() -> None:
    params = dict(
        num_root_nodes=4,
        num_trunk_nodes=20,
        num_output_nodes=4,
        trunk_node_in_degrees=[2],
        num_trunk_node_types=1,
    )
    moved = False
    for seed in range(20):
        rng = random.Random(seed)
        raw = _build_unmixed_interleaved_graph(**params, rng=rng)
        before = _raw_parent_signature(raw)
        _mcmc_rewire(raw, num_steps=2_000, rng=rng)
        if _raw_parent_signature(raw) != before:
            moved = True
            break

    assert moved


def test_native_flat_arrays_round_trip_when_available() -> None:
    if _native is None:
        return

    params = dict(
        num_root_nodes=4,
        num_trunk_nodes=40,
        num_output_nodes=3,
        trunk_node_in_degrees=[1, 3],
        num_trunk_node_types=2,
    )
    parent_offsets, parent_indices, node_types = _native.generate_random_graph_arrays(
        **params,
        seed=123,
        mcmc_passes_per_split_round=2,
    )
    graph = _description_from_flat_parent_arrays(
        **params,
        parent_offsets=parent_offsets,
        parent_indices=parent_indices,
        node_types=node_types,
    )

    _assert_edge_split_invariants(graph)


def main() -> None:
    test_exact_counts_and_degrees()
    test_invariants_survive_interleaved_mcmc_mixing()
    test_mcmc_rewire_preserves_raw_in_degrees_roles_and_acyclicity()
    test_interleaved_builder_round_robins_frozen_edges()
    test_interleaved_builder_shuffles_each_split_round()
    test_duplicate_parents_are_accepted()
    test_mcmc_rewire_allows_duplicate_parent_proposals()
    test_mcmc_rewire_round_robins_target_nodes()
    test_mcmc_rewire_excludes_inactive_future_trunks()
    test_interleaved_builder_runs_configured_final_full_size_mcmc_passes()
    test_torch_seed_determinism()
    test_mcmc_torch_seed_determinism()
    test_mcmc_chain_moves_from_edge_split_seed()
    test_native_flat_arrays_round_trip_when_available()
    print("ALL EDGE-SPLIT GENERATION CHECKS PASSED")


if __name__ == "__main__":
    main()
