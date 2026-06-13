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
    _build_edge_split_graph,
    _mcmc_rewire,
    _sample_valid_parent,
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


def test_invariants_survive_mcmc_mixing() -> None:
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
            graph = make_random_graph_description(**params, num_mixing_steps=200)
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
    raw = _build_edge_split_graph(**params, rng=rng)
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
            return 1
        assert stop == 4
        return 0

    def choice(self, seq: list[int]) -> int:
        assert seq == [0, 1, 2]
        return 1


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
        num_mixing_steps=500,
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
        raw = _build_edge_split_graph(**params, rng=rng)
        before = _raw_parent_signature(raw)
        _mcmc_rewire(raw, num_steps=2_000, rng=rng)
        if _raw_parent_signature(raw) != before:
            moved = True
            break

    assert moved


def test_incremental_tc_path_is_structurally_valid() -> None:
    params = dict(
        num_root_nodes=4,
        num_trunk_nodes=80,
        num_output_nodes=3,
        trunk_node_in_degrees=[1, 3],
        num_trunk_node_types=2,
    )
    for seed in range(20):
        rng = random.Random(seed)
        raw = _build_edge_split_graph(**params, rng=rng, use_incremental_tc=True)
        graph = _assemble_edge_split_description(graph=raw, rng=rng, **params)
        _assert_edge_split_invariants(graph)


def test_valid_parent_sampler_is_uniform_over_bruteforce_valid_set() -> None:
    eligible = [0, 1, 2, 3]
    forbidden = {2}
    valid = [node for node in eligible if node not in forbidden]
    assert valid == [0, 1, 3]

    rng = random.Random(0)
    counts = {node: 0 for node in eligible}
    draws = 12_000
    for _ in range(draws):
        counts[_sample_valid_parent(eligible, forbidden, rng)] += 1

    expected = draws / len(valid)
    assert counts[2] == 0
    for node in valid:
        assert abs(counts[node] - expected) < expected * 0.08, counts


def main() -> None:
    test_exact_counts_and_degrees()
    test_invariants_survive_mcmc_mixing()
    test_mcmc_rewire_preserves_raw_in_degrees_roles_and_acyclicity()
    test_duplicate_parents_are_accepted()
    test_mcmc_rewire_allows_duplicate_parent_proposals()
    test_torch_seed_determinism()
    test_mcmc_torch_seed_determinism()
    test_mcmc_chain_moves_from_edge_split_seed()
    test_incremental_tc_path_is_structurally_valid()
    test_valid_parent_sampler_is_uniform_over_bruteforce_valid_set()
    print("ALL EDGE-SPLIT GENERATION CHECKS PASSED")


if __name__ == "__main__":
    main()
