"""The candidate population: legality under compaction, straight-through
exactness against NodeTokens, gradient flow, and seeding.

The exactness test is the one that matters most: the whole Phase 1 premise is
that the soft token path and the hard graph are *the same circuit*, so any
drift between ``CandidatePopulation._tokens`` and ``NodeTokens.forward`` makes
the surrogate gradient point at a graph that is not the one being evaluated.
"""

import pytest
import torch

from dagnabbit.dag.candidates import CandidatePopulation
from dagnabbit.dag.graphs import (
    Geometry,
    GraphBatch,
    SamplingConfig,
    ranks_from_lists,
    sample,
    validate_lists,
)
from dagnabbit.dag.model import NodeTokens, SimulatorConfig

GEOMETRY = Geometry(
    num_root_nodes=4,
    num_trunk_nodes=12,
    num_output_nodes=2,
    num_trunk_node_types=1,
    trunk_node_in_degrees=(2,),
)
CONFIG = SimulatorConfig(embedding_dim=32, attention_head_dim=16)
SAMPLING = SamplingConfig(minimum_trunk_nodes=5)


def make_population(count: int = 8, seed: int = 0) -> CandidatePopulation:
    torch.manual_seed(seed)
    population = CandidatePopulation(count, GEOMETRY)
    # Random logits rather than zeros, so compaction and legality are exercised
    # by a population that actually disagrees with itself.
    with torch.no_grad():
        for parameter in population.parameters():
            parameter.normal_(std=1.5)
    return population


def graphs_to_lists(graphs: GraphBatch, index: int) -> tuple[list[list[int]], list[int]]:
    node_inputs = []
    for node in range(graphs.geometry.num_nodes):
        mask = graphs.parent_slot_mask[index, node]
        node_inputs.append(graphs.parent_indices[index, node][mask].tolist())
    return node_inputs, graphs.node_types[index].tolist()


def test_sampled_graphs_satisfy_every_graph_invariant():
    """validate_lists is the full contract: topology, trailing masks, types."""
    population = make_population()
    node_tokens = NodeTokens(GEOMETRY, CONFIG)
    result = population.sample(node_tokens)
    for index in range(len(result.graphs)):
        node_inputs, node_types = graphs_to_lists(result.graphs, index)
        validate_lists(node_inputs, node_types, GEOMETRY)


def test_ranks_match_the_reference_computation():
    population = make_population(seed=3)
    node_tokens = NodeTokens(GEOMETRY, CONFIG)
    result = population.sample(node_tokens)
    for index in range(len(result.graphs)):
        node_inputs, _ = graphs_to_lists(result.graphs, index)
        expected = ranks_from_lists(node_inputs, GEOMETRY)
        assert result.graphs.ranks[index].tolist() == expected


def test_straight_through_tokens_match_node_tokens_exactly():
    """The soft path's forward value must be the hard graph's encoding."""
    population = make_population(seed=1)
    node_tokens = NodeTokens(GEOMETRY, CONFIG)
    result = population.sample(node_tokens)
    expected = node_tokens(
        result.graphs.node_types,
        result.graphs.parent_indices,
        result.graphs.parent_slot_mask,
    )
    assert torch.allclose(result.tokens, expected, atol=1e-5)


def test_gradients_reach_every_logit_family():
    population = make_population(seed=2)
    node_tokens = NodeTokens(GEOMETRY, CONFIG)
    result = population.sample(node_tokens)
    result.tokens.square().sum().backward()
    for name, parameter in population.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.abs().sum()) > 0, name


def test_initialize_from_reproduces_the_seeds():
    """At high concentration, sampling gives back the seed graphs themselves."""
    torch.manual_seed(4)
    count = 8
    seeds = sample(count, GEOMETRY, seed=11, sampling=SAMPLING)
    population = CandidatePopulation(count, GEOMETRY)
    population.initialize_from(seeds, concentration=50.0)
    node_tokens = NodeTokens(GEOMETRY, CONFIG)
    result = population.sample(node_tokens)

    assert torch.equal(result.graphs.node_types, seeds.node_types)
    assert torch.equal(result.graphs.parent_slot_mask, seeds.parent_slot_mask)
    filled = seeds.parent_slot_mask
    assert torch.equal(
        result.graphs.parent_indices[filled], seeds.parent_indices[filled]
    )


def test_masked_positions_always_trail_and_are_never_referenced():
    """The invariant the stable sort exists for, checked directly."""
    population = make_population(seed=5, count=16)
    node_tokens = NodeTokens(GEOMETRY, CONFIG)
    graphs = population.sample(node_tokens).graphs

    masked = graphs.trunk_is_masked
    live = ~masked
    # Trailing block: once masked, masked forever.
    assert bool((masked[:, :-1] <= masked[:, 1:]).all())

    # No filled slot anywhere points at a masked position.
    roots = GEOMETRY.num_root_nodes
    filled = graphs.parent_slot_mask
    parents = graphs.parent_indices
    is_trunk_parent = parents >= roots
    trunk_parent_live = live.gather(
        1, (parents - roots).clamp(min=0).flatten(1)
    ).reshape_as(parents)
    ok = ~filled | ~is_trunk_parent | trunk_parent_live
    assert bool(ok.all())


def test_mixed_in_degrees_are_rejected():
    lopsided = Geometry(
        num_root_nodes=4,
        num_trunk_nodes=8,
        num_output_nodes=2,
        num_trunk_node_types=2,
        trunk_node_in_degrees=(2, 3),
    )
    with pytest.raises(ValueError, match="uniform in-degree"):
        CandidatePopulation(4, lopsided)
