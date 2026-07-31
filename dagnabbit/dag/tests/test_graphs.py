"""Tests for graph sampling and the batch representation.

The compiled generator in :mod:`dagnabbit.dag.generate` reimplements the rank
sweep that :mod:`dagnabbit.dag.graphs` also computes in Python, so the risk is
silent divergence rather than a crash. Four things are checked:

1. **Exactness of the ranks**, against the Python implementation on the same
   structure. Depth is what the eval stratifies by, so a subtle disagreement
   would corrupt the one metric that decides whether the simulator simulates.
2. **Equivalence of the two construction paths**, array and ragged-list.
3. **The invariants downstream code assumes.** Chiefly: every parent points
   strictly earlier and never at an output, so a sequential sweep in node order
   is a valid evaluation; and no producer is dead, which the coverage pass
   exists to guarantee.
4. **The sampled distribution**, statistically -- including that a gate really
   can draw the same parent twice, since that is the only way an inverter
   exists.
"""

import numpy as np
import pytest
import torch

from dagnabbit.dag.generate import generate_arrays
from dagnabbit.dag.graphs import (
    Geometry,
    SamplingConfig,
    from_lists,
    ranks_from_lists,
    sample,
)

# All satisfying the two generation preconditions: roots >= max in-degree, and
# roots <= trunk * (min in-degree - 1) + outputs.
GEOMETRIES = [
    pytest.param(Geometry(1, 4, 2, 1, (1,)), id="in-degree-1"),
    pytest.param(Geometry(4, 8, 2, 1, (4,)), id="in-degree-equals-roots"),
    pytest.param(Geometry(4, 6, 2, 3, (2, 3, 4)), id="mixed-in-degrees-ragged"),
    pytest.param(Geometry(16, 128, 8, 1, (2,)), id="training-geometry"),
    pytest.param(Geometry(3, 1, 3, 1, (2,)), id="single-trunk-node"),
    pytest.param(Geometry(2, 2, 4, 2, (1, 2)), id="mixed-with-in-degree-1"),
]
TRAINING_GEOMETRY = Geometry(16, 128, 8, 1, (2,))
SMALL_GEOMETRY = Geometry(8, 24, 4, 1, (2,))

# For the two prior knobs. The trunk floor satisfies coverage at its *minimum*
# (8 roots <= 8 trunk * 1 + 4 outputs), which is the binding case.
MIXED_GEOMETRY = Geometry(8, 32, 4, 3, (2, 2, 2))
MASKED = SamplingConfig(minimum_trunk_nodes=8)
DIRICHLET = SamplingConfig(trunk_type_concentration=1.0)


def raw_arrays(
    count: int,
    geometry: Geometry,
    seed: int,
    sampling: SamplingConfig = SamplingConfig(),
):
    return generate_arrays(
        count,
        geometry.num_root_nodes,
        sampling.resolve_minimum_trunk_nodes(geometry),
        geometry.num_trunk_nodes,
        geometry.num_output_nodes,
        geometry.num_trunk_node_types,
        sampling.concentration,
        geometry.mask_type,
        np.asarray(geometry.trunk_node_in_degrees, dtype=np.int64),
        geometry.maximum_indegree,
        seed,
    )


def ragged_parents(parents, in_degrees, index: int, num_nodes: int) -> list[list[int]]:
    return [
        parents[index, node, : in_degrees[index, node]].tolist()
        for node in range(num_nodes)
    ]


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_compiled_ranks_match_the_python_implementation(geometry) -> None:
    """Ranks must agree exactly, not approximately.

    The compiled *structure* is handed to the Python implementation, which
    derives the ranks itself, so a disagreement here is between the two
    algorithms rather than between two samplers.
    """
    count = 40
    node_types, in_degrees, parents, ranks = raw_arrays(count, geometry, 17)

    for index in range(count):
        inputs = ragged_parents(parents, in_degrees, index, geometry.num_nodes)
        assert ranks_from_lists(inputs, geometry) == ranks[index].tolist(), (
            f"ranks, graph {index}"
        )


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_the_two_construction_paths_agree(geometry) -> None:
    """``sample`` and ``from_lists`` must produce identical tensors."""
    count = 6
    node_types, in_degrees, parents, _ = raw_arrays(count, geometry, 4)
    batched = sample(count, geometry, seed=4)

    for index in range(count):
        one = from_lists(
            ragged_parents(parents, in_degrees, index, geometry.num_nodes),
            node_types[index].tolist(),
            geometry,
        )
        for name in ("node_types", "parent_indices", "parent_slot_mask", "ranks"):
            assert torch.equal(getattr(one, name)[0], getattr(batched, name)[index]), (
                f"{name}, graph {index}"
            )


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_batch_invariants(geometry) -> None:
    """Everything downstream code assumes about a sampled batch."""
    torch.manual_seed(0)
    graphs = sample(24, geometry)
    batch = len(graphs)
    num_nodes = geometry.num_nodes
    roots = geometry.num_root_nodes
    output_start = geometry.output_start
    indices = torch.arange(num_nodes)
    mask = graphs.parent_slot_mask

    assert graphs.node_types.shape == (batch, num_nodes)
    assert graphs.parent_indices.shape == (batch, num_nodes, geometry.maximum_indegree)

    # Roots occupy the first indices in slot order and have no parents; each
    # gets its own type. Outputs occupy the last, share one type, in-degree 1.
    expected_root_types = torch.arange(geometry.root_type_start, geometry.output_type)
    assert torch.equal(
        graphs.node_types[:, :roots], expected_root_types.expand(batch, roots)
    )
    assert not mask[:, :roots].any()
    assert (graphs.node_types[:, output_start:] == geometry.output_type).all()
    assert mask[:, output_start:, 0].all()
    assert not mask[:, output_start:, 1:].any()

    # Trunk nodes hold trunk types, and each node's filled slot count is exactly
    # its type's in-degree, with no gaps.
    trunk_types = graphs.node_types[:, roots:output_start]
    assert (trunk_types < geometry.num_trunk_node_types).all()
    in_degrees = torch.tensor(geometry.trunk_node_in_degrees)[trunk_types]
    slots = torch.arange(geometry.maximum_indegree)
    assert torch.equal(mask[:, roots:output_start], slots < in_degrees.unsqueeze(-1))

    # Node index order is a valid topological order: every parent is strictly
    # earlier, and never an output, since outputs are leaves. This is what makes
    # a single sweep in node order a correct evaluation.
    assert ((graphs.parent_indices < indices[None, :, None]) | ~mask).all()
    assert ((graphs.parent_indices < output_start) | ~mask).all()
    # Padded slots hold 0, not a stale index.
    assert (graphs.parent_indices[~mask] == 0).all()

    # Rank is 1 + max over parent ranks, with roots at 0.
    assert (graphs.ranks[:, :roots] == 0).all()
    parent_ranks = graphs.ranks.gather(
        1, graphs.parent_indices.reshape(batch, -1)
    ).reshape(graphs.parent_indices.shape)
    parent_ranks = torch.where(mask, parent_ranks, torch.full_like(parent_ranks, -1))
    assert torch.equal(
        parent_ranks.max(dim=2).values[:, roots:] + 1, graphs.ranks[:, roots:]
    )


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_no_producer_is_dead(geometry) -> None:
    """What the sampler's coverage pass exists to guarantee.

    Coverage runs to completion before any slot is filled at random, so it holds
    regardless of duplicate parents -- which is the whole reason duplicates could
    be allowed without giving anything up.
    """
    torch.manual_seed(0)
    graphs = sample(40, geometry)
    mask = graphs.parent_slot_mask
    for index in range(len(graphs)):
        referenced = set(graphs.parent_indices[index][mask[index]].tolist())
        assert referenced == set(range(geometry.output_start)), (
            f"unused producer in graph {index} at {geometry}"
        )


def test_gates_may_repeat_a_parent() -> None:
    """``NAND(x, x) = NOT x``, so this is the only way an inverter exists.

    Rare but real: roughly 1 in 80 gates, since a gate at index i draws from i
    producers. If this ever reads zero, the rejection loop is back and the
    sampler can no longer express the reference adder's inverters.
    """
    geometry = TRAINING_GEOMETRY
    torch.manual_seed(0)
    graphs = sample(256, geometry)
    slots = graphs.parent_indices[:, geometry.num_root_nodes : geometry.output_start]
    duplicate = slots[..., 0] == slots[..., 1]
    rate = float(duplicate.double().mean())
    assert 0.002 < rate < 0.05, f"duplicate-parent rate {rate:.4f} is implausible"


def test_generation_is_reproducible_and_uncorrelated() -> None:
    """numba keeps its own RNG state, so seeding is easy to get wrong.

    Identical batches step to step would silently destroy the premise of the
    whole setup -- that no example is ever seen twice -- while throughput looked
    fine.
    """
    geometry = SMALL_GEOMETRY

    torch.manual_seed(1234)
    first = sample(16, geometry)
    torch.manual_seed(1234)
    replay = sample(16, geometry)
    assert torch.equal(first.node_types, replay.node_types)
    assert torch.equal(first.parent_indices, replay.parent_indices)

    # Consecutive batches under one seed must differ from each other and from
    # the first: the generator is reseeded per call from torch.
    torch.manual_seed(99)
    signatures = set()
    for _ in range(4):
        batch = sample(8, geometry)
        for index in range(8):
            signatures.add(
                (
                    tuple(batch.node_types[index].tolist()),
                    tuple(map(tuple, batch.parent_indices[index].tolist())),
                )
            )
    assert len(signatures) == 32, (
        f"only {len(signatures)} distinct graphs out of 32; the compiled "
        "generator is probably being reseeded identically each call"
    )


# --------------------------------------------------------------------------
# The sampling prior: a variable gate count, and a per-graph gate mixture
# --------------------------------------------------------------------------


def test_masked_positions_are_an_inert_trailing_block() -> None:
    """What every downstream consumer assumes about ``<MASK>``.

    The whole point of masking is that it changes the *graph* without changing
    the tensor: the sequence length is fixed, and a masked position must be
    unreachable rather than merely unused. If anything could reference one, the
    evaluator would read a value that was never computed.
    """
    geometry = MIXED_GEOMETRY
    torch.manual_seed(0)
    graphs = sample(32, geometry, sampling=MASKED)
    roots = geometry.num_root_nodes
    output_start = geometry.output_start
    mask = graphs.parent_slot_mask

    masked = graphs.trunk_is_masked
    live_counts = graphs.num_live_trunk_nodes
    # Masked positions are a suffix of the trunk block, so the live count and
    # the layout are the same fact stated twice.
    trunk_offsets = torch.arange(geometry.num_trunk_nodes)
    assert torch.equal(masked, trunk_offsets[None, :] >= live_counts[:, None])

    # In-degree 0, no parent slots, rank 0 -- they sit on no path at all.
    assert not mask[:, roots:output_start][masked].any()
    assert (graphs.ranks[:, roots:output_start][masked] == 0).all()

    for index in range(len(graphs)):
        limit = roots + int(live_counts[index])
        referenced = set(graphs.parent_indices[index][mask[index]].tolist())
        # Coverage still holds, but over the live producers only: every root and
        # every live gate is read, and no masked position ever is.
        assert referenced == set(range(limit)), f"graph {index}"


def test_live_trunk_count_spans_the_configured_range() -> None:
    """``num_trunk_nodes`` is a maximum and the floor is drawn uniformly."""
    geometry = MIXED_GEOMETRY
    torch.manual_seed(1)
    counts = sample(400, geometry, sampling=MASKED).num_live_trunk_nodes

    minimum = MASKED.resolve_minimum_trunk_nodes(geometry)
    assert int(counts.min()) == minimum
    assert int(counts.max()) == geometry.num_trunk_nodes
    midpoint = (minimum + geometry.num_trunk_nodes) / 2
    assert abs(float(counts.double().mean()) - midpoint) < 1.5

    # And the default really is the old fixed-size behaviour.
    fixed = sample(8, geometry).num_live_trunk_nodes
    assert (fixed == geometry.num_trunk_nodes).all()


def test_gate_mixture_varies_per_graph_but_not_in_aggregate() -> None:
    """A Dirichlet mixture must move the *spread*, not the marginal.

    Under the old uniform draw a graph's share of any one gate type is a
    binomial proportion: at 32 gates over 3 types its standard deviation is
    about 0.083, so every graph looks like every other. Dirichlet(1) makes the
    share Beta(1, 2), whose standard deviation is 0.236 -- that gap is the
    entire point, since the hand-built adders are single-type circuits.
    """
    geometry = MIXED_GEOMETRY
    num_types = geometry.num_trunk_node_types

    def type_shares(sampling: SamplingConfig) -> torch.Tensor:
        torch.manual_seed(3)
        graphs = sample(500, geometry, sampling=sampling)
        counts = torch.stack(
            [(graphs.trunk_types == kind).sum(dim=1) for kind in range(num_types)],
            dim=1,
        )
        return counts.double() / counts.sum(dim=1, keepdim=True)

    uniform = type_shares(SamplingConfig())
    dirichlet = type_shares(DIRICHLET)

    assert float(uniform[:, 0].std()) < 0.12
    assert float(dirichlet[:, 0].std()) > 0.17

    # The batch-wide marginal is unchanged, which is what makes this a swap of
    # priors rather than a reweighting of the gate set.
    assert abs(float(dirichlet.mean(dim=0)[0]) - 1 / num_types) < 0.03
    # Lopsided graphs must actually occur, not just wider error bars.
    assert float(dirichlet.max(dim=1).values.max()) > 0.9


def test_masking_and_mixture_survive_the_list_path() -> None:
    """``from_lists`` accepts what ``sample`` emits, masks included.

    This is also the only check that :func:`validate_lists` understands
    ``<MASK>``: a hand-built circuit is how a masked graph would ever be
    written down by hand.
    """
    geometry = MIXED_GEOMETRY
    sampling = SamplingConfig(minimum_trunk_nodes=8, trunk_type_concentration=1.0)
    count = 8
    node_types, in_degrees, parents, _ = raw_arrays(count, geometry, 11, sampling)
    batched = sample(count, geometry, seed=11, sampling=sampling)

    seen_masks = 0
    for index in range(count):
        inputs = ragged_parents(parents, in_degrees, index, geometry.num_nodes)
        seen_masks += node_types[index].tolist().count(geometry.mask_type)
        # The compiled rank sweep must special-case <MASK> the same way.
        assert ranks_from_lists(inputs, geometry) == batched.ranks[index].tolist()

        one = from_lists(inputs, node_types[index].tolist(), geometry)
        for name in ("node_types", "parent_indices", "parent_slot_mask", "ranks"):
            assert torch.equal(getattr(one, name)[0], getattr(batched, name)[index]), (
                f"{name}, graph {index}"
            )
    assert seen_masks > 0, "no graph was actually masked; the test proved nothing"


def test_validate_lists_rejects_broken_mask_layouts() -> None:
    """Masks are a trailing block that nothing reads; both halves are checked."""
    from dagnabbit.dag.graphs import validate_lists

    geometry = Geometry(2, 3, 1, 1, (2,))
    mask = geometry.mask_type
    output = geometry.output_type
    roots = [geometry.root_type_start + i for i in range(geometry.num_root_nodes)]
    live = [[], [], [0, 1], [0, 1], [2, 3], [4]]
    types = [*roots, 0, 0, 0, output]

    validate_lists(live, types, geometry)  # the unmasked baseline

    # A gate after the mask block starts.
    with pytest.raises(ValueError, match="trailing block"):
        validate_lists(
            [[], [], [0, 1], [], [0, 1], [4]],
            [*roots, 0, mask, 0, output],
            geometry,
        )
    # A live node reading a masked position.
    with pytest.raises(ValueError, match="<MASK>"):
        validate_lists(
            [[], [], [0, 1], [], [], [3]],
            [*roots, 0, mask, mask, output],
            geometry,
        )
    # A masked position with parents.
    with pytest.raises(ValueError, match="but has parents"):
        validate_lists(live, [*roots, 0, 0, mask, output], geometry)


def test_sampled_distribution_is_unbiased() -> None:
    """Cheap statistics over the sampler's prior, at the training geometry.

    ``ascending`` is the one that catches a broken shuffle: both wiring passes
    append, so without the Fisher-Yates pass slot 0 would always hold the
    coverage parent and the share would sit far from a half.
    """
    geometry = Geometry(16, 128, 8, 2, (2, 2))
    torch.manual_seed(5)
    graphs = sample(600, geometry)
    mask = graphs.parent_slot_mask

    out_degrees = torch.bincount(
        graphs.parent_indices[mask], minlength=geometry.output_start
    )
    # Every slot is filled exactly once, so the mean out-degree is fixed by the
    # counts: (trunk * 2 + outputs) slots over (roots + trunk) producers.
    expected_mean = (geometry.num_trunk_nodes * 2 + geometry.num_output_nodes) / (
        geometry.output_start
    )
    assert abs(float(out_degrees.double().mean()) / len(graphs) - expected_mean) < 1e-9
    assert (out_degrees > 0).all(), "some producer was never used in any graph"

    types = torch.bincount(graphs.trunk_types.reshape(-1), minlength=2)
    type_share = types.double() / types.sum()
    assert abs(float(type_share[0]) - 0.5) < 0.02, f"trunk type draw biased: {types}"

    pairs = graphs.parent_indices[:, geometry.num_root_nodes : geometry.output_start]
    distinct = pairs[..., 0] != pairs[..., 1]
    share = float((pairs[..., 0] < pairs[..., 1])[distinct].double().mean())
    assert abs(share - 0.5) < 0.02, f"slot order looks biased: {share:.4f} ascending"
