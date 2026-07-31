"""Tests for graph sampling and the canonical representation.

The compiled generator in :mod:`dagnabbit.dag.generate` reimplements the two
derived overlays that :mod:`dagnabbit.dag.canonical` also computes in Python, so
the risk here is silent divergence rather than a crash. Four things are checked:

1. **Exactness of the overlays.** Given a structure, the compiled ``ranks`` and
   ``canonical_order`` must be bit-identical to :func:`ranks_from_lists` and
   :func:`canonical_order_from_lists`. Not a nicety: canonical position is what
   the model's tokens and the evaluator's gathers both index by, and the two
   implementations order the Kahn frontier with independently written
   comparisons.
2. **Equivalence of the two construction paths**, array and ragged-list.
3. **The invariants downstream code assumes**, checked in canonical space.
4. **The sampled distribution**, statistically, plus per-graph constraints.
"""

import random

import numpy as np
import pytest
import torch

from dagnabbit.dag.canonical import (
    Geometry,
    canonical_order_from_lists,
    from_lists,
    ranks_from_lists,
    sample,
)
from dagnabbit.dag.generate import generate_arrays

# All satisfying the two generation preconditions: roots >= max in-degree, and
# roots <= trunk * (min in-degree - 1) + outputs.
GEOMETRIES = [
    pytest.param(Geometry(1, 4, 2, 1, (1,)), id="in-degree-1"),
    pytest.param(Geometry(4, 8, 2, 1, (4,)), id="in-degree-equals-roots"),
    pytest.param(Geometry(4, 6, 2, 3, (2, 3, 4)), id="mixed-in-degrees-ragged"),
    pytest.param(Geometry(16, 128, 8, 2, (2, 2)), id="training-geometry"),
    pytest.param(Geometry(3, 1, 3, 1, (2,)), id="single-trunk-node"),
    pytest.param(Geometry(2, 2, 4, 2, (1, 2)), id="mixed-with-in-degree-1"),
]
TRAINING_GEOMETRY = Geometry(16, 128, 8, 2, (2, 2))
SMALL_GEOMETRY = Geometry(8, 24, 4, 2, (2, 2))


def raw_arrays(count: int, geometry: Geometry, seed: int):
    return generate_arrays(
        count,
        geometry.num_root_nodes,
        geometry.num_trunk_nodes,
        geometry.num_output_nodes,
        geometry.num_trunk_node_types,
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
def test_compiled_overlays_match_the_python_implementation(geometry) -> None:
    """Ranks and canonical order must agree exactly, not approximately.

    The compiled *structure* is handed to the Python implementation, which
    derives the overlays itself, so a disagreement here is a disagreement
    between the two algorithms rather than between two samplers.
    """
    count = 40
    node_types, in_degrees, parents, ranks, order = raw_arrays(count, geometry, 17)

    for index in range(count):
        inputs = ragged_parents(parents, in_degrees, index, geometry.num_nodes)
        types = node_types[index].tolist()
        assert ranks_from_lists(inputs, geometry) == ranks[index].tolist(), (
            f"ranks, graph {index}"
        )
        assert (
            canonical_order_from_lists(inputs, types, geometry) == order[index].tolist()
        ), f"canonical_order, graph {index}"


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_the_two_construction_paths_agree(geometry) -> None:
    """``sample`` and ``from_lists`` must produce identical tensors."""
    count = 6
    node_types, in_degrees, parents, _, _ = raw_arrays(count, geometry, 4)
    torch.manual_seed(4)
    batched = sample(count, geometry, seed=4)

    for index in range(count):
        one = from_lists(
            ragged_parents(parents, in_degrees, index, geometry.num_nodes),
            node_types[index].tolist(),
            geometry,
        )
        for name in ("node_types", "parent_positions", "parent_slot_mask", "ranks"):
            assert torch.equal(getattr(one, name)[0], getattr(batched, name)[index]), (
                f"{name}, graph {index}"
            )


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_canonical_invariants(geometry) -> None:
    """Everything downstream code assumes, checked in canonical space."""
    torch.manual_seed(0)
    graphs = sample(24, geometry)
    batch = len(graphs)
    num_nodes = geometry.num_nodes
    roots = geometry.num_root_nodes
    output_start = geometry.output_start
    positions = torch.arange(num_nodes)
    mask = graphs.parent_slot_mask

    assert graphs.node_types.shape == (batch, num_nodes)
    assert graphs.parent_positions.shape == (
        batch,
        num_nodes,
        geometry.maximum_indegree,
    )

    # Roots occupy the first positions in slot order and have no parents; each
    # gets its own type. Outputs occupy the last positions, share one type, and
    # are in-degree-1.
    expected_root_types = torch.arange(geometry.root_type_start, geometry.output_type)
    assert torch.equal(
        graphs.node_types[:, :roots], expected_root_types.expand(batch, roots)
    )
    assert not mask[:, :roots].any()
    assert (graphs.node_types[:, output_start:] == geometry.output_type).all()
    assert mask[:, output_start:, 0].all()
    assert not mask[:, output_start:, 1:].any()

    # Trunk positions hold trunk types, and each node's filled slot count is
    # exactly its type's in-degree, with no gaps.
    trunk_types = graphs.node_types[:, roots:output_start]
    assert (trunk_types < geometry.num_trunk_node_types).all()
    in_degrees = torch.tensor(geometry.trunk_node_in_degrees)[trunk_types]
    slots = torch.arange(geometry.maximum_indegree)
    assert torch.equal(mask[:, roots:output_start], slots < in_degrees.unsqueeze(-1))

    # Every valid parent points strictly earlier, and never into the output
    # block: outputs are leaves.
    assert ((graphs.parent_positions < positions[None, :, None]) | ~mask).all()
    assert ((graphs.parent_positions < output_start) | ~mask).all()
    # Padded slots hold 0, not a stale position.
    assert (graphs.parent_positions[~mask] == 0).all()

    # Rank is 1 + max over parent ranks, with roots at 0.
    assert (graphs.ranks[:, :roots] == 0).all()
    parent_ranks = graphs.ranks.gather(
        1, graphs.parent_positions.reshape(batch, -1)
    ).reshape(graphs.parent_positions.shape)
    parent_ranks = torch.where(mask, parent_ranks, torch.full_like(parent_ranks, -1))
    assert torch.equal(
        parent_ranks.max(dim=2).values[:, roots:] + 1, graphs.ranks[:, roots:]
    )


def test_handcrafted_order_and_positions() -> None:
    """Two gates whose only difference is slot order must still be ordered.

    Node layout: 0=r0, 1=r1, 2=gate(r1, r0), 3=gate(r0, r1), 4=out(2), 5=out(3).
    Both gates are ready immediately; the tie-break compares slot-ordered parent
    positions, so gate 3 with parents (0, 1) is emitted before gate 2 with (1, 0).
    """
    geometry = Geometry(2, 2, 2, 1, (2,))
    inputs = [[], [], [1, 0], [0, 1], [2], [3]]
    types = [1, 2, 0, 0, 3, 3]

    assert canonical_order_from_lists(inputs, types, geometry) == [0, 1, 3, 2, 4, 5]

    graphs = from_lists(inputs, types, geometry)
    positions = graphs.parent_positions[0]
    assert positions[2].tolist() == [0, 1]
    assert positions[3].tolist() == [1, 0]
    # Outputs keep their slot order and point at the repositioned gates.
    assert int(positions[4, 0]) == 3
    assert int(positions[5, 0]) == 2


def canonical_signature(graphs) -> list[tuple]:
    """Label-free sequence signature: per position, (type, valid parent tuple)."""
    types = graphs.node_types[0]
    positions = graphs.parent_positions[0]
    mask = graphs.parent_slot_mask[0]
    return [
        (int(types[position]), tuple(positions[position][mask[position]].tolist()))
        for position in range(graphs.geometry.num_nodes)
    ]


def relabel_trunks(
    inputs: list[list[int]], types: list[int], geometry: Geometry, rng: random.Random
) -> tuple[list[list[int]], list[int]]:
    """Re-store the trunk nodes in a different valid topological order.

    Roots and outputs keep their storage slots (the layout requires it); trunk
    nodes are re-emitted by a random Kahn walk over trunk->trunk edges and all
    parent references are remapped. Same labeled graph, different storage.
    """
    roots = geometry.num_root_nodes
    output_start = geometry.output_start
    trunk_indices = list(range(roots, output_start))

    blocking = {node: 0 for node in trunk_indices}
    children: dict[int, list[int]] = {node: [] for node in trunk_indices}
    for node in trunk_indices:
        for parent in inputs[node]:
            if parent >= roots:
                blocking[node] += 1
                children[parent].append(node)

    ready = [node for node in trunk_indices if blocking[node] == 0]
    new_trunk_order: list[int] = []
    while ready:
        node = ready.pop(rng.randrange(len(ready)))
        new_trunk_order.append(node)
        for child in children[node]:
            blocking[child] -= 1
            if blocking[child] == 0:
                ready.append(child)
    assert len(new_trunk_order) == len(trunk_indices)

    old_to_new = {old: old for old in range(roots)}
    for offset, old in enumerate(new_trunk_order):
        old_to_new[old] = roots + offset
    for old in range(output_start, geometry.num_nodes):
        old_to_new[old] = old
    new_to_old = {new: old for old, new in old_to_new.items()}

    return (
        [
            [old_to_new[parent] for parent in inputs[new_to_old[new]]]
            for new in range(geometry.num_nodes)
        ],
        [types[new_to_old[new]] for new in range(geometry.num_nodes)],
    )


def has_twin_ties(graphs) -> bool:
    """True when two trunk positions share (type, slot-ordered parent positions).

    Structural twins compare equal under the canonical tie-break, so their
    relative order -- and any consumer's parent tuple -- is not relabel-invariant.
    That residual ambiguity is accepted rather than paying for full canonization,
    so the invariance test has to sample around it.
    """
    geometry = graphs.geometry
    signature = canonical_signature(graphs)
    trunk = signature[geometry.num_root_nodes : geometry.output_start]
    return len(set(trunk)) != len(trunk)


def test_canonical_order_is_relabel_invariant() -> None:
    geometry = SMALL_GEOMETRY
    rng = random.Random(1)
    torch.manual_seed(1)

    checked = 0
    for attempt in range(60):
        node_types, in_degrees, parents, _, _ = raw_arrays(1, geometry, attempt)
        inputs = ragged_parents(parents, in_degrees, 0, geometry.num_nodes)
        types = node_types[0].tolist()
        graphs = from_lists(inputs, types, geometry)
        if has_twin_ties(graphs):
            continue

        reference = canonical_signature(graphs)
        for _ in range(3):
            relabeled_inputs, relabeled_types = relabel_trunks(
                inputs, types, geometry, rng
            )
            relabeled = from_lists(relabeled_inputs, relabeled_types, geometry)
            assert canonical_signature(relabeled) == reference
        checked += 1
        if checked == 10:
            return
    raise AssertionError(f"only found {checked} twin-free graphs to check")


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
    assert torch.equal(first.parent_positions, replay.parent_positions)

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
                    tuple(map(tuple, batch.parent_positions[index].tolist())),
                )
            )
    assert len(signatures) == 32, (
        f"only {len(signatures)} distinct graphs out of 32; the compiled "
        "generator is probably being reseeded identically each call"
    )


def test_sampled_graphs_have_no_orphaned_producers() -> None:
    """What the sampler's coverage pass exists to guarantee."""
    for geometry in (SMALL_GEOMETRY, TRAINING_GEOMETRY):
        torch.manual_seed(0)
        graphs = sample(40, geometry)
        mask = graphs.parent_slot_mask
        for index in range(len(graphs)):
            referenced = set(graphs.parent_positions[index][mask[index]].tolist())
            assert referenced == set(range(geometry.output_start)), (
                f"unused producer in graph {index} at {geometry}"
            )


def test_sampled_distribution_is_unbiased() -> None:
    """Cheap statistics over the sampler's prior, at the training geometry.

    ``ascending`` is the one that catches a broken shuffle: both wiring passes
    append, so without the Fisher-Yates pass slot 0 would always hold the
    coverage parent and the share would sit far from a half.
    """
    geometry = TRAINING_GEOMETRY
    torch.manual_seed(5)
    graphs = sample(600, geometry)
    mask = graphs.parent_slot_mask

    out_degrees = torch.bincount(
        graphs.parent_positions[mask], minlength=geometry.output_start
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

    pairs = graphs.parent_positions[:, geometry.num_root_nodes : geometry.output_start]
    share = float((pairs[..., 0] < pairs[..., 1]).double().mean())
    assert abs(share - 0.5) < 0.02, f"slot order looks biased: {share:.4f} ascending"
