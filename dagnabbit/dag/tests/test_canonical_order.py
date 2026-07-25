"""Tests for the structure-canonical topological sequence overlay.

The canonical order feeds the sequence compressor/decoder: roots pinned to the
first positions in slot order, outputs pinned to the last positions in slot
order, trunk nodes ordered by Kahn's algorithm with a lexicographic
(parent-positions-in-slot-order, node_type) tie-break.

Run directly for a quick eyeball pass::

    python -m dagnabbit.dag.tests.test_canonical_order
"""

import random

import torch

from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_random_graph_description,
)

NUM_ROOTS = 8
NUM_TRUNKS = 24
NUM_OUTPUTS = 4
NUM_TRUNK_TYPES = 2
IN_DEGREES = 2


def sample_graph() -> FixedInDegreeDAGDescription:
    return make_random_graph_description(
        num_root_nodes=NUM_ROOTS,
        num_trunk_nodes=NUM_TRUNKS,
        num_output_nodes=NUM_OUTPUTS,
        trunk_node_in_degrees=IN_DEGREES,
        num_trunk_node_types=NUM_TRUNK_TYPES,
    )


def handcrafted_graph() -> FixedInDegreeDAGDescription:
    """Two roots, two trunk gates with mirrored slot orders, two outputs.

    Node layout (original indices): 0=r0, 1=r1, 2=gate(r1, r0), 3=gate(r0, r1),
    4=out(2), 5=out(3). Both gates are ready immediately; the tie-break
    compares slot-ordered parent positions, so gate 3 with parents (0, 1) must
    be emitted before gate 2 with parents (1, 0).
    """
    return FixedInDegreeDAGDescription(
        num_root_nodes=2,
        num_trunk_nodes=2,
        num_output_nodes=2,
        num_trunk_node_types=1,
        trunk_node_in_degrees=[2],
        node_inputs_indices=[[], [], [1, 0], [0, 1], [2], [3]],
        node_types=[1, 2, 0, 0, 3, 3],
    )


def relabel_trunks(
    graph: FixedInDegreeDAGDescription,
    rng: random.Random,
) -> FixedInDegreeDAGDescription:
    """Rebuild the graph with trunk nodes stored in a different valid topo order.

    Roots and outputs keep their storage slots (the representation requires
    it); trunk nodes are re-emitted by a random Kahn walk over trunk->trunk
    edges, then all parent references are remapped. The result describes the
    same labeled graph, only with different trunk storage indices.
    """
    output_start = graph.num_root_nodes + graph.num_trunk_nodes
    trunk_indices = list(range(graph.num_root_nodes, output_start))

    blocking = {node_idx: 0 for node_idx in trunk_indices}
    children: dict[int, list[int]] = {node_idx: [] for node_idx in trunk_indices}
    for node_idx in trunk_indices:
        for parent in graph.node_inputs_indices[node_idx]:
            if parent >= graph.num_root_nodes:
                blocking[node_idx] += 1
                children[parent].append(node_idx)

    ready = [node_idx for node_idx in trunk_indices if blocking[node_idx] == 0]
    new_trunk_order: list[int] = []
    while ready:
        node_idx = ready.pop(rng.randrange(len(ready)))
        new_trunk_order.append(node_idx)
        for child in children[node_idx]:
            blocking[child] -= 1
            if blocking[child] == 0:
                ready.append(child)
    assert len(new_trunk_order) == len(trunk_indices)

    old_to_new = {old: old for old in range(graph.num_root_nodes)}
    for offset, old in enumerate(new_trunk_order):
        old_to_new[old] = graph.num_root_nodes + offset
    for old in range(output_start, graph.num_nodes):
        old_to_new[old] = old

    new_to_old = {new: old for old, new in old_to_new.items()}
    node_inputs_indices = [
        [old_to_new[parent] for parent in graph.node_inputs_indices[new_to_old[new]]]
        for new in range(graph.num_nodes)
    ]
    node_types = [graph.node_types[new_to_old[new]] for new in range(graph.num_nodes)]

    return FixedInDegreeDAGDescription(
        num_root_nodes=graph.num_root_nodes,
        num_trunk_nodes=graph.num_trunk_nodes,
        num_output_nodes=graph.num_output_nodes,
        num_trunk_node_types=graph.num_trunk_node_types,
        trunk_node_in_degrees=graph.trunk_node_in_degrees,
        node_inputs_indices=node_inputs_indices,
        node_types=node_types,
    )


def canonical_signature(graph: FixedInDegreeDAGDescription) -> list[tuple]:
    """Label-free sequence signature: per position, (type, parent positions)."""
    signature = []
    for position in range(graph.num_nodes):
        mask = graph.canonical_parent_slot_mask[position]
        parents = tuple(graph.canonical_parent_positions[position][mask].tolist())
        signature.append((int(graph.canonical_node_types[position]), parents))
    return signature


def has_twin_ties(graph: FixedInDegreeDAGDescription) -> bool:
    """True when two trunk positions share (type, slot-ordered parent positions).

    Such structural twins compare equal under the canonical tie-break, so their
    relative order (and any consumer's parent tuple) is not relabel-invariant.
    """
    output_start = graph.num_root_nodes + graph.num_trunk_nodes
    signature = canonical_signature(graph)
    seen = set()
    for position in range(graph.num_root_nodes, output_start):
        key = signature[position]
        if key in seen:
            return True
        seen.add(key)
    return False


def sample_twin_free_graph(max_attempts: int = 50) -> FixedInDegreeDAGDescription:
    for _ in range(max_attempts):
        graph = sample_graph()
        if not has_twin_ties(graph):
            return graph
    raise AssertionError("could not sample a twin-free graph")


def test_handcrafted_order_and_targets() -> None:
    graph = handcrafted_graph()
    assert graph.canonical_order == [0, 1, 3, 2, 4, 5]
    # Gate stored at index 3 (parents r0, r1) sits at position 2; gate stored
    # at index 2 (parents r1, r0) sits at position 3.
    assert graph.canonical_parent_positions[2].tolist() == [0, 1]
    assert graph.canonical_parent_positions[3].tolist() == [1, 0]
    # Outputs keep their slot order and point at the repositioned gates.
    assert graph.canonical_parent_positions[4].tolist()[:1] == [3]
    assert graph.canonical_parent_positions[5].tolist()[:1] == [2]


def test_canonical_order_basic_properties() -> None:
    torch.manual_seed(0)
    for _ in range(10):
        graph = sample_graph()
        output_start = graph.num_root_nodes + graph.num_trunk_nodes

        assert sorted(graph.canonical_order) == list(range(graph.num_nodes))
        # Roots pinned to the first positions in slot order; outputs pinned to
        # the final positions in slot order.
        assert graph.canonical_order[: graph.num_root_nodes] == list(
            range(graph.num_root_nodes)
        )
        assert graph.canonical_order[output_start:] == list(
            range(output_start, graph.num_nodes)
        )
        # Trunk positions hold trunk nodes only.
        for position in range(graph.num_root_nodes, output_start):
            assert graph.num_root_nodes <= graph.canonical_order[position] < output_start

        # Node types in sequence order match a direct gather.
        expected_types = [graph.node_types[idx] for idx in graph.canonical_order]
        assert graph.canonical_node_types.tolist() == expected_types

        # Slot masks mirror in-degrees; every valid target is strictly earlier
        # than its consumer and strictly before the output block.
        for position, node_idx in enumerate(graph.canonical_order):
            in_degree = len(graph.node_inputs_indices[node_idx])
            mask = graph.canonical_parent_slot_mask[position]
            assert int(mask.sum()) == in_degree
            assert not mask[in_degree:].any()
            for slot in range(in_degree):
                target = int(graph.canonical_parent_positions[position, slot])
                assert target < position
                assert target < output_start
                # The target really is that slot's parent, repositioned.
                parent = graph.node_inputs_indices[node_idx][slot]
                assert target == graph.canonical_positions[parent]


def test_canonical_order_is_relabel_invariant() -> None:
    torch.manual_seed(1)
    rng = random.Random(1)
    for _ in range(10):
        graph = sample_twin_free_graph()
        reference = canonical_signature(graph)
        for _ in range(3):
            relabeled = relabel_trunks(graph, rng)
            assert canonical_signature(relabeled) == reference


def main() -> None:
    test_handcrafted_order_and_targets()
    test_canonical_order_basic_properties()
    test_canonical_order_is_relabel_invariant()
    print("ALL CANONICAL-ORDER CHECKS PASSED")


if __name__ == "__main__":
    main()
