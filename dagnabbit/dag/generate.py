"""Compiled random-graph generation.

Generating graph batches used to be the training loop's dominant cost: 70-90% of
a step. Successive rewrites in Python took it from 443 to 190 us/graph, at which
point ~70% of what remained was sequential interpreted work with no vectorized
form. That is what compiled code is good at, and numba gets the whole pipeline
to roughly 10 us/graph.

Everything is produced into batch-wide preallocated arrays:

===========================  ============================  ======================
array                        shape                         meaning
===========================  ============================  ======================
``node_types``               ``[B, N]``                    raw type index per node
``in_degrees``               ``[B, N]``                    filled parent slots
``padded_parents``           ``[B, N, maximum_indegree]``  zero-padded parents
``ranks``                    ``[B, N]``                    longest-path depth
===========================  ============================  ======================

Node index order *is* a valid topological order: roots occupy ``[0, R)``,
outputs the final positions, and every non-root references only earlier indices.
Nothing downstream needs more than that, so no separate ordering is computed.

Duplicate parents
-----------------
A gate may draw the same producer into both slots. ``NAND(x, x) = NOT x``, so
this is how inverters exist at all -- without it the sampler could not express
one, while the hand-built reference adder is ~48% inverters. Coverage is
unaffected: pass 1 places every producer into some consumer before pass 2 fills
what is left, so the two concerns are independent and no node is ever dead.

Why one call per batch
----------------------
The compiled kernel is only worth having if the boundary is crossed rarely. Per
graph, converting a ragged ``list[list[int]]`` in and out would cost several
times the algorithm it replaces -- which is the trap an earlier attempt at this
in another language fell into. One call per batch amortizes the boundary ~256x
and returns flat arrays that ``torch.from_numpy`` wraps for free.

Fidelity
--------
``ranks`` must stay bit-identical to
:func:`~dagnabbit.dag.graphs.ranks_from_lists`, and the sampled distribution
must match statistically. Both are asserted in
``dagnabbit/dag/tests/test_graphs.py``.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _generate_one(
    num_root_nodes,
    num_trunk_nodes,
    num_output_nodes,
    num_trunk_node_types,
    type_in_degrees,
    node_types,
    in_degrees,
    parents,
    ranks,
    filled,
    coverable,
    slot_of,
):
    """Fill one graph's row of every output array. Scratch arrays are reused."""
    num_nodes = num_root_nodes + num_trunk_nodes + num_output_nodes
    output_start = num_root_nodes + num_trunk_nodes

    # --- types, and the in-degree each type implies ---
    for root in range(num_root_nodes):
        node_types[root] = num_trunk_node_types + root
        in_degrees[root] = 0
    for offset in range(num_trunk_nodes):
        node = num_root_nodes + offset
        trunk_type = np.random.randint(0, num_trunk_node_types)
        node_types[node] = trunk_type
        in_degrees[node] = type_in_degrees[trunk_type]
    for offset in range(num_output_nodes):
        node = output_start + offset
        # All outputs share the single output type.
        node_types[node] = num_trunk_node_types + num_root_nodes
        in_degrees[node] = 1

    for node in range(num_nodes):
        filled[node] = 0

    # --- pass 1: coverage, producers latest-first ---
    #
    # This is what guarantees no dead nodes, and it runs to completion before
    # pass 2 draws anything, which is why allowing duplicates below cannot
    # orphan a producer.
    #
    # ``coverable`` holds exactly the consumers eligible for the producer being
    # processed: strictly later than it, and still holding an open slot. Walking
    # producers latest-first is what lets it be maintained incrementally -- the
    # "strictly later" set only grows, one node per step -- and ``slot_of`` makes
    # a full consumer a constant-time swap-removal.
    size = 0
    for consumer in range(output_start, num_nodes):
        slot_of[consumer] = size
        coverable[size] = consumer
        size += 1

    for producer in range(output_start - 1, -1, -1):
        consumer = coverable[np.random.randint(0, size)]
        parents[consumer, filled[consumer]] = producer
        filled[consumer] += 1
        if filled[consumer] == in_degrees[consumer]:
            index = slot_of[consumer]
            size -= 1
            moved = coverable[size]
            if moved != consumer:
                coverable[index] = moved
                slot_of[moved] = index
            slot_of[consumer] = -1
        # The next producer is earlier, so this one becomes a legal consumer.
        if in_degrees[producer] > 0:
            slot_of[producer] = size
            coverable[size] = producer
            size += 1

    # --- pass 2: fill every remaining slot from a strictly-earlier producer ---
    #
    # Drawn uniformly with no rejection: a repeat of a slot already filled is a
    # legal gate, not a collision to resample away. See the module docstring.
    for consumer in range(num_root_nodes, num_nodes):
        limit = consumer if consumer < output_start else output_start
        while filled[consumer] < in_degrees[consumer]:
            parents[consumer, filled[consumer]] = np.random.randint(0, limit)
            filled[consumer] += 1

    # --- erase the slot-position artifact the two passes leave behind ---
    for node in range(num_root_nodes, num_nodes):
        for slot in range(in_degrees[node] - 1, 0, -1):
            swap = np.random.randint(0, slot + 1)
            held = parents[node, slot]
            parents[node, slot] = parents[node, swap]
            parents[node, swap] = held

    # --- longest-path ranks; nodes are already in topological order ---
    for node in range(num_root_nodes):
        ranks[node] = 0
    for node in range(num_root_nodes, num_nodes):
        deepest = 0
        for slot in range(in_degrees[node]):
            parent_rank = ranks[parents[node, slot]]
            if parent_rank > deepest:
                deepest = parent_rank
        ranks[node] = deepest + 1


@njit(cache=True)
def generate_arrays(
    count,
    num_root_nodes,
    num_trunk_nodes,
    num_output_nodes,
    num_trunk_node_types,
    type_in_degrees,
    maximum_indegree,
    seed,
):
    """``count`` graphs into batch-wide arrays. The only compiled entry point."""
    np.random.seed(seed)
    num_nodes = num_root_nodes + num_trunk_nodes + num_output_nodes

    node_types = np.zeros((count, num_nodes), dtype=np.int64)
    in_degrees = np.zeros((count, num_nodes), dtype=np.int64)
    padded_parents = np.zeros((count, num_nodes, maximum_indegree), dtype=np.int64)
    ranks = np.zeros((count, num_nodes), dtype=np.int64)

    # Scratch, reused across graphs: allocating per graph would show up.
    filled = np.zeros(num_nodes, dtype=np.int64)
    coverable = np.zeros(num_nodes, dtype=np.int64)
    slot_of = np.full(num_nodes, -1, dtype=np.int64)

    for index in range(count):
        _generate_one(
            num_root_nodes,
            num_trunk_nodes,
            num_output_nodes,
            num_trunk_node_types,
            type_in_degrees,
            node_types[index],
            in_degrees[index],
            padded_parents[index],
            ranks[index],
            filled,
            coverable,
            slot_of,
        )
    return node_types, in_degrees, padded_parents, ranks


def check_geometry(
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    num_trunk_node_types: int,
    trunk_node_in_degrees: list[int],
) -> None:
    """The preconditions the sampler relies on, checked once per batch.

    These depend only on the geometry, so checking them per graph would be pure
    waste -- but they must still be checked, because the compiled sampler would
    otherwise fail obscurely (an infeasible coverage pass draws from an empty
    pool, and njit does no bounds checking).
    """
    assert len(trunk_node_in_degrees) == num_trunk_node_types
    assert all(in_degree >= 1 for in_degree in trunk_node_in_degrees)
    assert num_root_nodes >= 1
    assert num_output_nodes >= 1
    assert num_trunk_node_types >= 1

    # Coverage feasibility: with every trunk at the smallest possible in-degree,
    # the downstream input slots must still outnumber the producers enough to
    # give each root (and trunk) a child. A producer can only be consumed by a
    # strictly-later node, so the producer->slot neighbourhoods are nested and
    # the latest-first greedy assignment saturates whenever this holds.
    min_in_degree = min(trunk_node_in_degrees)
    max_coverable_roots = num_trunk_nodes * (min_in_degree - 1) + num_output_nodes
    assert num_root_nodes <= max_coverable_roots, (
        f"to guarantee every root is used, need num_root_nodes "
        f"({num_root_nodes}) <= num_trunk_nodes * (min_in_degree - 1) + "
        f"num_output_nodes ({num_trunk_nodes} * {min_in_degree - 1} + "
        f"{num_output_nodes} = {max_coverable_roots}); add trunk nodes "
        "(ideally with in-degree > 1) or output nodes"
    )
