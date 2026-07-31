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

Variable gate count
-------------------
The trunk width is a *maximum*, not a fixed size: each graph draws its own
``T`` in ``[minimum_trunk_nodes, maximum_trunk_nodes]`` and the trailing
``maximum - T`` trunk positions are filled with the ``<MASK>`` type at in-degree
0. The sequence length never changes, so the batch stays one dense tensor and
the model's position table keeps one meaning per index -- only the number of
positions holding an actual gate varies.

Live gates occupy a prefix ``[R, R + T)`` of the trunk block rather than being
scattered through it. Wiring then needs no remapping (a gate at index ``i``
still draws from ``[0, i)``), and it puts the hand-built circuits on the same
footing: their cores are already emitted in topological order from ``R``, so
the same circuit can be presented with real padding gates or with ``<MASK>``.
Nothing reads a masked position -- every parent drawn below is strictly less
than ``R + T`` -- so whatever the evaluator computes there is inert.

Gate-type mixture
-----------------
Each graph also draws its own distribution over trunk types from a symmetric
Dirichlet, rather than every graph using the same uniform mix. At
concentration 1 that is uniform over the simplex, so a batch contains
near-even graphs, ~90/5/5 graphs, and everything between, while the marginal
over the whole batch stays uniform. Hand-built circuits are extremely lopsided
-- the all-NAND adder is 100% one type -- and a sampler that only ever produced
even mixtures never shows the model one.

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
    minimum_trunk_nodes,
    maximum_trunk_nodes,
    num_output_nodes,
    num_trunk_node_types,
    trunk_type_concentration,
    mask_type,
    type_in_degrees,
    node_types,
    in_degrees,
    parents,
    ranks,
    filled,
    coverable,
    slot_of,
    type_cumulative,
):
    """Fill one graph's row of every output array. Scratch arrays are reused."""
    num_nodes = num_root_nodes + maximum_trunk_nodes + num_output_nodes
    output_start = num_root_nodes + maximum_trunk_nodes

    # How many trunk positions this graph actually spends on gates. The rest
    # trail it as <MASK>, and ``producer_limit`` -- not ``output_start`` -- is
    # what everything below draws parents from.
    if minimum_trunk_nodes < maximum_trunk_nodes:
        num_trunk_nodes = np.random.randint(
            minimum_trunk_nodes, maximum_trunk_nodes + 1
        )
    else:
        num_trunk_nodes = maximum_trunk_nodes
    producer_limit = num_root_nodes + num_trunk_nodes

    # --- this graph's mixture over trunk types, as a cumulative distribution ---
    #
    # Dirichlet(concentration) via normalized gammas, which is the only form
    # available here -- numba has the gamma but not the Dirichlet.
    total = 0.0
    if trunk_type_concentration > 0.0:
        for trunk_type in range(num_trunk_node_types):
            weight = np.random.gamma(trunk_type_concentration, 1.0)
            type_cumulative[trunk_type] = weight
            total += weight
    if total > 0.0:
        running = 0.0
        for trunk_type in range(num_trunk_node_types):
            running += type_cumulative[trunk_type] / total
            type_cumulative[trunk_type] = running
    else:
        # Mixture sampling off, or a degenerate draw at tiny concentration.
        for trunk_type in range(num_trunk_node_types):
            type_cumulative[trunk_type] = (trunk_type + 1) / num_trunk_node_types
    # Guard the search below against the last bucket falling short of 1.
    type_cumulative[num_trunk_node_types - 1] = 1.0

    # --- types, and the in-degree each type implies ---
    for root in range(num_root_nodes):
        node_types[root] = num_trunk_node_types + root
        in_degrees[root] = 0
    for offset in range(num_trunk_nodes):
        node = num_root_nodes + offset
        draw = np.random.random()
        trunk_type = num_trunk_node_types - 1
        for candidate in range(num_trunk_node_types):
            if draw < type_cumulative[candidate]:
                trunk_type = candidate
                break
        node_types[node] = trunk_type
        in_degrees[node] = type_in_degrees[trunk_type]
    for node in range(producer_limit, output_start):
        # Unused trunk positions. In-degree 0 keeps them out of both wiring
        # passes below and leaves their parent slots at the zeros they were
        # allocated with.
        node_types[node] = mask_type
        in_degrees[node] = 0
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

    for producer in range(producer_limit - 1, -1, -1):
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
        limit = consumer if consumer < producer_limit else producer_limit
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
        # Only <MASK> reaches here with no parents, and it sits on no path at
        # all; 0 rather than 1 keeps it out of the depth histogram entirely.
        if in_degrees[node] == 0:
            ranks[node] = 0
            continue
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
    minimum_trunk_nodes,
    maximum_trunk_nodes,
    num_output_nodes,
    num_trunk_node_types,
    trunk_type_concentration,
    mask_type,
    type_in_degrees,
    maximum_indegree,
    seed,
):
    """``count`` graphs into batch-wide arrays. The only compiled entry point."""
    np.random.seed(seed)
    num_nodes = num_root_nodes + maximum_trunk_nodes + num_output_nodes

    node_types = np.zeros((count, num_nodes), dtype=np.int64)
    in_degrees = np.zeros((count, num_nodes), dtype=np.int64)
    padded_parents = np.zeros((count, num_nodes, maximum_indegree), dtype=np.int64)
    ranks = np.zeros((count, num_nodes), dtype=np.int64)

    # Scratch, reused across graphs: allocating per graph would show up.
    filled = np.zeros(num_nodes, dtype=np.int64)
    coverable = np.zeros(num_nodes, dtype=np.int64)
    slot_of = np.full(num_nodes, -1, dtype=np.int64)
    type_cumulative = np.zeros(num_trunk_node_types, dtype=np.float64)

    for index in range(count):
        _generate_one(
            num_root_nodes,
            minimum_trunk_nodes,
            maximum_trunk_nodes,
            num_output_nodes,
            num_trunk_node_types,
            trunk_type_concentration,
            mask_type,
            type_in_degrees,
            node_types[index],
            in_degrees[index],
            padded_parents[index],
            ranks[index],
            filled,
            coverable,
            slot_of,
            type_cumulative,
        )
    return node_types, in_degrees, padded_parents, ranks


def check_geometry(
    num_root_nodes: int,
    minimum_trunk_nodes: int,
    maximum_trunk_nodes: int,
    num_output_nodes: int,
    num_trunk_node_types: int,
    trunk_node_in_degrees: list[int],
) -> None:
    """The preconditions the sampler relies on, checked once per batch.

    These depend only on the geometry and the sampling bounds, so checking them
    per graph would be pure waste -- but they must still be checked, because the
    compiled sampler would otherwise fail obscurely (an infeasible coverage pass
    draws from an empty pool, and njit does no bounds checking).
    """
    assert len(trunk_node_in_degrees) == num_trunk_node_types
    assert all(in_degree >= 1 for in_degree in trunk_node_in_degrees)
    assert num_root_nodes >= 1
    assert num_output_nodes >= 1
    assert num_trunk_node_types >= 1
    assert 0 <= minimum_trunk_nodes <= maximum_trunk_nodes, (
        f"need 0 <= minimum_trunk_nodes ({minimum_trunk_nodes}) <= "
        f"num_trunk_nodes ({maximum_trunk_nodes})"
    )

    # Coverage feasibility: with every trunk at the smallest possible in-degree,
    # the downstream input slots must still outnumber the producers enough to
    # give each root (and trunk) a child. A producer can only be consumed by a
    # strictly-later node, so the producer->slot neighbourhoods are nested and
    # the latest-first greedy assignment saturates whenever this holds.
    #
    # Checked at the *minimum* trunk count: that is the graph with the fewest
    # consuming slots, and every larger draw only has more room.
    min_in_degree = min(trunk_node_in_degrees)
    max_coverable_roots = minimum_trunk_nodes * (min_in_degree - 1) + num_output_nodes
    assert num_root_nodes <= max_coverable_roots, (
        f"to guarantee every root is used, need num_root_nodes "
        f"({num_root_nodes}) <= minimum_trunk_nodes * (min_in_degree - 1) + "
        f"num_output_nodes ({minimum_trunk_nodes} * {min_in_degree - 1} + "
        f"{num_output_nodes} = {max_coverable_roots}); add trunk nodes "
        "(ideally with in-degree > 1) or output nodes, or raise "
        "minimum_trunk_nodes"
    )
