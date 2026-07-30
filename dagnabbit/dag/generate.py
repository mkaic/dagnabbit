"""Compiled random-graph generation.

Generating graph batches is the training loop's inner loop, and it was its
dominant cost: 70-90% of a stage-one step. Successive rewrites in Python took it
from 443 to 190 us/graph, at which point ~70% of what remained was sequential
interpreted work with no vectorized form -- two wiring passes, a longest-path
sweep, and a heap-ordered Kahn sort. Those are what compiled code is good at, and
numba gets the whole pipeline to ~11 us/graph.

Everything is produced into batch-wide preallocated arrays:

===========================  ==========================  ======================
array                        shape                       meaning
===========================  ==========================  ======================
``node_types``               ``[B, N]``                  raw type index per node
``in_degrees``               ``[B, N]``                  filled parent slots
``padded_parents``           ``[B, N, maximum_indegree]``  zero-padded parents
``ranks``                    ``[B, N]``                  longest-path depth
``canonical_order``          ``[B, N]``                  position -> node index
===========================  ==========================  ======================

:func:`sample_graph_batch` wraps those into descriptions via
:meth:`~dagnabbit.dag.description.FixedInDegreeDAGDescription.from_arrays`, which
adopts per-graph slices without recomputing anything.

Why one call per batch
----------------------
The compiled kernel is only worth having if the boundary is crossed rarely. Per
graph, converting a ragged ``list[list[int]]`` in and out would cost several
times the algorithm it replaces -- which is the trap an earlier attempt at this in
another language fell into. One call per batch amortizes the boundary ~256x and
returns flat arrays that ``torch.from_numpy`` wraps for free.

Fidelity to the Python implementation
-------------------------------------
The kernels below reimplement
:func:`~dagnabbit.dag.description.make_random_graph_description` and the two
derived overlays, so the risk is silent divergence. Two things pin them:

* ``ranks`` and ``canonical_order`` must be **bit-identical** to
  :meth:`compute_node_ranks` and :meth:`compute_canonical_order` for the same
  structure. The canonical order is load-bearing -- it defines the sequence
  model's targets, and isomorphic graphs must produce identical sequences -- so
  :func:`_ready_key_less` reproduces Python's tuple comparison exactly, including
  that a shorter tuple which is a prefix of a longer one sorts first.
* the sampled distribution must match statistically.

Both are asserted in ``dagnabbit/dag/tests/test_generate.py``.
"""

import numpy as np
import torch
from numba import njit

from dagnabbit.dag.description import FixedInDegreeDAGDescription

# ---- canonical order: a heap of node indices under a lexicographic key ------
#
# numba has no heapq with a custom comparator, so the heap is written out over an
# int64 array. The key is never materialized; it is compared field by field
# straight out of the arrays, which is also why this is cheap.


@njit(cache=True, inline="always")
def _ready_key_less(left, right, positions, parents, degrees, node_types):
    """Is ``left``'s ready key below ``right``'s?

    The key is ``(parent canonical positions in slot order, node_type, index)``.
    Parent positions are compared in *slot* order, not sorted, because input-slot
    order is semantically meaningful.
    """
    left_degree = degrees[left]
    right_degree = degrees[right]
    shared = left_degree if left_degree < right_degree else right_degree
    for slot in range(shared):
        left_position = positions[parents[left, slot]]
        right_position = positions[parents[right, slot]]
        if left_position != right_position:
            return left_position < right_position
    # A prefix sorts before the longer tuple, matching Python.
    if left_degree != right_degree:
        return left_degree < right_degree
    if node_types[left] != node_types[right]:
        return node_types[left] < node_types[right]
    return left < right


@njit(cache=True)
def _heap_push(heap, size, node, positions, parents, degrees, node_types):
    heap[size] = node
    child = size
    while child > 0:
        parent = (child - 1) // 2
        if _ready_key_less(
            heap[child], heap[parent], positions, parents, degrees, node_types
        ):
            heap[child], heap[parent] = heap[parent], heap[child]
            child = parent
        else:
            break
    return size + 1


@njit(cache=True)
def _heap_pop(heap, size, positions, parents, degrees, node_types):
    top = heap[0]
    size -= 1
    heap[0] = heap[size]
    parent = 0
    while True:
        left = 2 * parent + 1
        if left >= size:
            break
        smallest = left
        right = left + 1
        if right < size and _ready_key_less(
            heap[right], heap[left], positions, parents, degrees, node_types
        ):
            smallest = right
        if _ready_key_less(
            heap[smallest], heap[parent], positions, parents, degrees, node_types
        ):
            heap[parent], heap[smallest] = heap[smallest], heap[parent]
            parent = smallest
        else:
            break
    return top, size


# ---- one graph --------------------------------------------------------------


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
    order,
    positions,
    filled,
    coverable,
    slot_of,
    blocking,
    children_start,
    children_cursor,
    children,
    heap,
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
    # Rejection-sample rather than materialize the filtered pool: at most
    # in_degree - 1 values are excluded from a range at least num_root_nodes
    # wide, so a draw is accepted on the first try in almost every case.
    for consumer in range(num_root_nodes, num_nodes):
        limit = consumer if consumer < output_start else output_start
        while filled[consumer] < in_degrees[consumer]:
            chosen = np.random.randint(0, limit)
            duplicate = True
            while duplicate:
                duplicate = False
                for slot in range(filled[consumer]):
                    if parents[consumer, slot] == chosen:
                        duplicate = True
                        break
                if duplicate:
                    chosen = np.random.randint(0, limit)
            parents[consumer, filled[consumer]] = chosen
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

    # --- canonical order: Kahn's algorithm with the lexicographic tie-break ---
    for node in range(num_nodes):
        positions[node] = -1
    for root in range(num_root_nodes):
        positions[root] = root
        order[root] = root

    # Child lists in CSR form: count into the offsets, prefix-sum, then fill.
    for node in range(output_start + 2):
        children_start[node] = 0
    for node in range(num_root_nodes, output_start):
        blocking[node] = 0
        for slot in range(in_degrees[node]):
            parent = parents[node, slot]
            if parent >= num_root_nodes:
                blocking[node] += 1
                children_start[parent + 1] += 1
    for node in range(output_start):
        children_start[node + 1] += children_start[node]
    for node in range(output_start + 1):
        children_cursor[node] = children_start[node]
    for node in range(num_root_nodes, output_start):
        for slot in range(in_degrees[node]):
            parent = parents[node, slot]
            if parent >= num_root_nodes:
                children[children_cursor[parent]] = node
                children_cursor[parent] += 1

    size = 0
    for node in range(num_root_nodes, output_start):
        if blocking[node] == 0:
            size = _heap_push(
                heap, size, node, positions, parents, in_degrees, node_types
            )

    placed = num_root_nodes
    while size > 0:
        node, size = _heap_pop(heap, size, positions, parents, in_degrees, node_types)
        positions[node] = placed
        order[placed] = node
        placed += 1
        for index in range(children_start[node], children_start[node + 1]):
            child = children[index]
            blocking[child] -= 1
            if blocking[child] == 0:
                size = _heap_push(
                    heap, size, child, positions, parents, in_degrees, node_types
                )

    # Outputs are leaves and pinned to the final positions in slot order.
    for node in range(output_start, num_nodes):
        positions[node] = node
        order[node] = node


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
    output_start = num_root_nodes + num_trunk_nodes

    node_types = np.zeros((count, num_nodes), dtype=np.int64)
    in_degrees = np.zeros((count, num_nodes), dtype=np.int64)
    padded_parents = np.zeros((count, num_nodes, maximum_indegree), dtype=np.int64)
    ranks = np.zeros((count, num_nodes), dtype=np.int64)
    order = np.zeros((count, num_nodes), dtype=np.int64)
    positions = np.zeros((count, num_nodes), dtype=np.int64)

    # Scratch, reused across graphs: allocating per graph would show up.
    filled = np.zeros(num_nodes, dtype=np.int64)
    coverable = np.zeros(num_nodes, dtype=np.int64)
    slot_of = np.full(num_nodes, -1, dtype=np.int64)
    blocking = np.zeros(output_start + 1, dtype=np.int64)
    children_start = np.zeros(output_start + 2, dtype=np.int64)
    children_cursor = np.zeros(output_start + 1, dtype=np.int64)
    children = np.zeros(num_nodes * maximum_indegree, dtype=np.int64)
    heap = np.zeros(num_nodes + 1, dtype=np.int64)

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
            order[index],
            positions[index],
            filled,
            coverable,
            slot_of,
            blocking,
            children_start,
            children_cursor,
            children,
            heap,
        )
    return node_types, in_degrees, padded_parents, ranks, order


def check_geometry(
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    num_trunk_node_types: int,
    trunk_node_in_degrees: list[int],
) -> None:
    """The preconditions the sampler relies on, checked once per batch.

    These used to live inside the per-graph function. They depend only on the
    geometry, so checking them per graph was pure waste -- but they must still be
    checked, because the compiled sampler would otherwise fail obscurely (an
    infeasible coverage pass draws from an empty pool).
    """
    assert len(trunk_node_in_degrees) == num_trunk_node_types
    assert all(in_degree >= 1 for in_degree in trunk_node_in_degrees)
    assert num_root_nodes >= 1
    assert num_output_nodes >= 1
    assert num_trunk_node_types >= 1
    if num_trunk_nodes > 0:
        # The earliest trunk can draw distinct inputs only from the roots, so
        # there must be at least max-in-degree of them.
        assert num_root_nodes >= max(trunk_node_in_degrees), (
            "num_root_nodes must be >= the largest trunk in-degree so every "
            "gate can be given distinct inputs"
        )

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


def sample_graph_batch(
    count: int,
    *,
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    num_trunk_node_types: int,
    trunk_node_in_degrees: int | list[int],
) -> list[FixedInDegreeDAGDescription]:
    """``count`` freshly sampled graphs.

    Seeded from ``torch.randint``, so ``torch.manual_seed`` still determines the
    whole stream exactly as it did when generation was pure Python. numba keeps
    its own RNG state, separate from numpy's, which is why the seed is passed in
    explicitly rather than inherited.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if isinstance(trunk_node_in_degrees, int):
        trunk_node_in_degrees = [trunk_node_in_degrees] * num_trunk_node_types
    else:
        trunk_node_in_degrees = list(trunk_node_in_degrees)
    check_geometry(
        num_root_nodes,
        num_trunk_nodes,
        num_output_nodes,
        num_trunk_node_types,
        trunk_node_in_degrees,
    )

    maximum_indegree = max([1, *trunk_node_in_degrees])
    # numba's seed is a uint32, so draw in that range rather than truncating.
    seed = int(torch.randint(0, 2**32, (1,), dtype=torch.int64).item())

    node_types, in_degrees, padded_parents, ranks, order = generate_arrays(
        count,
        num_root_nodes,
        num_trunk_nodes,
        num_output_nodes,
        num_trunk_node_types,
        np.asarray(trunk_node_in_degrees, dtype=np.int64),
        maximum_indegree,
        seed,
    )

    # One vectorized comparison for the whole batch's slot masks, rather than one
    # per graph. Everything else is already in its final form.
    slot_masks = np.arange(maximum_indegree) < in_degrees[..., None]

    return [
        FixedInDegreeDAGDescription.from_arrays(
            num_root_nodes=num_root_nodes,
            num_trunk_nodes=num_trunk_nodes,
            num_output_nodes=num_output_nodes,
            num_trunk_node_types=num_trunk_node_types,
            trunk_node_in_degrees=trunk_node_in_degrees,
            node_types_array=node_types[index],
            padded_parents=padded_parents[index],
            parent_slot_mask=slot_masks[index],
            ranks_array=ranks[index],
            canonical_order_array=order[index],
        )
        for index in range(count)
    ]
