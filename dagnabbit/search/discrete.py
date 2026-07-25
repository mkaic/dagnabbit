"""Evolutionary search directly on graph structure.

A (mu + lambda) evolution strategy with point mutations, in the style of
Cartesian Genetic Programming -- the standard approach for evolving logic
circuits, and the baseline any latent-space method has to beat to justify
itself.

Two details do most of the work in CGP and are kept here:

* **Neutral drift.** Offspring that merely *tie* the parent replace it. Circuit
  fitness landscapes are full of plateaus, and the ability to wander across one
  while dragging along silent structural changes is what eventually finds the
  edge of it. Without this the search stalls almost immediately.
* **Dead genes are free.** Nodes that feed no output cost nothing to carry, and
  act as a reservoir of material that mutation can splice back in later. The
  fixed node count means every genome has this reservoir automatically.

Every mutation produces a structurally valid DAG by construction: parents are
only ever redrawn from strictly-earlier non-output nodes, and a type change
resizes the input list to that type's in-degree.
"""

import random
from collections.abc import Sequence

import torch

from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_random_graph_description,
)
from dagnabbit.search.common import DEFAULT_EVAL_CHUNK, SearchLoop, SearchResult
from dagnabbit.tasks.logic_gates.evaluate import BitpackedTask


def _random_parent(node: int, output_start: int, rng: random.Random) -> int:
    """A legal parent for ``node``: strictly earlier, and never an output."""
    return rng.randrange(min(node, output_start))


def _force_one_change(
    graph: FixedInDegreeDAGDescription,
    node_types: list[int],
    node_inputs: list[list[int]],
    output_start: int,
    rng: random.Random,
) -> None:
    """Change exactly one gene to a genuinely *different* value.

    Redrawing a parent uniformly can land on the value already there, so the
    replacement is drawn from the candidates excluding the current one -- an
    "ensured" mutation that silently produced a clone would defeat the point.
    Falls back to a type flip for the degenerate case of a node with only one
    legal parent, and does nothing if the genome truly has no alternative.
    """
    node = rng.randrange(graph.num_root_nodes, graph.num_nodes)
    num_candidates = min(node, output_start)

    if num_candidates > 1:
        slot = rng.randrange(len(node_inputs[node]))
        replacement = rng.randrange(num_candidates - 1)
        if replacement >= node_inputs[node][slot]:
            replacement += 1
        node_inputs[node][slot] = replacement
    elif node < output_start and graph.num_trunk_node_types > 1:
        replacement = rng.randrange(graph.num_trunk_node_types - 1)
        if replacement >= node_types[node]:
            replacement += 1
        node_types[node] = replacement


def mutate_graph(
    graph: FixedInDegreeDAGDescription,
    rng: random.Random,
    mutation_rate: float,
    ensure_mutation: bool = True,
) -> FixedInDegreeDAGDescription:
    """Point-mutate a graph's type and connection genes.

    Each gene -- one type per trunk node, one parent per input slot -- is
    redrawn independently with probability ``mutation_rate``. Type mutations
    always land on a different type. Connection mutations redraw uniformly over
    the legal parents and so may, as in standard CGP, happen to re-pick the
    current one; with ~80 candidates per slot at the default geometry that is a
    ~1% no-op rate, well inside the noise of the rate itself.

    With ``ensure_mutation`` a genome that drew no mutations at all gets one
    forced change that is guaranteed to differ, so an offspring is never a pure
    clone of its parent (which would burn an evaluation to learn nothing).
    """
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}")

    node_types = list(graph.node_types)
    node_inputs = [list(parents) for parents in graph.node_inputs_indices]
    output_start = graph.num_root_nodes + graph.num_trunk_nodes
    can_change_type = graph.num_trunk_node_types > 1
    mutations = 0

    for node in range(graph.num_root_nodes, output_start):
        if can_change_type and rng.random() < mutation_rate:
            # Draw from the other types by skipping over the current one.
            replacement = rng.randrange(graph.num_trunk_node_types - 1)
            if replacement >= node_types[node]:
                replacement += 1
            node_types[node] = replacement
            mutations += 1

            # A type carries its own in-degree, so resize the slot list to
            # match. Truncation drops trailing slots; growth fills with fresh
            # legal parents.
            in_degree = graph.trunk_node_in_degrees[replacement]
            slots = node_inputs[node]
            del slots[in_degree:]
            while len(slots) < in_degree:
                slots.append(_random_parent(node, output_start, rng))

        for slot in range(len(node_inputs[node])):
            if rng.random() < mutation_rate:
                node_inputs[node][slot] = _random_parent(node, output_start, rng)
                mutations += 1

    for node in range(output_start, graph.num_nodes):
        if rng.random() < mutation_rate:
            node_inputs[node][0] = _random_parent(node, output_start, rng)
            mutations += 1

    if ensure_mutation and mutations == 0:
        _force_one_change(graph, node_types, node_inputs, output_start, rng)

    return FixedInDegreeDAGDescription(
        num_root_nodes=graph.num_root_nodes,
        num_trunk_nodes=graph.num_trunk_nodes,
        num_output_nodes=graph.num_output_nodes,
        num_trunk_node_types=graph.num_trunk_node_types,
        trunk_node_in_degrees=graph.trunk_node_in_degrees,
        node_inputs_indices=node_inputs,
        node_types=node_types,
    )


def evolve_discrete(
    task: BitpackedTask,
    *,
    num_generations: int,
    num_parents: int = 1,
    num_offspring: int = 16,
    mutation_rate: float = 0.03,
    seed: int = 0,
    initial_graphs: Sequence[FixedInDegreeDAGDescription] | None = None,
    num_root_nodes: int = 16,
    num_trunk_nodes: int = 128,
    num_output_nodes: int = 8,
    num_trunk_node_types: int = 2,
    trunk_node_in_degrees: int | list[int] = 2,
    target_fitness: float = 1.0,
    chunk_size: int = DEFAULT_EVAL_CHUNK,
    on_generation=None,
) -> SearchResult:
    """Run a (mu + lambda) ES on graph structure.

    ``initial_graphs`` seeds the parent pool -- pass a known-good circuit to
    warm-start, or leave it out to start from random graphs. Parent fitness is
    carried between generations, so each generation costs exactly
    ``num_offspring`` evaluations.
    """
    if num_parents < 1:
        raise ValueError("num_parents must be at least 1")
    if num_offspring < 1:
        raise ValueError("num_offspring must be at least 1")

    rng = random.Random(seed)
    loop = SearchLoop(
        task=task,
        num_generations=num_generations,
        target_fitness=target_fitness,
        chunk_size=chunk_size,
        on_generation=on_generation,
    )

    if initial_graphs is None:
        # make_random_graph_description seeds its own random.Random from
        # torch.randint, so torch -- not the stdlib global RNG -- is the
        # channel that makes initial graphs reproducible from ``seed``.
        torch.manual_seed(seed)
        parents = [
            make_random_graph_description(
                num_root_nodes=num_root_nodes,
                num_trunk_nodes=num_trunk_nodes,
                num_output_nodes=num_output_nodes,
                trunk_node_in_degrees=trunk_node_in_degrees,
                num_trunk_node_types=num_trunk_node_types,
            )
            for _ in range(num_parents)
        ]
    else:
        parents = list(initial_graphs)[:num_parents]
        if not parents:
            raise ValueError("initial_graphs was empty")

    parent_fitness = loop.score(parents)
    loop.record(generation=0, fitness=parent_fitness)
    scored_parents = list(zip(parents, parent_fitness.tolist()))

    for generation in range(1, num_generations + 1):
        if loop.should_stop:
            break

        offspring = [
            mutate_graph(
                scored_parents[index % len(scored_parents)][0], rng, mutation_rate
            )
            for index in range(num_offspring)
        ]
        offspring_fitness = loop.score(offspring)

        # Neutral drift: sort so that on equal fitness the offspring (rank key
        # 0) precedes the incumbent parent (key 1) and takes its place.
        pool = [
            (fitness, 0, graph)
            for graph, fitness in zip(offspring, offspring_fitness.tolist())
        ] + [(fitness, 1, graph) for graph, fitness in scored_parents]
        pool.sort(key=lambda entry: (-entry[0], entry[1]))
        scored_parents = [(graph, fitness) for fitness, _, graph in pool[:num_parents]]

        loop.record(generation=generation, fitness=offspring_fitness)

    return loop.result()
