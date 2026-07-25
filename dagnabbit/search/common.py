"""Shared scaffolding for circuit search: scoring, bookkeeping, results."""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from dagnabbit.dag.description import FixedInDegreeDAGDescription
from dagnabbit.tasks.logic_gates.evaluate import (
    BitpackedTask,
    bit_accuracy,
    evaluate_graphs,
)

# Circuits per evaluation chunk. The evaluator holds a [chunk, num_nodes,
# num_words] uint8 buffer, which for the adder table is ~1.2 MB per circuit,
# so an unbounded population would happily allocate tens of gigabytes.
DEFAULT_EVAL_CHUNK = 256


@dataclass
class GenerationRecord:
    """One generation's summary, for plotting and for stall detection."""

    generation: int
    evaluations: int
    best_fitness: float
    mean_fitness: float
    elapsed_seconds: float


@dataclass
class SearchResult:
    best_graph: FixedInDegreeDAGDescription
    best_fitness: float
    evaluations: int
    history: list[GenerationRecord] = field(default_factory=list)

    @property
    def generations(self) -> int:
        return len(self.history)


def score_population(
    graphs: Sequence[FixedInDegreeDAGDescription],
    task: BitpackedTask,
    chunk_size: int = DEFAULT_EVAL_CHUNK,
) -> torch.Tensor:
    """Fitness of every graph, as a [P] float64 CPU tensor in [0, 1].

    Chunked so population size and evaluator memory stay decoupled.
    """
    if not graphs:
        return torch.empty(0, dtype=torch.float64)
    scores = [
        bit_accuracy(evaluate_graphs(graphs[start : start + chunk_size], task), task)[0]
        for start in range(0, len(graphs), chunk_size)
    ]
    return torch.cat(scores)


ProgressCallback = Callable[[GenerationRecord], None]


class SearchLoop:
    """Bookkeeping shared by every search: timing, best-so-far, stopping.

    Keeps the algorithms themselves free of logging and termination logic, so
    the difference between them is only how they propose the next population.
    """

    def __init__(
        self,
        task: BitpackedTask,
        num_generations: int,
        target_fitness: float = 1.0,
        chunk_size: int = DEFAULT_EVAL_CHUNK,
        on_generation: ProgressCallback | None = None,
    ):
        self.task = task
        self.num_generations = num_generations
        self.target_fitness = target_fitness
        self.chunk_size = chunk_size
        self.on_generation = on_generation

        self.best_graph: FixedInDegreeDAGDescription | None = None
        self.best_fitness = float("-inf")
        self.evaluations = 0
        self.history: list[GenerationRecord] = []
        self._start = time.perf_counter()

    def score(
        self,
        graphs: Sequence[FixedInDegreeDAGDescription],
    ) -> torch.Tensor:
        """Score a population and fold it into the best-so-far."""
        fitness = score_population(graphs, self.task, self.chunk_size)
        self.evaluations += len(graphs)
        if fitness.numel():
            best_index = int(fitness.argmax())
            if float(fitness[best_index]) > self.best_fitness:
                self.best_fitness = float(fitness[best_index])
                self.best_graph = graphs[best_index]
        return fitness

    def record(self, generation: int, fitness: torch.Tensor) -> GenerationRecord:
        record = GenerationRecord(
            generation=generation,
            evaluations=self.evaluations,
            best_fitness=self.best_fitness,
            mean_fitness=float(fitness.mean()) if fitness.numel() else float("nan"),
            elapsed_seconds=time.perf_counter() - self._start,
        )
        self.history.append(record)
        if self.on_generation is not None:
            self.on_generation(record)
        return record

    @property
    def should_stop(self) -> bool:
        return self.best_fitness >= self.target_fitness

    def result(self) -> SearchResult:
        if self.best_graph is None:
            raise RuntimeError("search finished without evaluating any graph")
        return SearchResult(
            best_graph=self.best_graph,
            best_fitness=self.best_fitness,
            evaluations=self.evaluations,
            history=self.history,
        )
