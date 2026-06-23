from __future__ import annotations

import multiprocessing as mp
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass

import torch

from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_random_graph_description,
)


@dataclass(frozen=True)
class GraphGenerationConfig:
    num_root_nodes: int
    num_trunk_nodes: int
    num_output_nodes: int
    trunk_node_in_degrees: int | list[int]
    num_trunk_node_types: int


def _draw_graph_seeds(batch_size: int) -> list[int]:
    return [
        int(torch.randint(0, 2**63 - 1, (1,), dtype=torch.int64).item())
        for _ in range(batch_size)
    ]


def _make_graph_batch_from_seeds(
    seeds: list[int],
    graph_config: GraphGenerationConfig,
) -> list[FixedInDegreeDAGDescription]:
    return [
        make_random_graph_description(
            num_root_nodes=graph_config.num_root_nodes,
            num_trunk_nodes=graph_config.num_trunk_nodes,
            num_output_nodes=graph_config.num_output_nodes,
            trunk_node_in_degrees=graph_config.trunk_node_in_degrees,
            num_trunk_node_types=graph_config.num_trunk_node_types,
            seed=seed,
        )
        for seed in seeds
    ]


class GraphBatchSource:
    """Ordered graph-batch producer with optional process prefetching."""

    def __init__(
        self,
        *,
        batch_size: int,
        graph_config: GraphGenerationConfig,
        prefetch: bool,
        num_workers: int,
        prefetch_batches: int,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if prefetch_batches < 0:
            raise ValueError("prefetch_batches must be non-negative")

        self.batch_size = batch_size
        self.graph_config = graph_config
        self.prefetch = prefetch and num_workers > 0 and prefetch_batches > 0
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self._executor: ProcessPoolExecutor | None = None
        self._futures: deque[Future[list[FixedInDegreeDAGDescription]]] = deque()
        self._closed = False

        if self.prefetch:
            self._executor = ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp.get_context("spawn"),
            )
            for _ in range(prefetch_batches):
                self._submit_next()

    def _make_sync(self) -> list[FixedInDegreeDAGDescription]:
        seeds = _draw_graph_seeds(self.batch_size)
        return _make_graph_batch_from_seeds(seeds, self.graph_config)

    def _submit_next(self) -> None:
        if self._executor is None:
            return
        seeds = _draw_graph_seeds(self.batch_size)
        self._futures.append(
            self._executor.submit(
                _make_graph_batch_from_seeds,
                seeds,
                self.graph_config,
            )
        )

    def next(self) -> list[FixedInDegreeDAGDescription]:
        if self._closed:
            raise RuntimeError("GraphBatchSource is closed")
        if self._executor is None:
            return self._make_sync()

        if not self._futures:
            self._submit_next()
        future = self._futures.popleft()
        batch = future.result()
        if not self._closed:
            self._submit_next()
        return batch

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._futures.clear()

    def __enter__(self) -> GraphBatchSource:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
