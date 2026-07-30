"""Generating graph batches in background processes.

Both training stages consume the same thing -- a list of freshly sampled
:class:`~dagnabbit.dag.description.FixedInDegreeDAGDescription` -- and in both
of them producing that list is the dominant cost of a step, not the backward
pass. Measured at 70-90% of a stage-one step. The generator is pure Python and
GIL-bound, so the only way to hide it is another *process*.

Nothing here is stage-specific. Stage one feeds the batch to
``training_forward_batch``; stage two evaluates it and encodes it. Both just
want a list.

Why the data can never be cached instead
----------------------------------------
Because unlimited fresh uncorrelated data is the premise of the whole setup: no
example is ever seen twice, so the training loss *is* the generalization loss.
Storing a corpus would trade that away for exactly the latency this module
hides for free. See the module docstring of
:mod:`dagnabbit.scripts.train_flow_proposer`.

The open question this is built to answer
-----------------------------------------
Whether it actually pays. A fully-constructed description is not a small object:
``rank_batches`` and the canonical tensors are built eagerly in the constructor,
so one description carries on the order of ninety small CPU tensors, and a batch
of 256 carries tens of thousands. Every one of those has to cross a process
boundary, and unpickling happens on the critical path -- exactly where we were
trying to remove work.

So the win is real only if unpickling a batch is cheaper than generating it.
That is an empirical question about a specific machine, which is why
``num_workers=0`` routes through this same class as an honest control, and why
:mod:`dagnabbit.scripts.profile_batch_loader` exists to measure both. If the
measurement says IPC is eating the gain, the next move is to collate batches
*inside* the worker so that a handful of large tensors cross the boundary
instead of thousands of small ones -- a bigger change, and not worth making
before the numbers ask for it.

Pitfalls this module handles, all of which are silent when wrong
---------------------------------------------------------------
* **Every worker must be seeded differently.** Otherwise they generate identical
  graph streams and the "uncorrelated data" property -- the entire reason for
  in-loop generation -- quietly dies while throughput looks great.
* **Workers must not oversubscribe threads.** Each sets
  ``torch.set_num_threads(1)``; without it N workers each spin up a full intra-op
  pool and contend, which can make the whole thing slower than synchronous.
* **The start method is ``spawn``.** Forking a process that has initialized CUDA
  is undefined behaviour, and the parent here will have. Spawn costs a few
  seconds of startup per worker (it re-imports torch) and buys not having to
  reason about that.
* **The queue is bounded.** An unbounded queue lets workers sprint ahead of the
  trainer and grow without limit.
* **Shutdown drains before joining.** A process that has put items on a queue
  will not exit until they are consumed; joining first is a classic deadlock.
* **A dead worker must not hang the trainer.** ``next_batch`` polls with a
  timeout and re-checks liveness rather than blocking forever on ``get``.
* **Tensor sharing strategy is set to ``file_system``.** With many small tensors
  in flight the default file-descriptor strategy exhausts the fd limit. This is
  the same fix ``DataLoader`` users reach for.
"""

import multiprocessing as mp
import os
import queue as queue_module
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_random_graph_description,
)

if TYPE_CHECKING:
    from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder

# How long ``next_batch`` waits on the queue before re-checking that the workers
# are still alive. Purely a liveness poll: a healthy warm queue returns
# immediately and never reaches this.
QUEUE_POLL_SECONDS = 1.0
# How long to wait for a worker to exit on shutdown before terminating it.
SHUTDOWN_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class GraphGeometry:
    """The shape of the graphs to sample. Plain data, and picklable on purpose.

    This is the *only* thing sent to a worker at startup. It deliberately cannot
    carry a model, a task, a device or a tensor: under ``spawn`` everything in
    the worker's arguments is pickled, and a CUDA-backed object crossing that
    boundary either fails loudly or -- worse -- initializes CUDA in a process
    that has no business touching it.
    """

    num_root_nodes: int
    num_trunk_nodes: int
    num_output_nodes: int
    num_trunk_node_types: int
    # Tuple rather than list so the dataclass stays frozen and hashable.
    trunk_node_in_degrees: tuple[int, ...]

    @classmethod
    def from_config(cls, cfg) -> "GraphGeometry":
        """Read the geometry stage-one trains at out of ``config.py``."""
        in_degrees = cfg.TRUNK_NODE_TYPE_IN_DEGREES
        if isinstance(in_degrees, int):
            in_degrees = [in_degrees] * cfg.NUM_TRUNK_NODE_TYPES
        return cls(
            num_root_nodes=cfg.NUM_ROOT_NODES,
            num_trunk_nodes=cfg.NUM_TRUNK_NODES,
            num_output_nodes=cfg.NUM_OUTPUT_NODES,
            num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
            trunk_node_in_degrees=tuple(in_degrees),
        )

    @classmethod
    def from_model(cls, model: "DagnabbitAutoEncoder") -> "GraphGeometry":
        """Read the geometry a *frozen checkpoint* was trained at.

        Stage two must use this rather than ``from_config``: the config tracks
        whichever run is current and drifts away from older checkpoints, and a
        geometry mismatch here produces graphs the loaded decoder cannot
        describe.
        """
        return cls(
            num_root_nodes=model.num_root_nodes,
            num_trunk_nodes=model.num_trunk_nodes,
            num_output_nodes=model.num_output_nodes,
            num_trunk_node_types=model.num_trunk_node_types,
            trunk_node_in_degrees=tuple(model.trunk_node_in_degrees),
        )

    def sample_batch(self, count: int) -> list[FixedInDegreeDAGDescription]:
        """``count`` freshly sampled graphs. The unit of work everywhere."""
        return [
            make_random_graph_description(
                num_root_nodes=self.num_root_nodes,
                num_trunk_nodes=self.num_trunk_nodes,
                num_output_nodes=self.num_output_nodes,
                trunk_node_in_degrees=list(self.trunk_node_in_degrees),
                num_trunk_node_types=self.num_trunk_node_types,
            )
            for _ in range(count)
        ]


def _worker_loop(
    geometry: GraphGeometry,
    batch_size: int,
    batch_queue,
    stop_event,
    seed: int,
) -> None:
    """Generate batches until told to stop. Module-level so ``spawn`` can find it.

    A closure or a bound method would not survive pickling into a spawned
    process, which is the whole reason this is not a method on the loader.
    """
    # One thread per worker. Without this each worker builds its own intra-op
    # pool and they fight over the same cores; graph construction is serial
    # Python anyway, so the pool buys nothing and costs contention.
    torch.set_num_threads(1)
    # Distinct streams per worker. Identical seeds here would silently produce
    # correlated batches, which is the one thing this whole design exists to
    # avoid.
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except (RuntimeError, AttributeError):
        # Not fatal: the strategy is an optimization against fd exhaustion, and
        # a platform that refuses it simply keeps the default.
        pass

    try:
        while not stop_event.is_set():
            batch = geometry.sample_batch(batch_size)
            # Bounded put with a timeout rather than a blocking one: a blocking
            # put on a full queue would ignore stop_event and hang shutdown.
            while not stop_event.is_set():
                try:
                    batch_queue.put(batch, timeout=QUEUE_POLL_SECONDS)
                    break
                except queue_module.Full:
                    continue
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        # Normal shutdown paths when the parent goes away first.
        pass
    finally:
        # Do not block process exit on flushing the feeder thread; the parent
        # drains what it wants and discards the rest.
        batch_queue.cancel_join_thread()


class GraphBatchLoader:
    """A source of graph batches, optionally generated in background processes.

    ``num_workers=0`` generates inline, on the calling thread, and is the
    control condition rather than a separate code path -- both training scripts
    call ``next_batch`` either way, so switching is one flag and the comparison
    is honest.

    Use as a context manager, or call :meth:`close`. Leaked workers keep
    generating graphs nobody reads.
    """

    def __init__(
        self,
        geometry: GraphGeometry,
        batch_size: int,
        num_workers: int = 0,
        prefetch_batches: int = 2,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative; got {num_workers}")
        if prefetch_batches <= 0:
            raise ValueError(
                f"prefetch_batches must be positive; got {prefetch_batches}"
            )

        self.geometry = geometry
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._processes: list = []
        self._queue = None
        self._stop_event = None
        self._closed = False

        if num_workers == 0:
            return

        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except (RuntimeError, AttributeError):
            pass

        # Spawn, explicitly, and not the process-wide default: the parent will
        # have initialized CUDA by the time this is built, and forking after
        # that is undefined behaviour.
        context = mp.get_context("spawn")
        self._queue = context.Queue(maxsize=num_workers * prefetch_batches)
        self._stop_event = context.Event()

        for index in range(num_workers):
            process = context.Process(
                target=_worker_loop,
                args=(
                    geometry,
                    batch_size,
                    self._queue,
                    self._stop_event,
                    # Distinct, and reproducible from the run seed. The stride is
                    # large so nearby run seeds do not overlap worker streams.
                    seed * 100_003 + index,
                ),
                daemon=True,
            )
            process.start()
            self._processes.append(process)

    def next_batch(self) -> list[FixedInDegreeDAGDescription]:
        """One batch of graphs. Blocks only if the workers have not kept up."""
        if self.num_workers == 0:
            return self.geometry.sample_batch(self.batch_size)
        if self._closed:
            raise RuntimeError("GraphBatchLoader is closed")

        while True:
            try:
                return self._queue.get(timeout=QUEUE_POLL_SECONDS)
            except queue_module.Empty:
                # Distinguish "slow" from "dead". Blocking forever on get() is
                # how a crashed worker turns into a hung training run.
                if not any(process.is_alive() for process in self._processes):
                    raise RuntimeError(
                        "every graph-generation worker exited before producing a "
                        "batch; re-run with --loader-workers 0 to see the "
                        "underlying error on the main process"
                    ) from None

    def close(self) -> None:
        """Stop the workers and reap them. Idempotent."""
        if self._closed or self.num_workers == 0:
            self._closed = True
            return
        self._closed = True

        self._stop_event.set()

        # Drain before joining. A worker that has queued a batch nobody read
        # will not exit, so joining first deadlocks.
        while True:
            try:
                self._queue.get_nowait()
            except (queue_module.Empty, OSError, ValueError):
                break

        for process in self._processes:
            process.join(timeout=SHUTDOWN_GRACE_SECONDS)
            if process.is_alive():
                process.terminate()
                process.join(timeout=SHUTDOWN_GRACE_SECONDS)

        try:
            self._queue.close()
        except (OSError, ValueError):
            pass

    def __enter__(self) -> "GraphBatchLoader":
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    def __del__(self) -> None:
        # Belt and braces for a loader that escaped its context manager.
        try:
            self.close()
        except Exception:
            pass

    def describe(self) -> str:
        """One line for a training script to print at startup."""
        if self.num_workers == 0:
            return (
                f"graph loader: synchronous, batch {self.batch_size} "
                "(generation is on the critical path)"
            )
        return (
            f"graph loader: {self.num_workers} spawned workers, batch "
            f"{self.batch_size}, queue {self._queue._maxsize}"
        )


def default_num_workers() -> int:
    """A sane worker count for this machine, leaving the trainer a core.

    Deliberately conservative. Graph generation is CPU-bound serial Python, so
    more workers help right up until they contend with the training process
    itself for cores, after which they hurt.
    """
    available = os.cpu_count() or 2
    return max(1, min(8, available - 1))
