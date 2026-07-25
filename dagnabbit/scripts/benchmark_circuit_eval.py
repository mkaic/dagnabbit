"""Throughput of bitpacked circuit evaluation across batch sizes and devices.

Answers the practical question for a search loop: how many candidate circuits
per second can be scored, and at what batch size does the device stop being
launch-overhead-bound. Also times graph construction, which is pure Python and
is usually the real ceiling once evaluation is batched.

Usage::

    python -m dagnabbit.scripts.benchmark_circuit_eval
    python -m dagnabbit.scripts.benchmark_circuit_eval --batch-sizes 64 256 1024
"""

import argparse
import time

import torch

from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.scripts import config as cfg
from dagnabbit.tasks.logic_gates.evaluate import (
    adder_task,
    bit_accuracy,
    evaluate_graphs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[8, 32, 128, 512],
        help="Batch sizes to time.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Devices to time (default: cpu, plus cuda or mps when available).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed repeats per configuration; the fastest is reported.",
    )
    return parser.parse_args()


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    elif torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main() -> None:
    args = parse_args()
    devices = args.devices or available_devices()

    def make_graphs(count: int):
        return [
            make_random_graph_description(
                num_root_nodes=cfg.NUM_ROOT_NODES,
                num_trunk_nodes=cfg.NUM_TRUNK_NODES,
                num_output_nodes=cfg.NUM_OUTPUT_NODES,
                trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
                num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
            )
            for _ in range(count)
        ]

    start = time.perf_counter()
    construction_sample = make_graphs(64)
    construction_seconds = (time.perf_counter() - start) / 64
    print(
        f"graph construction: {construction_seconds * 1000:.2f} ms/graph "
        f"({1 / construction_seconds:.0f} graphs/s, single-threaded Python)"
    )
    print(f"ranks per graph:    {len(construction_sample[0].rank_batches)}")

    largest = max(args.batch_sizes)
    graphs = make_graphs(largest)

    for device in devices:
        task = adder_task(device)
        buffer_bytes = cfg.NUM_ROOT_NODES + cfg.NUM_TRUNK_NODES + cfg.NUM_OUTPUT_NODES
        print(f"\n--- {device} ---")
        print(f"{'batch':>7} {'ms':>9} {'ms/graph':>9} {'graphs/s':>10} {'buffer':>9}")
        for batch_size in args.batch_sizes:
            batch = graphs[:batch_size]
            # Warm up allocator, kernel caches, and any lazy device init.
            bit_accuracy(evaluate_graphs(batch[: min(4, batch_size)], task), task)
            synchronize(device)

            best = float("inf")
            for _ in range(args.repeats):
                start = time.perf_counter()
                bit_accuracy(evaluate_graphs(batch, task), task)
                synchronize(device)
                best = min(best, time.perf_counter() - start)

            gigabytes = batch_size * buffer_bytes * task.num_words / 1e9
            print(
                f"{batch_size:>7} {best * 1000:>9.1f} {best / batch_size * 1000:>9.3f} "
                f"{batch_size / best:>10.0f} {gigabytes:>8.2f}G"
            )


if __name__ == "__main__":
    main()
