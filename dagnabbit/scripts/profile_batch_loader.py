"""Does moving graph generation into background processes actually pay?

Graph generation is serial Python and has measured at 70-90% of a stage-one
step, so hiding it behind the optimizer step is the largest remaining structural
win available. But it is not free. A batch of descriptions has to be pickled
across a process boundary, and the *unpickling* lands on the training critical
path in exactly the place the generation it replaced used to sit. So the whole
strategy reduces to one inequality:

    time to unpickle a batch  <  time to generate a batch

Both sides are machine-specific, and the answer has already flipped once during
development: before the pickling of ``rank_batches`` was fixed, unpickling a
batch of 256 cost 418 ms against 129 ms to generate one, making background
workers a 0.7x *regression*. After the fix it is 85 ms against 129 ms. Whether
the remaining margin survives real contention on a real machine is what this
script measures.

What it reports
---------------
For each worker count, and for whichever stages you ask for:

``step_ms``        median wall-clock of a whole training step.
``batch_ms``       median time the step spent *waiting for graphs*. This is the
                   number the strategy is trying to shrink; with workers it is
                   unpickle time, without them it is generation time.
``compute_ms``     median time in forward/backward. Should be flat across worker
                   counts -- if it climbs, the workers are stealing cores from
                   the trainer and you have too many.
``graphs_per_s``   end-to-end throughput. The bottom line.
``speedup``        ``graphs_per_s`` against the ``workers=0`` control.

How to read it
--------------
* ``batch_ms`` collapsing while ``compute_ms`` stays flat is the win, and
  ``speedup`` will show it.
* ``batch_ms`` collapsing while ``compute_ms`` *rises* means CPU contention is
  eating the gain. Try fewer workers.
* ``batch_ms`` barely moving means unpickling costs about what generating did,
  and per-graph IPC is the wall. The fix then is to collate batches inside the
  worker so a handful of large tensors cross the boundary instead of thousands
  of small ones -- a real change, worth making only once these numbers ask for
  it.
* ``speedup`` capping near ``step_ms / compute_ms`` is expected and means you
  have won: generation is fully hidden and compute is the floor.

Both stages are measured because both pay the same cost. Stage one is the
autoencoder's own training step; stage two is behaviour evaluation, encoding,
and a flow-matching step against a frozen checkpoint.

Usage::

    # stage one only, no checkpoint needed
    python -m dagnabbit.scripts.profile_batch_loader --stages encoder

    # both, against a frozen checkpoint
    python -m dagnabbit.scripts.profile_batch_loader \\
        --checkpoint runs/<run> --stages encoder flow --workers 0 2 4 8

Add ``--json out.json`` to write the raw numbers somewhere they can be pasted
back for interpretation.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.checkpoint import load_model, pick_device
from dagnabbit.dag.loader import GraphBatchLoader, GraphGeometry, default_num_workers
from dagnabbit.scripts import config as cfg
from dagnabbit.tasks.logic_gates.evaluate import adder_task
from dagnabbit.tasks.logic_gates.proposer import (
    TruthTableFlowProposer,
    behaviour_images,
)


def synchronize(device: torch.device) -> None:
    """Make wall-clock timing mean something on an asynchronous device."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_encoder_model(device: torch.device) -> DagnabbitAutoEncoder:
    """A stage-one model at the config's geometry, freshly initialized.

    No checkpoint needed: this measures the cost of a *step*, and an untrained
    model's step costs exactly what a trained one's does.
    """
    return DagnabbitAutoEncoder(
        node_embedding_dim=cfg.NODE_EMBEDDING_DIM,
        trunk_node_type_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_trunk_nodes=cfg.NUM_TRUNK_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        mlp_expansion_factor=cfg.MLP_EXPANSION_FACTOR,
        encoder_num_layers=cfg.ENCODER_NUM_LAYERS,
        compressor_num_layers=cfg.COMPRESSOR_NUM_LAYERS,
        decoder_num_layers=cfg.DECODER_NUM_LAYERS,
    ).to(device)


def encoder_step(model, optimizer, graphs, device) -> None:
    """One stage-one training step: the autoencoder's own reconstruction loss."""
    losses = model.training_forward_batch(graphs)
    total = (
        cfg.W_TYPE_CLASSIFICATION * losses.node_classification_losses.mean()
        + cfg.W_PARENT_POINTER
        * losses.parent_pointer_losses.sum()
        / losses.parent_pointer_slot_mask.sum().clamp(min=1)
    )
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()


def flow_step(proposer, model, optimizer, graphs, task, device, gray) -> None:
    """One stage-two step: evaluate, encode, then a flow-matching update.

    Deliberately includes the evaluation and the encode. Those are part of what a
    stage-two step pays per batch, and unlike generation they are *not* moved
    into the worker (they need the task and the frozen model), so a fair
    measurement has to keep them on the critical path.
    """
    images = behaviour_images(graphs, task, gray=gray).to(device)
    clean_latent = model.encode_to_latent(graphs).float()
    losses = proposer(images, clean_latent, condition_dropout=0.1)
    optimizer.zero_grad(set_to_none=True)
    losses.loss.backward()
    optimizer.step()


def measure(
    stage_step,
    geometry: GraphGeometry,
    batch_size: int,
    num_workers: int,
    prefetch: int,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
    seed: int,
) -> dict:
    """Time ``measure_steps`` steps after ``warmup_steps``, split by phase.

    Warmup matters more than usual here: spawning a worker re-imports torch and
    costs seconds, and the queue starts empty, so the first few steps of a
    worker run are unrepresentatively slow. Medians rather than means for the
    same reason -- one stall should not move the number.
    """
    batch_times: list[float] = []
    compute_times: list[float] = []
    step_times: list[float] = []

    with GraphBatchLoader(
        geometry=geometry,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_batches=prefetch,
        seed=seed,
    ) as loader:
        for index in range(warmup_steps + measure_steps):
            synchronize(device)
            started = time.perf_counter()
            graphs = loader.next_batch()
            got_batch = time.perf_counter()
            stage_step(graphs)
            synchronize(device)
            finished = time.perf_counter()

            if index >= warmup_steps:
                batch_times.append(got_batch - started)
                compute_times.append(finished - got_batch)
                step_times.append(finished - started)

    step_ms = statistics.median(step_times) * 1000
    return {
        "workers": num_workers,
        "step_ms": round(step_ms, 2),
        "batch_ms": round(statistics.median(batch_times) * 1000, 2),
        "compute_ms": round(statistics.median(compute_times) * 1000, 2),
        "graphs_per_s": round(batch_size / (step_ms / 1000), 1),
    }


def report(stage: str, rows: list[dict]) -> None:
    baseline = next((row for row in rows if row["workers"] == 0), None)
    print(f"\n=== {stage} ===")
    header = (
        f"{'workers':>7} {'step_ms':>9} {'batch_ms':>9} "
        f"{'compute_ms':>11} {'graphs/s':>9} {'speedup':>8}"
    )
    print(header)
    for row in rows:
        speedup = (
            row["graphs_per_s"] / baseline["graphs_per_s"] if baseline else float("nan")
        )
        row["speedup"] = round(speedup, 3)
        print(
            f"{row['workers']:>7} {row['step_ms']:>9.2f} {row['batch_ms']:>9.2f} "
            f"{row['compute_ms']:>11.2f} {row['graphs_per_s']:>9.1f} "
            f"{speedup:>7.2f}x"
        )

    if baseline is None:
        return
    best = max(rows, key=lambda row: row["graphs_per_s"])

    if best["workers"] == 0:
        # Two very different failures look the same in the throughput column, and
        # they have opposite fixes, so separate them explicitly.
        with_workers = [row for row in rows if row["workers"] > 0]
        cheapest_wait = min(with_workers, key=lambda row: row["batch_ms"])
        wait_fell = cheapest_wait["batch_ms"] < baseline["batch_ms"] * 0.9
        compute_rose = cheapest_wait["compute_ms"] > baseline["compute_ms"] * 1.05
        print("  read: workers never beat inline generation.")
        if wait_fell and compute_rose:
            print(
                f"    Waiting for graphs did fall "
                f"({baseline['batch_ms']:.0f} -> {cheapest_wait['batch_ms']:.0f} ms), "
                f"but compute rose {cheapest_wait['compute_ms'] / baseline['compute_ms']:.2f}x "
                "-- the workers and the trainer are fighting over the same cores. "
                "This is the expected outcome when compute is on the CPU; on a "
                "machine where the step runs on an accelerator there is nothing "
                "for them to contend with. Try fewer workers, and trust the "
                "accelerator measurement over this one."
            )
        elif not wait_fell:
            print(
                f"    Waiting for graphs barely moved "
                f"({baseline['batch_ms']:.0f} -> {cheapest_wait['batch_ms']:.0f} ms), "
                "so unpickling a batch costs about what generating one did and "
                "per-graph IPC is the wall. Collating batches inside the worker "
                "is the next move."
            )
        else:
            print(
                "    No single cause stands out; the margin is simply thin on "
                "this machine."
            )
        return

    contention = best["compute_ms"] / baseline["compute_ms"]
    print(
        f"  read: best is {best['workers']} workers at {best['speedup']:.2f}x. "
        f"Waiting for graphs fell {baseline['batch_ms']:.0f} -> "
        f"{best['batch_ms']:.0f} ms."
    )
    if contention > 1.15:
        print(
            f"  warning: compute_ms rose {contention:.2f}x against the control, "
            "so the workers are competing with the trainer for cores. A lower "
            "worker count may do better."
        )
    remaining = best["batch_ms"] / best["step_ms"]
    if remaining > 0.15:
        print(
            f"  note: waiting for graphs is still {remaining:.0%} of a step, so "
            "there is headroom left; worker-side collation would attack it."
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["encoder"],
        choices=["encoder", "flow"],
        help="'flow' needs --checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="frozen autoencoder for the flow stage; also supplies its geometry",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=None,
        help="worker counts to sweep; defaults to 0 1 2 4 and this machine's "
        "suggested count",
    )
    parser.add_argument("--batch-size", type=int, default=cfg.GRAPH_BATCH_SIZE)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--measure-steps", type=int, default=25)
    parser.add_argument("--flow-batch-size", type=int, default=64)
    parser.add_argument("--gray", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="write raw results here")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    worker_counts = args.workers
    if worker_counts is None:
        suggested = default_num_workers()
        worker_counts = sorted({0, 1, 2, 4, suggested})

    print(f"device={device}  cpu_count={__import__('os').cpu_count()}")
    print(f"worker counts: {worker_counts}")
    print(f"warmup={args.warmup_steps} measure={args.measure_steps} (medians reported)")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    results: dict[str, list[dict]] = {}

    if "encoder" in args.stages:
        model = build_encoder_model(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        geometry = GraphGeometry.from_config(cfg)
        rows = [
            measure(
                stage_step=lambda graphs: encoder_step(
                    model, optimizer, graphs, device
                ),
                geometry=geometry,
                batch_size=args.batch_size,
                num_workers=count,
                prefetch=args.prefetch,
                device=device,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                seed=args.seed,
            )
            for count in worker_counts
        ]
        results["encoder"] = rows
        report(f"stage 1: autoencoder pretraining (batch {args.batch_size})", rows)

    if "flow" in args.stages:
        if not args.checkpoint:
            raise SystemExit("--stages flow requires --checkpoint")
        model, _ = load_model(args.checkpoint, device)
        model.requires_grad_(False)
        model.eval()
        task = adder_task(device)
        proposer = TruthTableFlowProposer.for_task(task=task, model=model).to(device)
        # The sampler is not exercised here, but its statistics must be present
        # for the module to be usable; a small fit is enough for a timing run.
        geometry = GraphGeometry.from_model(model)
        proposer.normalizer.fit(
            model.encode_to_latent(geometry.sample_batch(256)).float()
        )
        optimizer = torch.optim.AdamW(proposer.parameters(), lr=3e-4)
        rows = [
            measure(
                stage_step=lambda graphs: flow_step(
                    proposer, model, optimizer, graphs, task, device, args.gray
                ),
                geometry=geometry,
                batch_size=args.flow_batch_size,
                num_workers=count,
                prefetch=args.prefetch,
                device=device,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                seed=args.seed,
            )
            for count in worker_counts
        ]
        results["flow"] = rows
        report(
            f"stage 2: flow proposer adaptation (batch {args.flow_batch_size})", rows
        )

    if args.json:
        payload = {
            "device": str(device),
            "cpu_count": __import__("os").cpu_count(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "batch_size": args.batch_size,
            "flow_batch_size": args.flow_batch_size,
            "measure_steps": args.measure_steps,
            "results": results,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
