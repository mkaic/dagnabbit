"""Phase 0: train the simulator to predict a random DAG's truth table.

Every step samples fresh graphs, runs them through the exact bitpacked
evaluator to get labels, and scores the model on a random subset of truth-table
patches. No example is ever seen twice, so the training loss *is* the
generalization loss and there is no held-out set to speak of -- the eval below
exists to slice the same distribution by depth, not to measure overfitting.

The gate
--------
Average bit accuracy is not the number to watch. A model can score well above
chance on statistical regularities of random NAND/NOR circuits without
simulating anything, and such a model has a useless loss landscape near an
actual target circuit. What matters is ``eval/accuracy_by_rank``: bit accuracy
bucketed by each output node's longest-path depth. If it falls off a cliff at
the simulator's layer count, the receptive field is binding and the fix is more
layers (or a weight-tied recurrent simulator). If it degrades smoothly past the
layer count, the model found something cheaper than hop-by-hop propagation.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dagnabbit.dag import canonical
from dagnabbit.dag.model import (
    GraphSimulator,
    format_parameter_count,
    parameter_count,
    patch_targets,
    sample_patch_indices,
)
from dagnabbit.scripts import config as cfg
from dagnabbit.tasks.logic_gates.evaluate import (
    adder_task,
    bit_accuracy,
    evaluate_graphs,
    exhaustive_root_values,
)


def make_batch(batch_size: int, root_values: torch.Tensor, device: torch.device):
    """Fresh graphs plus their exact packed truth tables."""
    graphs = canonical.sample(batch_size, cfg.GEOMETRY, device)
    return graphs, evaluate_graphs(graphs, root_values)


def loss_and_accuracy(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """BCE loss, and per-(graph, output) bit accuracy averaged over patches."""
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    correct = ((logits > 0).float() == targets).float()
    # [B, K, C, rows] -> [B, C]: mean over patches and rows.
    return loss, correct.mean(dim=3).mean(dim=1)


@torch.no_grad()
def evaluate(
    model: GraphSimulator,
    root_values: torch.Tensor,
    rows_per_patch: int,
    device: torch.device,
):
    """Bit accuracy on fresh graphs, bucketed by each output node's depth."""
    model.eval()
    graphs, packed = make_batch(cfg.EVAL_BATCH_SIZE, root_values, device)
    patch_indices = sample_patch_indices(
        cfg.MODEL.num_patches, cfg.EVAL_PATCHES, device
    )
    with torch.autocast(
        device_type=device.type, dtype=cfg.AMP_DTYPE, enabled=cfg.AMP_ENABLED
    ):
        logits = model.forward_graphs(graphs, patch_indices)
    targets = patch_targets(packed, patch_indices, rows_per_patch)
    _, accuracy = loss_and_accuracy(logits.float(), targets)

    # [B, C] accuracy against [B, C] output ranks, flattened and bucketed.
    ranks = graphs.output_ranks.reshape(-1)
    flat = accuracy.reshape(-1)
    num_buckets = int(ranks.max().item()) + 1
    totals = torch.zeros(num_buckets, device=device, dtype=torch.float64)
    counts = torch.zeros(num_buckets, device=device, dtype=torch.float64)
    totals.scatter_add_(0, ranks, flat.double())
    counts.scatter_add_(0, ranks, torch.ones_like(flat, dtype=torch.float64))

    by_rank = {
        rank: float(totals[rank] / counts[rank])
        for rank in range(num_buckets)
        if counts[rank] > 0
    }
    model.train()
    return float(accuracy.mean()), by_rank, {r: int(counts[r]) for r in by_rank}


@torch.no_grad()
def probe_adder(
    model: GraphSimulator, rows_per_patch: int, device: torch.device
) -> float:
    """How well does the simulator predict the *reference adder's* behaviour?

    Not a training signal -- an out-of-distribution check. The hand-built adder
    is a structured circuit, which is exactly what random sampling never
    produces and exactly what Phase 1 will be searching for.
    """
    from dagnabbit.tasks.logic_gates.reference_circuits import nand_ripple_carry_adder

    graphs = nand_ripple_carry_adder(cfg.GEOMETRY).to(device)
    task = adder_task(device)
    patch_indices = torch.arange(cfg.MODEL.num_patches, device=device)
    with torch.autocast(
        device_type=device.type, dtype=cfg.AMP_DTYPE, enabled=cfg.AMP_ENABLED
    ):
        logits = model.forward_graphs(graphs, patch_indices)

    # The circuit really computes the adder, so the task's targets are its truth
    # table; this checks the *prediction* against it. Sanity-check the premise
    # too, since a mis-wired reference would silently make this meaningless.
    packed = evaluate_graphs(graphs, task.root_values)
    exact, _ = bit_accuracy(packed, task)
    assert float(exact[0]) == 1.0, "reference adder does not compute the adder"

    targets = patch_targets(packed, patch_indices, rows_per_patch)
    _, accuracy = loss_and_accuracy(logits.float(), targets)
    return float(accuracy.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=None, help="run directory name")
    parser.add_argument("--steps", type=int, default=cfg.NUM_STEPS)
    parser.add_argument("--device", default=cfg.DEVICE)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(cfg.SEED)
    device = torch.device(args.device)
    run_name = args.name or time.strftime("%Y%m%d-%H%M%S")
    run_directory = Path(cfg.LOG_DIR) / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_directory))

    model = GraphSimulator(cfg.GEOMETRY, cfg.MODEL).to(device)
    print(f"simulator: {format_parameter_count(parameter_count(model))} parameters")
    # Read off the geometry-derived constants before compile wraps the module.
    rows_per_patch = model.decoder.rows_per_patch
    if cfg.TORCH_COMPILE and not args.no_compile and device.type == "cuda":
        model = torch.compile(model, mode=cfg.TORCH_COMPILE_MODE)

    optimizer = cfg.OPTIMIZER_CLASS(model, **cfg.OPTIMIZER_KWARGS)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda update: min(1.0, (update + 1) / cfg.LR_WARMUP_OPTIMIZER_STEPS),
    )

    root_values = exhaustive_root_values(cfg.GEOMETRY.num_root_nodes).to(device)
    started = time.time()

    for step in range(args.steps):
        graphs, packed = make_batch(cfg.GRAPH_BATCH_SIZE, root_values, device)
        patch_indices = sample_patch_indices(
            cfg.MODEL.num_patches, cfg.PATCHES_PER_STEP, device
        )
        targets = patch_targets(packed, patch_indices, rows_per_patch)

        with torch.autocast(
            device_type=device.type, dtype=cfg.AMP_DTYPE, enabled=cfg.AMP_ENABLED
        ):
            logits = model.forward_graphs(graphs, patch_indices)
        loss, accuracy = loss_and_accuracy(logits.float(), targets)

        (loss / cfg.GRADIENT_ACCUMULATION_STEPS).backward()
        loss = loss.detach()
        if (step + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
            if cfg.GRADIENT_CLIP_MAX_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.GRADIENT_CLIP_MAX_NORM
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if step % cfg.LOG_EVERY == 0:
            graphs_per_second = (
                cfg.GRAPH_BATCH_SIZE * (step + 1) / (time.time() - started)
            )
            writer.add_scalar("train/loss", float(loss), step)
            writer.add_scalar("train/bit_accuracy", float(accuracy.mean()), step)
            writer.add_scalar("train/graphs_per_second", graphs_per_second, step)
            print(
                f"step {step:>7}  loss {float(loss):.4f}  "
                f"acc {float(accuracy.mean()):.4f}  {graphs_per_second:.0f} graphs/s"
            )

        if step % cfg.EVAL_EVERY == 0:
            mean_accuracy, by_rank, counts = evaluate(
                model, root_values, rows_per_patch, device
            )
            writer.add_scalar("eval/bit_accuracy", mean_accuracy, step)
            for rank, value in by_rank.items():
                writer.add_scalar(f"eval/accuracy_by_rank/{rank:02d}", value, step)
            writer.add_scalar(
                "probe/adder_bit_accuracy",
                probe_adder(model, rows_per_patch, device),
                step,
            )
            ladder = "  ".join(
                f"r{rank}={value:.3f}(n={counts[rank]})"
                for rank, value in sorted(by_rank.items())
            )
            print(f"  eval acc {mean_accuracy:.4f}   {ladder}")

        if cfg.CHECKPOINT_EVERY and step and step % cfg.CHECKPOINT_EVERY == 0:
            torch.save(
                {
                    "step": step,
                    "model": getattr(model, "_orig_mod", model).state_dict(),
                    "geometry": cfg.GEOMETRY,
                    "model_config": cfg.MODEL,
                },
                run_directory / f"step_{step:08d}.pt",
            )

    writer.close()


if __name__ == "__main__":
    main()
