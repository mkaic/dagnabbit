"""Phase 1: hill-climb candidate circuits through a trained simulator.

A population of independent candidates (per-node categorical logits, see
:mod:`dagnabbit.dag.candidates`) is optimized by gradient descent on the
simulator's predicted truth table against the adder target. Every step, every
candidate's hard graph is also run through the *exact* evaluator, which serves
three purposes at once:

1. the true fitness curve -- the only number that counts;
2. the exploitation detector -- the gap between what the simulator predicts a
   candidate computes and what it actually computes;
3. hindsight labels -- (graph, exact truth table) pairs are correct training
   data no matter how unfit the graph, so the simulator fine-tunes on the
   population's own shifting distribution while the population climbs it.

Random graphs mixed into the fine-tune batch double as a running random-search
baseline at the same evaluator budget: if best-of-random keeps pace with the
population, the gradient is buying nothing and the run has failed its kill
criterion.

Runs log to ``ADAPT_LOG_DIR`` (default ``adaptations/``), not ``runs/`` --
different loop, different metrics.

Usage::

    uv run python -m dagnabbit.scripts.adapt_circuit \\
        --checkpoint runs/<run>/step_XXXXXXXX.pt
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dagnabbit.dag.candidates import CandidatePopulation
from dagnabbit.dag.graphs import GraphBatch
from dagnabbit.dag.graphs import sample as sample_graphs
from dagnabbit.dag.model import patch_targets, sample_patch_indices
from dagnabbit.scripts import config as cfg
from dagnabbit.scripts import provenance
from dagnabbit.scripts.checkpoints import load_simulator
from dagnabbit.tasks.logic_gates.evaluate import (
    adder_task,
    bit_accuracy,
    evaluate_graphs,
)


def concatenate(a: GraphBatch, b: GraphBatch) -> GraphBatch:
    if a.geometry != b.geometry:
        raise ValueError("cannot concatenate batches over different geometries")
    return GraphBatch(
        node_types=torch.cat([a.node_types, b.node_types]),
        parent_indices=torch.cat([a.parent_indices, b.parent_indices]),
        parent_slot_mask=torch.cat([a.parent_slot_mask, b.parent_slot_mask]),
        ranks=torch.cat([a.ranks, b.ranks]),
        geometry=a.geometry,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--name",
        default=None,
        help="label appended to the run directory's timestamp prefix",
    )
    parser.add_argument("--steps", type=int, default=cfg.ADAPT_STEPS)
    parser.add_argument("--device", default=cfg.DEVICE)
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="disable the surrogate fine-tune: pure frozen-simulator "
        "hill-climbing, the exploitation ablation",
    )
    args = parser.parse_args()

    torch.manual_seed(cfg.SEED)
    device = torch.device(args.device)

    run_name = time.strftime("%Y%m%d-%H%M%S")
    if args.name:
        run_name = f"{run_name}-{args.name}"
    run_directory = Path(cfg.ADAPT_LOG_DIR) / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_directory))
    record = provenance.capture(run_directory, cfg, sys.argv, writer)
    print(provenance.format_summary(record))

    model, loaded_step = load_simulator(
        args.checkpoint, device, cfg.GATES.num_types
    )
    model.train()
    geometry = model.geometry
    rows_per_patch = model.decoder.rows_per_patch
    print(f"simulator from {args.checkpoint} (step {loaded_step})")

    population = CandidatePopulation(cfg.ADAPT_POPULATION, geometry).to(device)
    seeds = sample_graphs(
        cfg.ADAPT_POPULATION, geometry, device, sampling=cfg.SAMPLING
    )
    population.initialize_from(seeds, cfg.ADAPT_INIT_CONCENTRATION)
    candidate_optimizer = torch.optim.Adam(
        population.parameters(), lr=cfg.ADAPT_CANDIDATE_LR
    )

    finetune_optimizer = None
    if not args.frozen:
        finetune_optimizer = cfg.OPTIMIZER_CLASS(
            model,
            **{
                **cfg.OPTIMIZER_KWARGS,
                "muon_lr": cfg.ADAPT_FINETUNE_LR,
                "adam_lr": cfg.ADAPT_FINETUNE_LR,
            },
        )

    task = adder_task(device)
    target_packed = task.target_values.unsqueeze(0)  # [1, C, num_words]

    best_fitness = 0.0
    best_random_fitness = 0.0
    candidate_evaluations = 0
    random_evaluations = 0

    progress = tqdm(range(args.steps), unit="step", dynamic_ncols=True)
    postfix: dict[str, str] = {}
    for step in progress:
        # --- draw hard graphs + straight-through tokens, score them exactly ---
        # Freeze before sampling: the token construction reads the model's
        # embedding tables, and autograd decides at graph-build time whether
        # their gradients will be materialized.
        model.requires_grad_(False)
        sample = population.sample(model.node_tokens, cfg.ADAPT_TEMPERATURE)
        packed = evaluate_graphs(
            sample.graphs, task.root_values, cfg.GATES.operators
        )
        fitness, per_output = bit_accuracy(packed, task)
        candidate_evaluations += len(fitness)

        # --- candidate update through the frozen simulator ---
        patch_indices = sample_patch_indices(
            cfg.MODEL.num_patches, cfg.ADAPT_PATCHES_PER_STEP, device
        )
        target_bits = patch_targets(target_packed, patch_indices, rows_per_patch)
        target_bits = target_bits.expand(cfg.ADAPT_POPULATION, -1, -1, -1)
        with torch.autocast(
            device_type=device.type, dtype=cfg.AMP_DTYPE, enabled=cfg.AMP_ENABLED
        ):
            logits = model.decoder(model.simulator(sample.tokens), patch_indices)
        surrogate_loss = F.binary_cross_entropy_with_logits(
            logits.float() / cfg.ADAPT_LOSS_TEMPERATURE, target_bits
        )
        candidate_optimizer.zero_grad(set_to_none=True)
        surrogate_loss.backward()
        candidate_optimizer.step()
        surrogate_loss = surrogate_loss.detach()

        # The exploitation diagnostics, all on the same patch subset:
        # what the simulator says the candidates compute, versus the target
        # and versus what they actually compute.
        with torch.no_grad():
            mean_abs_logit = logits.detach().abs().mean()
            predictions = logits.detach() > 0
            predicted_vs_target = (
                (predictions == target_bits.bool()).float().mean()
            )
            actual_bits = patch_targets(packed, patch_indices, rows_per_patch)
            actual_vs_target = (actual_bits == target_bits).float().mean()
            faithfulness = (predictions == actual_bits.bool()).float().mean()

        # --- hindsight refresh + random-search baseline ---
        random_graphs = sample_graphs(
            cfg.ADAPT_RANDOM_GRAPHS, geometry, device, sampling=cfg.SAMPLING
        )
        random_packed = evaluate_graphs(
            random_graphs, task.root_values, cfg.GATES.operators
        )
        random_fitness, _ = bit_accuracy(random_packed, task)
        random_evaluations += len(random_fitness)
        best_random_fitness = max(
            best_random_fitness, float(random_fitness.max())
        )

        finetune_loss = None
        if finetune_optimizer is not None:
            model.requires_grad_(True)
            refresh = concatenate(sample.graphs, random_graphs)
            refresh_packed = torch.cat([packed, random_packed])
            refresh_patches = sample_patch_indices(
                cfg.MODEL.num_patches, cfg.ADAPT_PATCHES_PER_STEP, device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=cfg.AMP_DTYPE,
                enabled=cfg.AMP_ENABLED,
            ):
                refresh_logits = model.forward_graphs(refresh, refresh_patches)
            finetune_loss = F.binary_cross_entropy_with_logits(
                refresh_logits.float(),
                patch_targets(refresh_packed, refresh_patches, rows_per_patch),
            )
            finetune_optimizer.zero_grad(set_to_none=True)
            finetune_loss.backward()
            finetune_optimizer.step()
            finetune_loss = finetune_loss.detach()

        # --- track the champion ---
        step_best = float(fitness.max())
        if step_best > best_fitness:
            best_fitness = step_best
            champion = int(fitness.argmax())
            torch.save(
                {
                    "step": step,
                    "fitness": best_fitness,
                    "per_output": per_output[champion].tolist(),
                    "node_types": sample.graphs.node_types[champion].cpu(),
                    "parent_indices": sample.graphs.parent_indices[champion].cpu(),
                    "parent_slot_mask": sample.graphs.parent_slot_mask[
                        champion
                    ].cpu(),
                    "geometry": geometry,
                },
                run_directory / "best.pt",
            )
            progress.write(
                f"step {step:>8}  new best fitness {best_fitness:.6f}  "
                f"(per bit: "
                + " ".join(f"{value:.3f}" for value in per_output[champion])
                + ")"
            )

        if step % cfg.ADAPT_LOG_EVERY == 0:
            live = sample.graphs.num_live_trunk_nodes.float()
            writer.add_scalar("adapt/surrogate_loss", float(surrogate_loss), step)
            writer.add_scalar("adapt/fitness_mean", float(fitness.mean()), step)
            writer.add_scalar("adapt/fitness_max", step_best, step)
            writer.add_scalar("adapt/fitness_best", best_fitness, step)
            writer.add_scalar(
                "adapt/predicted_vs_target", float(predicted_vs_target), step
            )
            writer.add_scalar(
                "adapt/actual_vs_target", float(actual_vs_target), step
            )
            writer.add_scalar(
                "adapt/exploitation_gap",
                float(predicted_vs_target - actual_vs_target),
                step,
            )
            writer.add_scalar("adapt/faithfulness", float(faithfulness), step)
            writer.add_scalar("adapt/mean_abs_logit", float(mean_abs_logit), step)
            writer.add_scalar("adapt/live_gates", float(live.mean()), step)
            writer.add_scalar(
                "adapt/max_output_rank",
                float(sample.graphs.output_ranks.max(dim=1).values.float().mean()),
                step,
            )
            for name, value in population.mean_entropy().items():
                writer.add_scalar(f"adapt/entropy_{name}", value, step)
            writer.add_scalar(
                "baseline/random_fitness_best", best_random_fitness, step
            )
            writer.add_scalar(
                "baseline/candidate_evaluations", candidate_evaluations, step
            )
            writer.add_scalar(
                "baseline/random_evaluations", random_evaluations, step
            )
            if finetune_loss is not None:
                writer.add_scalar("finetune/loss", float(finetune_loss), step)

            postfix["best"] = f"{best_fitness:.4f}"
            postfix["mean"] = f"{float(fitness.mean()):.4f}"
            postfix["rand"] = f"{best_random_fitness:.4f}"
            postfix["gap"] = f"{float(predicted_vs_target - actual_vs_target):+.3f}"
            postfix["faith"] = f"{float(faithfulness):.3f}"
            progress.set_postfix(postfix, refresh=False)

        if (
            cfg.ADAPT_CHECKPOINT_EVERY
            and step
            and step % cfg.ADAPT_CHECKPOINT_EVERY == 0
        ):
            torch.save(
                {
                    "step": step,
                    "population": population.state_dict(),
                    "model": model.state_dict(),
                    "geometry": geometry,
                    "model_config": model.config,
                    "best_fitness": best_fitness,
                },
                run_directory / f"step_{step:08d}.pt",
            )

    progress.close()
    writer.close()
    print(
        f"best fitness {best_fitness:.6f} after {candidate_evaluations} candidate "
        f"evaluations; best random {best_random_fitness:.6f} after "
        f"{random_evaluations}"
    )


if __name__ == "__main__":
    main()
