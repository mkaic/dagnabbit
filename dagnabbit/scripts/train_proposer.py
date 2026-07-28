"""Train a truth-table -> latent proposer against a frozen autoencoder.

The experiment this exists to run: **can a network trained only on the
behaviour of random graphs propose a latent for a structured target it has
never seen?**

Training data costs nothing and needs no search. Sample random graphs, evaluate
them to see what each one actually computes, and use that behaviour as the
conditioning input with the graph itself as the label -- hindsight relabelling,
with exact labels and unlimited supply. The autoencoder stays frozen; the loss
is its own type/pointer cross-entropy applied to whatever the proposed latent
decodes to.

Two evaluations run side by side, and the *gap between them* is the result:

* **in-distribution** -- held-out random graphs, the same kind the model trains
  on. Says whether the model learned the mapping at all.
* **reference** -- the ripple-carry adder and bitwise XOR from
  :mod:`dagnabbit.tasks.logic_gates.roundtrip_probe`: structured, never trained
  on, and the thing actually worth caring about.

Both good means the idea works. In-distribution good and reference bad means
the model learned fine and the *training distribution* is the problem, which is
far more tractable (bias the sampler, add a curriculum) than a model that
cannot learn the mapping at all. Both bad means the premise needs rethinking.

Hyperparameters are argparse flags rather than ``config.py`` entries: this
hangs off a frozen checkpoint rather than being the project's main training
path. Promote them if it graduates.
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from dagnabbit.dag.checkpoint import load_model, pick_device
from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_random_graph_description,
)
from dagnabbit.dag.proposal import canonical_targets, reconstruction_losses
from dagnabbit.scripts import config as cfg
from dagnabbit.tasks.logic_gates.evaluate import adder_task
from dagnabbit.tasks.logic_gates.proposer import (
    TruthTableProposer,
    behaviour_images,
    evaluate_proposals,
)
from dagnabbit.tasks.logic_gates.roundtrip_probe import reference_circuits


def sample_graphs(count: int) -> list[FixedInDegreeDAGDescription]:
    """A fresh batch of random graphs, shaped by the training config."""
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="frozen autoencoder .ckpt or run directory")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--mlp-expansion-factor", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gray", action="store_true", help="Gray-code the axes")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-graphs", type=int, default=16)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    model, checkpoint = load_model(args.checkpoint, device)
    model.requires_grad_(False)
    model.eval()
    print(f"frozen autoencoder from step {checkpoint.get('step')} on {device}")

    task = adder_task(device)
    proposer = TruthTableProposer.for_task(
        task=task,
        model=model,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        mlp_expansion_factor=args.mlp_expansion_factor,
    ).to(device)
    num_parameters = sum(p.numel() for p in proposer.parameters())
    print(
        f"proposer: {num_parameters / 1e6:.1f}M params, "
        f"{proposer.num_patches} patches of {args.patch_size}x{args.patch_size}, "
        f"latent [{model.num_output_nodes}, {model.node_embedding_dim}], "
        f"layout={'gray' if args.gray else 'binary'}"
    )

    optimizer = torch.optim.AdamW(proposer.parameters(), lr=args.learning_rate)
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S-proposer")
    run_dir = Path(cfg.TENSORBOARD_LOG_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"run_dir={run_dir}")

    # Held-out structured circuits, scored against their own tasks. Built by
    # explicit constructors rather than the random sampler, so they stay out of
    # the training distribution by construction.
    circuits = reference_circuits(device)
    reference_graphs = [circuit.graph for circuit in circuits]
    reference_tasks = [circuit.task for circuit in circuits]
    reference_images = behaviour_images(reference_graphs, task, gray=args.gray)

    for step in range(1, args.steps + 1):
        started = time.perf_counter()
        graphs = sample_graphs(args.batch_size)
        sampled = time.perf_counter()
        images = behaviour_images(graphs, task, gray=args.gray).to(device)
        evaluated = time.perf_counter()

        metrics = reconstruction_losses(
            model,
            proposer(images),
            canonical_targets(graphs, device),
        )
        loss = metrics.total(cfg.W_TYPE_CLASSIFICATION, cfg.W_PARENT_POINTER)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(proposer.parameters(), args.grad_clip)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * min(
                1.0, step / max(1, args.warmup_steps)
            )
        optimizer.step()
        finished = time.perf_counter()

        if step % args.log_every == 0:
            writer.add_scalar("train/loss", loss.detach(), step)
            for name, value in metrics.scalars("train").items():
                writer.add_scalar(name, value, step)
            writer.add_scalar("time/sample_graphs", sampled - started, step)
            writer.add_scalar("time/evaluate_graphs", evaluated - sampled, step)
            writer.add_scalar("time/optimize", finished - evaluated, step)
            print(
                f"step {step:>6} loss {loss.item():.4f} "
                f"type {metrics.type_accuracy.item():.3f} "
                f"ptr {metrics.pointer_accuracy.item():.3f} "
                f"| {finished - started:.2f}s "
                f"(gen {sampled - started:.2f} eval {evaluated - sampled:.2f})"
            )

        if step % args.eval_every == 0:
            held_out = sample_graphs(args.eval_graphs)
            splits = {
                "in_distribution": evaluate_proposals(
                    model,
                    proposer,
                    held_out,
                    [task] * len(held_out),
                    behaviour_images(held_out, task, gray=args.gray),
                    device,
                ),
                "reference": evaluate_proposals(
                    model,
                    proposer,
                    reference_graphs,
                    reference_tasks,
                    reference_images,
                    device,
                ),
            }
            for split, results in splits.items():
                for name, value in results.items():
                    writer.add_scalar(f"{split}/{name}", value, step)
            print(
                f"  eval @ {step}: "
                f"in-dist ptr {splits['in_distribution']['pointer_accuracy']:.3f} "
                f"fit {splits['in_distribution']['fitness_mean']:.4f} "
                f"| reference ptr {splits['reference']['pointer_accuracy']:.3f} "
                f"fit {splits['reference']['fitness_mean']:.4f} "
                f"(best {splits['reference']['fitness_best']:.4f})"
            )

    writer.close()


if __name__ == "__main__":
    main()
