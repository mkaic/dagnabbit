"""Train a conditional flow-matching proposer against a frozen autoencoder.

The experiment: **can a proposal distribution over graphs, drawn from
repeatedly, hit a behaviour a single guess cannot?**

Behaviour -> graph is massively one-to-many (dead nodes and unselected producers
alone put whole families of structurally different circuits at bit-identical
behaviour), so any regression onto a single latent targets the per-slot marginal
over consistent parents, and the argmax of independent marginals need not be a
coherent circuit. Sampling a modelled distribution avoids that by construction,
and turns the proposer into a search distribution: draw N, score them with the
evaluator, keep the best.

What to watch, in order
-----------------------
1. ``in_distribution/best_of_n_correlation`` must **rise with**
   ``--eval-candidates``. That is the whole thesis; a flat curve across N means
   the distribution has collapsed and you may as well have regressed.
2. ``in_distribution/distinct_fraction`` is the diagnostic for when it does
   not. Collapsing toward ``1/candidates`` means every draw is the same circuit;
   suspect ``--guidance`` before anything else.
3. ``reference/*`` is the real prize and the likely disappointment, because it
   is a *distribution shift* problem rather than a modelling one -- random
   circuits' behaviours look nothing like an adder's. If 1 and 2 look good and
   this does not, the fix is in the graph sampler, not here.

Every step generates its own graphs
-----------------------------------
There is no dataset and there is deliberately no cache. Random graphs are
sampled fresh each step, evaluated to see what they compute, and that behaviour
becomes the conditioning input with the graph's own latent as the target --
hindsight relabelling, exact labels, **unlimited supply**.

That unlimited supply is the whole point and is worth protecting. A
pregenerated corpus would reintroduce everything this task gets to skip: a
finite sample to overfit, epochs, a train/validation split, a decision about
corpus size, and gigabytes on disk. Instead every batch is drawn from the true
distribution and no example is ever seen twice, so the training loss *is* the
generalization loss and there is nothing to hold out.

It is not free -- graph generation and evaluation is the dominant cost of a step
here, not the backward pass. That is a known and accepted trade: paying it buys
uncorrelated data forever. If it ever needs to be hidden rather than paid, the
answer is to overlap generation with the optimizer step, not to store a dataset.

The training loss is nearly uninformative on its own: it is dominated by
irreducible noise variance and is not comparable across runs. ``loss_noisy_half``
is the half worth watching.

Usage::

    python -m dagnabbit.scripts.train_flow_proposer runs/<run>
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from dagnabbit.dag.checkpoint import load_model, pick_device
from dagnabbit.dag.geometry import GraphGeometry
from dagnabbit.scripts import config as cfg
from dagnabbit.tasks.logic_gates.evaluate import adder_task
from dagnabbit.tasks.logic_gates.proposer import (
    TruthTableFlowProposer,
    behaviour_images,
    evaluate_flow_proposals,
)
from dagnabbit.tasks.logic_gates.rewards import packed_behaviours
from dagnabbit.tasks.logic_gates.roundtrip_probe import reference_circuits


def fit_normalizer(
    proposer: TruthTableFlowProposer,
    model,
    geometry: GraphGeometry,
    num_samples: int,
    batch_size: int,
) -> None:
    """Fit the latent statistics from a one-off sample of encoded graphs.

    The only place anything is generated in bulk, and it happens once before
    training rather than being kept. The statistics are a property of the frozen
    autoencoder; the graphs used to measure them are thrown away. Synchronous on
    purpose -- this runs once, so there is nothing to overlap it with.
    """
    collected = []
    remaining = num_samples
    while remaining > 0:
        chunk = min(batch_size, remaining)
        graphs = geometry.sample_batch(chunk)
        collected.append(model.encode_to_latent(graphs).float())
        remaining -= chunk
    proposer.normalizer.fit(torch.cat(collected))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="frozen autoencoder .ckpt or run directory")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
        help="shared width of the behaviour encoder and the velocity model",
    )
    parser.add_argument("--encoder-layers", type=int, default=8)
    parser.add_argument(
        "--velocity-layers",
        type=int,
        default=4,
        help="deliberately shallow; the latent is only a handful of tokens, so "
        "depth costs serial latency once per sampler step",
    )
    parser.add_argument("--mlp-expansion-factor", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--condition-dropout",
        type=float,
        default=0.1,
        help="fraction of examples trained with the null specification; this is "
        "what makes guidance (and unconditional sampling) possible",
    )
    parser.add_argument(
        "--normalizer-samples",
        type=int,
        default=65_536,
        help="latents used to fit the per-dimension statistics",
    )
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="1.0 disables guidance and halves sampling cost. Above 1 trades "
        "coverage for single-draw fidelity, which is the wrong trade when the "
        "point is best-of-N",
    )
    parser.add_argument("--gray", action="store_true", help="Gray-code the axes")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-graphs", type=int, default=16)
    parser.add_argument(
        "--eval-candidates",
        type=int,
        nargs="+",
        default=[1, 16],
        help="candidate counts to report best-of-N at; the headline result is "
        "whether the score rises across these",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default=cfg.ADAPTATION_LOG_DIR)
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
    proposer = TruthTableFlowProposer.for_task(
        task=task,
        model=model,
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        encoder_num_layers=args.encoder_layers,
        velocity_num_layers=args.velocity_layers,
        mlp_expansion_factor=args.mlp_expansion_factor,
    ).to(device)

    encoder_parameters = sum(p.numel() for p in proposer.encoder.parameters())
    velocity_parameters = sum(p.numel() for p in proposer.velocity_model.parameters())
    print(
        f"proposer: encoder {encoder_parameters / 1e6:.1f}M over "
        f"{proposer.num_patches} patches (runs once), velocity "
        f"{velocity_parameters / 1e6:.1f}M over {model.num_output_nodes} tokens "
        f"(runs {args.sample_steps}x), latent "
        f"[{model.num_output_nodes}, {model.node_embedding_dim}]"
    )

    # From the *checkpoint*, not from config.py: the config tracks whichever run
    # is current and drifts away from older checkpoints, and a mismatch here
    # produces graphs the frozen decoder cannot describe.
    geometry = GraphGeometry.from_model(model)
    fit_normalizer(proposer, model, geometry, args.normalizer_samples, args.batch_size)
    print(
        "fitted latent statistics: mean |.| "
        f"{proposer.normalizer.mean.abs().mean():.4f}, std "
        f"{proposer.normalizer.std.mean():.4f}"
    )

    optimizer = torch.optim.AdamW(
        proposer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S-flow")
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"run_dir={run_dir}")

    # Held-out structured circuits, built by explicit constructors so they stay
    # out of the training distribution by construction.
    circuits = reference_circuits(device)
    reference_graphs = [circuit.graph for circuit in circuits]
    reference_images = behaviour_images(reference_graphs, task, gray=args.gray)
    reference_targets = torch.stack(
        [circuit.task.target_values for circuit in circuits]
    )

    def evaluate(step: int) -> None:
        """Evaluate the live weights. There is one set of weights and this is it."""
        held_out = geometry.sample_batch(args.eval_graphs)
        held_out_images = behaviour_images(held_out, task, gray=args.gray)
        held_out_targets = packed_behaviours(held_out, task)

        for num_candidates in args.eval_candidates:
            splits = {
                "in_distribution": evaluate_flow_proposals(
                    model=model,
                    proposer=proposer,
                    images=held_out_images,
                    targets=held_out_targets,
                    task=task,
                    device=device,
                    num_candidates=num_candidates,
                    num_steps=args.sample_steps,
                    guidance_strength=args.guidance,
                    source_graphs=held_out,
                ),
                "reference": evaluate_flow_proposals(
                    model=model,
                    proposer=proposer,
                    images=reference_images,
                    targets=reference_targets,
                    task=task,
                    device=device,
                    num_candidates=num_candidates,
                    num_steps=args.sample_steps,
                    guidance_strength=args.guidance,
                ),
            }
            for split, results in splits.items():
                for name, value in results.items():
                    writer.add_scalar(f"{split}/{name}_n{num_candidates}", value, step)
            print(
                f"  eval @ {step} n={num_candidates}: "
                f"in-dist corr {splits['in_distribution']['best_of_n_correlation']:.4f} "
                f"distinct {splits['in_distribution']['distinct_fraction']:.2f} "
                f"| ref corr {splits['reference']['best_of_n_correlation']:.4f} "
                f"acc {splits['reference']['best_of_n_accuracy']:.4f}"
            )

    def save(step: int, name: str) -> None:
        torch.save(
            {
                "step": step,
                "proposer_state_dict": proposer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "source_checkpoint_step": checkpoint.get("step"),
            },
            run_dir / name,
        )

    try:
        train(
            args=args,
            model=model,
            proposer=proposer,
            geometry=geometry,
            task=task,
            device=device,
            optimizer=optimizer,
            writer=writer,
            evaluate=evaluate,
            save=save,
        )
    finally:
        writer.close()


def train(
    *,
    args,
    model,
    proposer,
    geometry: GraphGeometry,
    task,
    device,
    optimizer,
    writer,
    evaluate,
    save,
) -> None:
    """The training loop proper, split out so ``main`` can guarantee cleanup."""
    for step in range(1, args.steps + 1):
        started = time.perf_counter()
        # Every step sees graphs that have never been seen before and will never
        # be seen again. No dataset, no epochs, no overfitting a finite sample --
        # see the module docstring on why this is worth paying for.
        graphs = geometry.sample_batch(args.batch_size)
        images = behaviour_images(graphs, task, gray=args.gray).to(device)
        clean_latent = model.encode_to_latent(graphs).float()
        generated = time.perf_counter()

        losses = proposer(
            images,
            clean_latent,
            condition_dropout=args.condition_dropout,
        )

        optimizer.zero_grad(set_to_none=True)
        losses.loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(proposer.parameters(), args.grad_clip)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * min(
                1.0, step / max(1, args.warmup_steps)
            )
        optimizer.step()
        finished = time.perf_counter()

        if step % args.log_every == 0:
            for name, value in losses.scalars("train").items():
                writer.add_scalar(name, value, step)
            writer.add_scalar("time/generate_batch", generated - started, step)
            writer.add_scalar("time/optimize", finished - generated, step)
            print(
                f"step {step:>7} loss {losses.loss.item():.4f} "
                f"(noisy {losses.noisy_half_loss.item():.4f}) "
                f"| {finished - started:.3f}s "
                f"(gen {generated - started:.3f})"
            )

        if step % args.eval_every == 0:
            evaluate(step)

        if args.checkpoint_every and step % args.checkpoint_every == 0:
            save(step, "latest.ckpt")

    save(args.steps, "latest.ckpt")


if __name__ == "__main__":
    main()
