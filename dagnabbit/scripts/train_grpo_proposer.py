"""GRPO-train a truth-table proposer against a frozen autoencoder.

The policy is: read a truth table, write a latent, sample a graph out of the
frozen decoder. The reward is how well that graph reproduces the truth table it
was shown. Targets come from hindsight relabelling -- a random graph's own
behaviour is the goal -- which is what gives a group of samples enough reward
spread for group-relative advantages to mean anything.

This replaces supervised (behaviour -> this specific graph) training, which
measurement showed cannot generalize: behaviour distance and structure distance
correlate at +0.003, so the supervised target is an arbitrary injection. Under
GRPO any graph achieving the target behaviour earns full reward, which is what
program-synthesis work calls the fix for *program aliasing*.

Two evaluations run alongside training, and the gap between them is the result:

* **in-distribution** -- fresh relabelled targets, the same kind trained on.
* **reference** -- the ripple-carry adder and bitwise XOR, structured, never
  trained on, and the thing actually worth caring about. The frozen decoder is
  known to reconstruct both exactly from their latents, so these targets *are*
  reachable; whether the proposer can find them is the open question.

Both report best-of-group, not mean: this is a search, and one good circuit out
of a group is a success.

Reward and credit assignment
----------------------------
The reward is a per-output-plane Matthews correlation, not bit accuracy.
Accuracy has a degenerate optimum -- emit the target's per-plane majority bit
-- which the runs of 2026-07-28 found and sat on, reaching reward_mean 0.67
with ``constant_outputs`` at 0.98 and the reference evaluation pinned at
exactly 0.5 for 1910 steps. See :mod:`dagnabbit.tasks.logic_gates.rewards`.

Scoring the planes separately also gives the update somewhere to put them: the
latent has one token per output node, so each plane's advantage is paid to the
token at that output's position rather than averaged into a single scalar over
the whole log-probability. ``--credit-assignment aggregate`` restores the old
behaviour, and ``--reward-statistic accuracy`` the old signal, so either change
can be isolated.
"""

import argparse
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from dagnabbit.dag.checkpoint import load_model, pick_device
from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    make_random_graph_description,
)
from dagnabbit.dag.policy import sample_from_decoder, sample_from_latent_noise
from dagnabbit.scripts import config as cfg
from dagnabbit.search.grpo import GRPOConfig, grpo_step
from dagnabbit.tasks.logic_gates.evaluate import adder_task, evaluate_choices
from dagnabbit.tasks.logic_gates.proposer import TruthTableProposer
from dagnabbit.tasks.logic_gates.rewards import (
    BEHAVIOUR_STATISTICS,
    behaviour_accuracy_per_output,
    behaviour_correlation,
    behaviour_match_reward,
    constant_output_fraction,
    packed_behaviours,
)
from dagnabbit.tasks.logic_gates.roundtrip_probe import reference_circuits
from dagnabbit.tasks.logic_gates.truth_table_image import (
    image_dimensions,
    outputs_to_image,
)


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


def behaviours_to_images(behaviours: torch.Tensor, task, gray: bool) -> torch.Tensor:
    """Packed ``[P, C, W]`` behaviours -> the ``[P, C, H, W]`` proposer input."""
    height, width = image_dimensions(task.root_values.shape[0])
    return outputs_to_image(behaviours, height, width, gray=gray).float()


@torch.no_grad()
def evaluate_targets(
    proposer,
    targets: torch.Tensor,
    task,
    sampler,
    group_size: int,
    gray: bool,
    device: torch.device,
    trunk_node_in_degrees: list[int],
) -> dict[str, float]:
    """Sample a group per target and report how close the best one gets.

    Uses the same stochastic policy as training rather than an argmax decode:
    what matters is what the search can find, and a group is the unit of search.

    Reports correlation *and* accuracy whichever one training optimizes.
    Accuracy is what every run before 2026-07-28 logged, so keeping it here
    holds new curves comparable against those; correlation is what the reward
    now is. Reading them together is also the hack detector: accuracy climbing
    while correlation sits at zero is the constant-output policy, which is
    exactly what those earlier runs turned out to be doing.
    """
    was_training = proposer.training
    proposer.eval()
    try:
        images = behaviours_to_images(targets, task, gray).to(device)
        prompt_indices = torch.arange(
            targets.shape[0], device=device
        ).repeat_interleave(group_size)

        # Encode once per target, then repeat the latent -- see grpo_step.
        latents = proposer(images).repeat_interleave(group_size, dim=0)
        sampled = sampler(latents)
        predicted = evaluate_choices(
            sampled.trunk_types,
            sampled.parent_choices,
            task,
            trunk_node_in_degrees,
        )
        goals = targets.to(predicted.device)[prompt_indices.to(predicted.device)]
        groups = (targets.shape[0], group_size)
        accuracy = behaviour_accuracy_per_output(predicted, goals, task).mean(
            dim=-1
        ).view(groups)
        correlation = behaviour_correlation(predicted, goals, task).mean(
            dim=-1
        ).view(groups)
    finally:
        if was_training:
            proposer.train()

    return {
        "reward_mean": float(correlation.mean()),
        "reward_best": float(correlation.max(dim=1).values.mean()),
        "reward_best_overall": float(correlation.max()),
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_best": float(accuracy.max(dim=1).values.mean()),
        "accuracy_best_overall": float(accuracy.max()),
        "constant_outputs": constant_output_fraction(predicted, task),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="frozen autoencoder .ckpt or run directory")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--prompts", type=int, default=8, help="targets per step")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--latent-sigma",
        type=float,
        default=0.1,
        help="exploration noise on the latent; measured useful band 0.03-0.3",
    )
    parser.add_argument(
        "--decoder-temperature",
        type=float,
        default=None,
        help="explore by sampling the decoder instead of the latent "
        "(weaker; see dagnabbit.dag.policy)",
    )
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument(
        "--reward-statistic",
        choices=sorted(BEHAVIOUR_STATISTICS),
        default="correlation",
        help="per-output-plane training signal. 'correlation' (Matthews) is "
        "zero for a constant predictor; 'accuracy' is what runs before "
        "2026-07-28 used, and is reward-hackable by emitting the target's "
        "majority bit -- see dagnabbit.tasks.logic_gates.rewards",
    )
    parser.add_argument(
        "--credit-assignment",
        choices=("factored", "aggregate"),
        default="factored",
        help="'factored' pays each output plane's advantage to the latent "
        "token at that output's position; 'aggregate' averages the planes "
        "into one scalar over the whole log-probability (the pre-2026-07-28 "
        "behaviour). See GRPOConfig.factored_credit for the bias this trades",
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
        help="ViT width; independent of the checkpoint's latent dim",
    )
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--mlp-expansion-factor", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gray", action="store_true", help="Gray-code the axes")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-targets", type=int, default=4)
    parser.add_argument("--eval-group-size", type=int, default=64)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--log-dir",
        default=cfg.ADAPTATION_LOG_DIR,
        help="stage-two runs live apart from the stage-one checkpoints "
        f"in {cfg.TENSORBOARD_LOG_DIR}/",
    )
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
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        mlp_expansion_factor=args.mlp_expansion_factor,
    ).to(device)
    print(
        f"proposer: {sum(p.numel() for p in proposer.parameters()) / 1e6:.1f}M params, "
        f"{args.num_layers}L x {args.embedding_dim}d over "
        f"{proposer.num_patches} patches -> latent "
        f"[{model.num_output_nodes}, {model.node_embedding_dim}], "
        f"{args.prompts} prompts x {args.group_size} samples per step"
    )

    grpo_config = GRPOConfig(
        group_size=args.group_size,
        entropy_weight=args.entropy_weight,
        normalize_advantages=not args.no_normalize_advantages,
        factored_credit=args.credit_assignment == "factored",
    )
    # Latent noise unless explicitly overridden: sampling the decoder stops
    # exploring once the proposer emits meaningful latents.
    if args.decoder_temperature is None:
        sampler = partial(sample_from_latent_noise, model, sigma=args.latent_sigma)
        exploration = f"latent noise sigma={args.latent_sigma}"
    else:
        # Decoder sampling factorizes over trunk nodes and input slots, not
        # over latent tokens, so there is no per-output log-probability to pay
        # a per-output advantage into.
        if grpo_config.factored_credit:
            raise SystemExit(
                "--credit-assignment factored needs the latent-noise policy; "
                "decoder sampling has no latent-token factorization. Pass "
                "--credit-assignment aggregate to use --decoder-temperature."
            )
        sampler = partial(
            sample_from_decoder, model, temperature=args.decoder_temperature
        )
        exploration = f"decoder sampling T={args.decoder_temperature}"
    print(f"exploration: {exploration}")
    print(
        f"reward: per-output {args.reward_statistic}, "
        f"{args.credit_assignment} credit assignment"
    )
    optimizer = torch.optim.AdamW(proposer.parameters(), lr=args.learning_rate)
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S-grpo")
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"run_dir={run_dir}")

    # Held-out structured behaviours. Built by explicit constructors rather
    # than the random sampler, so they stay outside the training distribution.
    circuits = reference_circuits(device)
    reference_targets = packed_behaviours(
        [circuit.graph for circuit in circuits], task
    ).to(device)

    for step in range(1, args.steps + 1):
        started = time.perf_counter()
        # Hindsight relabelling: sample graphs, and make what they compute the
        # goal. The graphs themselves are then thrown away -- only behaviour is
        # the specification, and any circuit reproducing it earns full reward.
        targets = packed_behaviours(sample_graphs(args.prompts), task).to(device)
        images = behaviours_to_images(targets, task, args.gray).to(device)
        prepared = time.perf_counter()

        loss, stats = grpo_step(
            proposer=proposer,
            specifications=images,
            sampler=sampler,
            reward_fn=partial(
                behaviour_match_reward,
                targets=targets,
                task=task,
                trunk_node_in_degrees=model.trunk_node_in_degrees,
                statistic=args.reward_statistic,
            ),
            config=grpo_config,
        )

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
            for name, value in stats.scalars().items():
                writer.add_scalar(name, value, step)
            writer.add_scalar("time/prepare_targets", prepared - started, step)
            writer.add_scalar("time/grpo_step", finished - prepared, step)
            print(
                f"step {step:>6} reward {stats.reward_mean:.4f} "
                f"best {stats.reward_best:.4f} "
                f"gstd {stats.reward_group_std:.4f} "
                f"dead {stats.dead_plane_fraction:.2f} "
                f"| {finished - started:.2f}s"
            )

        if step % args.eval_every == 0:
            in_distribution = evaluate_targets(
                proposer,
                packed_behaviours(sample_graphs(args.eval_targets), task).to(device),
                task,
                sampler,
                args.eval_group_size,
                args.gray,
                device,
                model.trunk_node_in_degrees,
            )
            reference = evaluate_targets(
                proposer,
                reference_targets,
                task,
                sampler,
                args.eval_group_size,
                args.gray,
                device,
                model.trunk_node_in_degrees,
            )
            for split, results in (
                ("in_distribution", in_distribution),
                ("reference", reference),
            ):
                for name, value in results.items():
                    writer.add_scalar(f"{split}/{name}", value, step)
            print(
                f"  eval @ {step}: in-dist corr {in_distribution['reward_best']:.4f} "
                f"acc {in_distribution['accuracy_best']:.4f} "
                f"| reference corr {reference['reward_best']:.4f} "
                f"acc {reference['accuracy_best']:.4f} "
                f"(top {reference['reward_best_overall']:.4f}, "
                f"const {reference['constant_outputs']:.2f})"
            )

    writer.close()


if __name__ == "__main__":
    main()
