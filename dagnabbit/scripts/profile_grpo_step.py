"""Phase-timing profile of one GRPO proposer step.

Mirrors ``train_grpo_proposer`` but splits the step into its component phases
with a device sync around each, so time is attributed to the phase that
actually spent it rather than to whichever later op happened to block.

Usage::

    uv run python -m dagnabbit.scripts.profile_grpo_step <checkpoint> \
        [--device auto] [--prompts 64] [--group-size 64]

Measured 2026-07-28 on an M4 Max at prompts=64 group=64 (4096 graphs/step),
before/after the tensorized reward path (choice tensors straight into
``evaluate_choices``, no description objects):

* CPU: 10.6 s/step -> 7.9 s
* MPS: 10.2 s/step -> 5.2 s (remaining cost is MPS launch overhead in the
  per-position evaluation sweep and the frozen decode; CUDA launches are an
  order of magnitude cheaper, so expect a larger win there)
"""

import argparse
import contextlib
import time

import torch
from torch.distributions import Normal

from dagnabbit.dag.checkpoint import load_model, pick_device
from dagnabbit.dag.policy import project_to_shell
from dagnabbit.scripts.train_grpo_proposer import behaviours_to_images, sample_graphs
from dagnabbit.search.grpo import GRPOConfig, group_advantages
from dagnabbit.tasks.logic_gates.evaluate import adder_task, evaluate_choices
from dagnabbit.tasks.logic_gates.proposer import TruthTableProposer
from dagnabbit.tasks.logic_gates.rewards import behaviour_accuracy, packed_behaviours


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


class PhaseTimer:
    """Wall-clock per phase, synced on entry and exit so GPU work is owned."""

    def __init__(self, device: torch.device):
        self.device = device
        self.times: dict[str, list[float]] = {}

    @contextlib.contextmanager
    def phase(self, name: str):
        synchronize(self.device)
        started = time.perf_counter()
        yield
        synchronize(self.device)
        self.times.setdefault(name, []).append(time.perf_counter() - started)

    def report(self, batch_size: int) -> None:
        medians = {
            name: sorted(values)[len(values) // 2] * 1000
            for name, values in sorted(self.times.items())
        }
        total = sum(medians.values())
        print("\n=== phase medians (ms) ===")
        for name, median in medians.items():
            print(f"{name:24s} {median:9.1f} ms   {100 * median / total:5.1f}%")
        print(f"{'TOTAL':24s} {total:9.1f} ms")
        print(f"throughput ~ {batch_size / (total / 1000):.0f} graphs/s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--latent-sigma", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    print(f"device: {device}")

    model, _ = load_model(args.checkpoint, device)
    model.requires_grad_(False)
    model.eval()

    task = adder_task(device)
    proposer = TruthTableProposer.for_task(
        task=task,
        model=model,
        patch_size=16,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        mlp_expansion_factor=4.0,
    ).to(device)
    optimizer = torch.optim.AdamW(proposer.parameters(), lr=1e-4)
    config = GRPOConfig(group_size=args.group_size)

    timer = PhaseTimer(device)
    null = contextlib.nullcontext()
    for step in range(args.steps + 1):
        # Step 0 is warmup (allocator growth, kernel autotuning) and untimed.
        phase = timer.phase if step > 0 else lambda name: null

        with phase("1_sample_prompt_graphs"):
            prompt_graphs = sample_graphs(args.prompts)
        with phase("2_targets_eval"):
            targets = packed_behaviours(prompt_graphs, task).to(device)
        with phase("3_images"):
            images = behaviours_to_images(targets, task, gray=False).to(device)
        with phase("4_proposer_forward"):
            latents = proposer(images).repeat_interleave(args.group_size, dim=0)
        with phase("5_noise_sample"):
            distribution = Normal(latents, args.latent_sigma)
            noisy = distribution.sample()
            log_probs = distribution.log_prob(noisy).sum(dim=(1, 2))
        with phase("6_decode_choices"):
            trunk_types, parent_choices = model.generate_choices(
                project_to_shell(noisy)
            )
        with phase("7_evaluate_choices"):
            predicted = evaluate_choices(
                trunk_types, parent_choices, task, model.trunk_node_in_degrees
            )
        with phase("8_reward_score"):
            prompt_indices = torch.arange(
                args.prompts, device=device
            ).repeat_interleave(args.group_size)
            goals = targets[prompt_indices]
            rewards = behaviour_accuracy(predicted, goals, task).to(
                device=log_probs.device, dtype=log_probs.dtype
            )
        with phase("9_loss_backward_opt"):
            grouped = rewards.view(args.prompts, args.group_size)
            advantages = group_advantages(grouped, config).flatten().detach()
            loss = -(advantages * log_probs).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proposer.parameters(), 1.0)
            optimizer.step()

    timer.report(args.prompts * args.group_size)


if __name__ == "__main__":
    main()
