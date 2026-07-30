"""Measure how the decoded circuit and its fitness respond to latent perturbation.

This is the go/no-go diagnostic for gradient- or RL-based latent search. It
answers two questions, both of which can independently kill the approach:

**1. Is the decode piecewise-constant with wide plateaus?**
    ``generate`` takes an argmax over type logits and pointer logits, so
    ``f(decode(z))`` is a step function of ``z``. If the step treads are wide,
    every zeroth-order gradient estimate is exactly zero almost everywhere and
    only a learned surrogate can produce a usable direction. Measured here as
    the fraction of probes whose decoded token sequence is *identical* to the
    anchor's, swept over perturbation radius.

**2. Is fitness locally smooth, or is it white noise?**
    The headline number is ``local_std / global_std``: the spread of fitness
    among probes at a given radius, divided by the spread of fitness across
    unrelated latents. Near 0 means neighbours resemble each other and the
    latent is a searchable landscape. Near 1 means a small step lands you
    somewhere fitness-wise unrelated -- the latent carries no local gradient
    signal at all, and no amount of surrogate fitting or policy gradient will
    invent one. In that case searching the latent is the wrong plan and the
    generator itself has to be the thing you optimize.

Perturbation respects the latent's geometry: every latent token exits a
LayerNorm, so tokens live on a thin spherical shell of radius ~sqrt(D). Probes
step along the tangent plane and retract back onto each token's own shell, so
radius is a relative angular step and never carries the probe off-manifold in
the one direction we already know is invalid.

Nothing here is circuit-specific except :func:`score_latents`, which is the one
place the objective is named. Everything else only ever consumes a scalar per
graph, so pointing this at a different task is a one-function edit.
"""

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.checkpoint import load_model, pick_device
from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.dag.proposal import choice_signatures
from dagnabbit.tasks.logic_gates.evaluate import (
    BitpackedTask,
    adder_task,
    evaluate_and_score,
)

# Circuits per evaluation chunk. The evaluator holds a [chunk, num_nodes,
# num_words] uint8 buffer -- ~1.2 MB per circuit for the adder table -- so an
# unbounded batch would allocate tens of gigabytes.
DEFAULT_EVAL_CHUNK = 128


@torch.no_grad()
def random_latents(
    model: DagnabbitAutoEncoder,
    count: int,
    batch_size: int = 32,
) -> Tensor:
    """Encode random graphs to get latents from the distribution decode saw."""
    latents: list[Tensor] = []
    remaining = count
    while remaining > 0:
        chunk = min(batch_size, remaining)
        graphs = [
            make_random_graph_description(
                num_root_nodes=model.num_root_nodes,
                num_trunk_nodes=model.num_trunk_nodes,
                num_output_nodes=model.num_output_nodes,
                trunk_node_in_degrees=model.trunk_node_in_degrees,
                num_trunk_node_types=model.num_trunk_node_types,
            )
            for _ in range(chunk)
        ]
        latents.append(model.encode_to_latent(graphs).float())
        remaining -= chunk
    return torch.cat(latents)


def perturb_on_shell(anchor: Tensor, count: int, radius: float) -> Tensor:
    """``count`` probes at relative angular ``radius`` around a ``[K, D]`` anchor.

    Each token is perturbed within its own tangent plane and retracted onto its
    own shell, so token norms are preserved exactly and ``radius`` reads as
    "fraction of the token's norm, moved sideways".
    """
    probes = anchor.unsqueeze(0).expand(count, -1, -1)
    norms = probes.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit = probes / norms

    noise = torch.randn_like(probes)
    # Strip the radial component: a pure-radial step is undone by the retraction
    # below and would silently shrink the effective step size.
    tangent = noise - (noise * unit).sum(dim=-1, keepdim=True) * unit
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    stepped = probes + radius * norms * tangent
    return stepped * (norms / stepped.norm(dim=-1, keepdim=True).clamp(min=1e-8))


@torch.no_grad()
def decode_tokens(model: DagnabbitAutoEncoder, latent: Tensor) -> Tensor:
    """The discrete decode of ``[B, K, D]`` latents as a flat ``[B, T]`` tensor.

    Mirrors the argmaxes :meth:`DagnabbitAutoEncoder.generate` takes, but keeps
    the result as integers so two decodes can be compared by Hamming distance
    without building descriptions.
    """
    trunk_types, parent_choices = model.generate_choices(latent)
    return choice_signatures(model, trunk_types, parent_choices)


@torch.no_grad()
def score_latents(
    model: DagnabbitAutoEncoder,
    latents: Tensor,
    task: BitpackedTask,
    eval_chunk: int,
) -> Tensor:
    """Objective value of every decoded latent, as a ``[B]`` float64 CPU tensor.

    The only place the task enters. Swap this one function to point the
    diagnostic at a different objective.
    """
    scores: list[Tensor] = []
    for start in range(0, latents.shape[0], eval_chunk):
        graphs = model.generate(latents[start : start + eval_chunk])
        scores.append(evaluate_and_score(graphs, task)[0])
    return torch.cat(scores)


@dataclass
class RadiusResult:
    radius: float
    identical_fraction: float  # probes whose decode is bit-identical to anchor
    token_hamming: float  # mean fraction of active tokens that changed
    local_std: float  # fitness spread among probes
    mean_abs_delta: float  # mean |f(probe) - f(anchor)|


def probe_radius(
    model: DagnabbitAutoEncoder,
    anchor: Tensor,
    anchor_tokens: Tensor,
    anchor_fitness: float,
    task: BitpackedTask,
    radius: float,
    num_probes: int,
    eval_chunk: int,
) -> RadiusResult:
    probes = perturb_on_shell(anchor, num_probes, radius)
    tokens = decode_tokens(model, probes)
    fitness = score_latents(model, probes, task, eval_chunk)

    # Compare only slots active in *either* decode: a slot that appeared or
    # vanished because a trunk type flipped is a real structural change.
    both_inactive = (tokens < 0) & (anchor_tokens.unsqueeze(0) < 0)
    differing = (tokens != anchor_tokens.unsqueeze(0)) & ~both_inactive
    comparable = (~both_inactive).sum(dim=1).clamp(min=1)

    return RadiusResult(
        radius=radius,
        identical_fraction=float((differing.sum(dim=1) == 0).float().mean()),
        token_hamming=float((differing.sum(dim=1) / comparable).mean()),
        local_std=float(fitness.std()),
        mean_abs_delta=float((fitness - anchor_fitness).abs().mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="a .ckpt file or a run directory")
    parser.add_argument("--num-anchors", type=int, default=4)
    parser.add_argument("--num-probes", type=int, default=32)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        help="relative angular step sizes, as a fraction of token norm",
    )
    parser.add_argument(
        "--num-global",
        type=int,
        default=128,
        help="unrelated latents used for the global fitness spread baseline",
    )
    parser.add_argument("--eval-chunk", type=int, default=DEFAULT_EVAL_CHUNK)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    print(
        f"loaded step {checkpoint.get('step')} on {device}; "
        f"latent is [{model.num_output_nodes}, {model.node_embedding_dim}] "
        f"= {model.num_output_nodes * model.node_embedding_dim} dims"
    )

    task = adder_task(device)

    # Baseline: how much does fitness vary between latents that have nothing to
    # do with each other? Every local spread below is read against this.
    global_latents = random_latents(model, args.num_global)
    global_fitness = score_latents(model, global_latents, task, args.eval_chunk)
    global_std = float(global_fitness.std())
    token_norm = float(global_latents.norm(dim=-1).mean())
    print(
        f"global fitness {global_fitness.mean():.4f} +/- {global_std:.4f} "
        f"over {args.num_global} random latents; "
        f"mean token norm {token_norm:.2f} (sqrt(D) = "
        f"{model.node_embedding_dim**0.5:.2f})\n"
    )

    anchors = random_latents(model, args.num_anchors)
    anchor_tokens = decode_tokens(model, anchors)
    anchor_fitness = score_latents(model, anchors, task, args.eval_chunk)

    header = (
        f"{'radius':>8} {'identical':>10} {'tok_delta':>10} "
        f"{'local_std':>10} {'/global':>8} {'mean|df|':>9}"
    )
    for anchor_index in range(args.num_anchors):
        print(
            f"anchor {anchor_index} (fitness {float(anchor_fitness[anchor_index]):.4f})"
        )
        print(header)
        for radius in args.radii:
            result = probe_radius(
                model=model,
                anchor=anchors[anchor_index],
                anchor_tokens=anchor_tokens[anchor_index],
                anchor_fitness=float(anchor_fitness[anchor_index]),
                task=task,
                radius=radius,
                num_probes=args.num_probes,
                eval_chunk=args.eval_chunk,
            )
            ratio = result.local_std / global_std if global_std > 0 else float("nan")
            print(
                f"{result.radius:>8.3f} {result.identical_fraction:>10.2f} "
                f"{result.token_hamming:>10.4f} {result.local_std:>10.5f} "
                f"{ratio:>8.2f} {result.mean_abs_delta:>9.5f}"
            )
        print()

    print(
        "read: 'identical' near 1.0 at a radius means the decode has not moved "
        "there, so zeroth-order gradients are exactly zero and a surrogate is "
        "mandatory. '/global' near 1.0 at the smallest radius where the decode "
        "does move means the landscape is white noise and latent search is the "
        "wrong tool; well below 1.0 means there is local structure to exploit."
    )


if __name__ == "__main__":
    main()
