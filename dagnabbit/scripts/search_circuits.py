"""Search for a circuit that solves the 8-bit adder task.

Two methods on the same budget and the same fitness function:

* ``discrete`` -- a (mu + lambda) ES with point mutations on graph structure.
* ``latent`` -- CEM over a trained autoencoder's graph latent, decoding each
  sample into a circuit.

``both`` runs them back to back on an equal evaluation budget, which is the
only way to tell whether the latent representation is buying anything. A known
optimum exists inside the node budget (see
:func:`~dagnabbit.tasks.logic_gates.reference_circuits.nand_ripple_carry_adder`),
so a fitness short of 1.0 is a shortfall of the search, not of the search space.

Usage::

    python -m dagnabbit.scripts.search_circuits discrete --generations 2000
    python -m dagnabbit.scripts.search_circuits latent --checkpoint runs/<run>
    python -m dagnabbit.scripts.search_circuits both --checkpoint runs/<run>
"""

import argparse
import json
from pathlib import Path

import torch

from dagnabbit.dag.checkpoint import load_model
from dagnabbit.scripts import config as cfg
from dagnabbit.search.common import SearchResult
from dagnabbit.search.discrete import evolve_discrete
from dagnabbit.search.latent import evolve_latent, measure_latent_geometry
from dagnabbit.tasks.logic_gates.evaluate import adder_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "method",
        choices=["discrete", "latent", "both"],
        help="Which search to run.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Autoencoder checkpoint or run directory. Required for latent search.",
    )
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument(
        "--offspring",
        type=int,
        default=16,
        help="Offspring per generation for the discrete ES.",
    )
    parser.add_argument("--parents", type=int, default=1)
    parser.add_argument("--mutation-rate", type=float, default=0.03)
    parser.add_argument(
        "--population",
        type=int,
        default=64,
        help="Samples per generation for latent CEM.",
    )
    parser.add_argument("--elite-fraction", type=float, default=0.125)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help=(
            "Total evaluations to allow. Overrides --generations for each "
            "method, so 'both' is an equal-budget comparison."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Generations between progress lines (0 to silence).",
    )
    parser.add_argument(
        "--history-out",
        default=None,
        help="Write per-generation history to this JSON file.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_reporter(label: str, log_every: int):
    def report(record) -> None:
        if log_every and (
            record.generation % log_every == 0 or record.best_fitness >= 1.0
        ):
            rate = record.evaluations / max(record.elapsed_seconds, 1e-9)
            print(
                f"  [{label}] gen {record.generation:>6}  "
                f"evals {record.evaluations:>8}  "
                f"best {record.best_fitness:.6f}  "
                f"mean {record.mean_fitness:.6f}  "
                f"{rate:>6.0f} eval/s"
            )

    return report


def summarize(label: str, result: SearchResult) -> None:
    elapsed = result.history[-1].elapsed_seconds if result.history else 0.0
    print(
        f"\n{label}: best {result.best_fitness:.6f} after "
        f"{result.evaluations} evaluations / {result.generations} generations "
        f"({elapsed:.1f}s, {result.evaluations / max(elapsed, 1e-9):.0f} eval/s)"
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    task = adder_task(device)
    results: dict[str, SearchResult] = {}

    if args.method in ("discrete", "both"):
        generations = args.generations
        if args.budget is not None:
            # The first generation evaluates the parents, every later one the
            # offspring, so the budget maps to generations directly.
            generations = max(1, (args.budget - args.parents) // args.offspring)
        print(f"discrete ES: {generations} generations x {args.offspring} offspring")
        results["discrete"] = evolve_discrete(
            task,
            num_generations=generations,
            num_parents=args.parents,
            num_offspring=args.offspring,
            mutation_rate=args.mutation_rate,
            seed=args.seed,
            num_root_nodes=cfg.NUM_ROOT_NODES,
            num_trunk_nodes=cfg.NUM_TRUNK_NODES,
            num_output_nodes=cfg.NUM_OUTPUT_NODES,
            num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
            trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
            on_generation=make_reporter("discrete", args.log_every),
        )
        summarize("discrete", results["discrete"])

    if args.method in ("latent", "both"):
        if args.checkpoint is None:
            raise SystemExit("latent search needs --checkpoint")
        model, checkpoint = load_model(args.checkpoint, device)
        print(
            f"\nloaded {args.checkpoint} "
            f"(graphs_seen={checkpoint.get('graphs_seen')}, "
            f"D={model.node_embedding_dim}, "
            f"latent {model.num_output_nodes}x{model.node_embedding_dim})"
        )
        geometry = measure_latent_geometry(model, seed=args.seed)
        print(
            f"latent geometry: token norm {geometry.token_norm:.3f}, "
            f"mean per-dim sigma {geometry.std.mean():.4f}"
        )

        generations = args.generations
        if args.budget is not None:
            generations = max(1, args.budget // args.population)
        print(f"latent CEM: {generations} generations x {args.population} samples")
        results["latent"] = evolve_latent(
            model,
            task,
            num_generations=generations,
            population_size=args.population,
            elite_fraction=args.elite_fraction,
            sigma_scale=args.sigma_scale,
            seed=args.seed,
            geometry=geometry,
            on_generation=make_reporter("latent", args.log_every),
        )
        summarize("latent", results["latent"])

    if len(results) > 1:
        print("\n--- comparison ---")
        for label, result in results.items():
            print(
                f"  {label:>9}: {result.best_fitness:.6f} "
                f"in {result.evaluations} evaluations"
            )

    if args.history_out:
        path = Path(args.history_out)
        path.write_text(
            json.dumps(
                {
                    label: {
                        "best_fitness": result.best_fitness,
                        "evaluations": result.evaluations,
                        "history": [record.__dict__ for record in result.history],
                    }
                    for label, result in results.items()
                },
                indent=2,
            )
        )
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
