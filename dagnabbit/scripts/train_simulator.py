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
actual target circuit. Two metrics separate the cases:

``eval/mcc`` -- Matthews correlation between predicted and exact bits. It is 0
for any constant predictor no matter how lopsided the target, so it only moves
when a prediction tracks the *specific* circuit. Accuracy climbing while MCC
sits at zero is the signature of a model that has learned the marginal bit.

``eval/accuracy_by_rank`` and ``eval/mcc_by_rank`` -- the same two bucketed by
each output node's longest-path depth. If they fall off a cliff at the
simulator's layer count, the receptive field is binding and the fix is more
layers (or a weight-tied recurrent simulator). If they degrade smoothly past the
layer count, the model found something cheaper than hop-by-hop propagation.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dagnabbit.dag.graphs import sample as sample_graphs
from dagnabbit.dag.metrics import (
    bucket_by_rank,
    confusion_counts,
    matthews_correlation,
)
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
from dagnabbit.tasks.logic_gates.reference_circuits import (
    mixed_ripple_carry_adder,
    nand_ripple_carry_adder,
)


def make_batch(batch_size: int, root_values: torch.Tensor, device: torch.device):
    """Fresh graphs plus their exact packed truth tables."""
    graphs = sample_graphs(batch_size, cfg.GEOMETRY, device, sampling=cfg.SAMPLING)
    return graphs, evaluate_graphs(graphs, root_values, cfg.GATES.operators)


def loss_and_accuracy(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """BCE loss, and per-(graph, output) bit accuracy averaged over patches."""
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    correct = ((logits > 0).float() == targets).float()
    # [B, K, C, rows] -> [B, C]: mean over patches and rows.
    return loss, correct.mean(dim=3).mean(dim=1)


@dataclass(frozen=True)
class EvalResult:
    """One depth-stratified evaluation pass.

    ``mcc`` averages only over outputs whose target actually varied over the
    bits scored; ``constant_target_fraction`` is how many were excluded. A high
    exclusion rate is itself informative -- it means the sampled circuits are
    collapsing to constants, and bit accuracy on those outputs is measuring
    almost nothing.
    """

    bit_accuracy: float
    mcc: float
    constant_target_fraction: float
    # Mean gates per graph, out of GEOMETRY.num_trunk_nodes positions. Fixed
    # unless SAMPLING.minimum_trunk_nodes is set; logged so a change to the
    # sampling prior is visible in the same place as its effect on the ladder.
    mean_live_trunk_nodes: float
    accuracy_by_rank: dict[int, float]
    mcc_by_rank: dict[int, float]
    outputs_by_rank: dict[int, int]


@torch.no_grad()
def evaluate(
    model: GraphSimulator,
    root_values: torch.Tensor,
    rows_per_patch: int,
    device: torch.device,
) -> EvalResult:
    """Bit accuracy and MCC on fresh graphs, bucketed by each output's depth."""
    model.eval()
    graphs, packed = make_batch(cfg.EVAL_BATCH_SIZE, root_values, device)
    patch_indices = sample_patch_indices(
        cfg.MODEL.num_patches, cfg.EVAL_PATCHES, device
    )
    with torch.autocast(
        device_type=device.type, dtype=cfg.AMP_DTYPE, enabled=cfg.AMP_ENABLED
    ):
        logits = model.forward_graphs(graphs, patch_indices)
    logits = logits.float()
    targets = patch_targets(packed, patch_indices, rows_per_patch)

    _, accuracy = loss_and_accuracy(logits, targets)
    mcc, defined = matthews_correlation(confusion_counts(logits, targets))

    # [B, C] scores against [B, C] output ranks, flattened and bucketed.
    ranks = graphs.output_ranks.reshape(-1)
    accuracy_by_rank, outputs_by_rank = bucket_by_rank(accuracy.reshape(-1), ranks)
    mcc_by_rank, _ = bucket_by_rank(mcc.reshape(-1), ranks, valid=defined.reshape(-1))

    defined_count = int(defined.sum())
    model.train()
    return EvalResult(
        bit_accuracy=float(accuracy.mean()),
        mcc=float(mcc.reshape(-1)[defined.reshape(-1)].mean())
        if defined_count
        else 0.0,
        constant_target_fraction=1.0 - defined_count / defined.numel(),
        mean_live_trunk_nodes=float(graphs.num_live_trunk_nodes.double().mean()),
        accuracy_by_rank=accuracy_by_rank,
        mcc_by_rank=mcc_by_rank,
        outputs_by_rank=outputs_by_rank,
    )


# Each reference circuit is written in a particular vocabulary, so a configured
# gate set that lacks one of its gates cannot build it at all. The requirement
# is declared here rather than discovered by catching the builder's error: a
# probe is a side observation, and a gate set chosen for the *training*
# distribution should not abort the run because a probe happens to be
# unbuildable. Catching the exception instead would also swallow a genuinely
# mis-wired reference, which is exactly what the fitness assertion below exists
# to catch.
REFERENCE_PROBES: tuple[tuple[str, object, tuple[str, ...]], ...] = (
    ("nand", nand_ripple_carry_adder, ("NAND",)),
    ("mixed", mixed_ripple_carry_adder, ("NAND", "XOR")),
)


def available_probes(gates) -> list[tuple[str, object]]:
    """The reference circuits ``gates`` can express, in declaration order."""
    return [
        (name, build)
        for name, build, required in REFERENCE_PROBES
        if all(gate in gates.names for gate in required)
    ]


@torch.no_grad()
def probe_references(
    model: GraphSimulator, rows_per_patch: int, device: torch.device
) -> dict[str, tuple[float, float]]:
    """Predict the hand-built adders' behaviour. Returns {name: (accuracy, mcc)}.

    Not a training signal -- an out-of-distribution check. Two circuits compute
    the same function from different vocabularies: ``nand`` is all-NAND with a
    67-gate core, ``mixed`` spends XOR where the NAND version spends four gates
    and has a 35-gate core. Both are padded to the same trunk budget and end up
    at comparable depth, so the difference between the two scores isolates gate
    vocabulary from depth and structure.

    MCC matters more here than on random graphs: every bit of ``a + b`` is
    balanced almost exactly 50/50 over the full table, so bit accuracy has a
    hard floor near 0.5 that says nothing, while MCC starts at 0.

    Only the probes ``cfg.GATES`` can express are run -- see
    :func:`available_probes`. With the default gate set that is both of them.
    """
    task = adder_task(device)
    patch_indices = torch.arange(cfg.MODEL.num_patches, device=device)
    results: dict[str, tuple[float, float]] = {}

    for name, build in available_probes(cfg.GATES):
        graphs = build(cfg.GEOMETRY, cfg.GATES).to(device)
        with torch.autocast(
            device_type=device.type, dtype=cfg.AMP_DTYPE, enabled=cfg.AMP_ENABLED
        ):
            logits = model.forward_graphs(graphs, patch_indices)

        # The circuit really computes the adder, so the task's targets are its
        # truth table; this checks the *prediction* against it. Sanity-check the
        # premise too, since a mis-wired reference would make this meaningless.
        packed = evaluate_graphs(graphs, task.root_values, cfg.GATES.operators)
        exact, _ = bit_accuracy(packed, task)
        assert float(exact[0]) == 1.0, f"the {name} reference is not an adder"

        targets = patch_targets(packed, patch_indices, rows_per_patch)
        logits = logits.float()
        _, accuracy = loss_and_accuracy(logits, targets)
        mcc, _ = matthews_correlation(confusion_counts(logits, targets))
        results[name] = (float(accuracy.mean()), float(mcc.mean()))

    return results


def format_rank_ladder(
    result: EvalResult, per_line: int = 4, indent: str = "  "
) -> list[str]:
    """The depth ladder as fixed-width columns, wrapped.

    Each cell is ``rank, bit accuracy, MCC, outputs in the bucket``. Printed
    rather than only logged because a cliff at the simulator's layer count is
    the thing worth catching by eye, and that is far easier to see in a grid
    than in a scrolling stream of scalars.

    Accuracy and MCC sit side by side because the gap between them is the
    signal: accuracy drifting up while MCC stays at zero means the model is
    learning the marginal bit, not the circuit. The sample count is there
    because the deepest ranks are both the ones that matter and the ones with
    the fewest outputs -- an ``n2`` bucket swinging around is noise, not a cliff.
    """
    cells = [
        f"r{rank:<2} {result.accuracy_by_rank[rank]:.3f} "
        f"{result.mcc_by_rank.get(rank, float('nan')):+.3f} "
        f"n{result.outputs_by_rank[rank]:<4}"
        for rank in sorted(result.accuracy_by_rank)
    ]
    return [
        indent + " ".join(cells[start : start + per_line]).rstrip()
        for start in range(0, len(cells), per_line)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=None,
        help="label appended to the run directory's timestamp prefix",
    )
    parser.add_argument("--steps", type=int, default=cfg.NUM_STEPS)
    parser.add_argument("--device", default=cfg.DEVICE)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(cfg.SEED)
    device = torch.device(args.device)
    # Timestamp first, always: the run directories are read in a file browser
    # and in ``ls``, where lexical order is the only order on offer. ``--name``
    # is a suffix rather than the whole name, because a bare name sorts
    # alphabetically and buries the newest run in the middle of the list.
    run_name = time.strftime("%Y%m%d-%H%M%S")
    if args.name:
        run_name = f"{run_name}-{args.name}"
    run_directory = Path(cfg.LOG_DIR) / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_directory))

    model = GraphSimulator(cfg.GEOMETRY, cfg.MODEL).to(device)
    print(f"simulator: {format_parameter_count(parameter_count(model))} parameters")
    print(f"gates: {', '.join(cfg.GATES.names)}")
    # Say which probes are running. A configured gate set that cannot express a
    # reference circuit silently drops its curve, and a missing curve is far
    # easier to misread as a broken probe than as a deliberate skip.
    active = {name for name, _ in available_probes(cfg.GATES)}
    for name, _, required in REFERENCE_PROBES:
        if name in active:
            print(f"probe {name}: on")
        else:
            missing = [gate for gate in required if gate not in cfg.GATES.names]
            print(f"probe {name}: off -- needs {', '.join(missing)}")
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
    progress = tqdm(range(args.steps), unit="step", dynamic_ncols=True)
    postfix: dict[str, str] = {}
    window_started = time.time()
    window_start_step = 0

    for step in progress:
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
            # Rate over the last window rather than since launch, so a slowdown
            # shows up instead of being averaged away by a fast start.
            elapsed = max(time.time() - window_started, 1e-9)
            graphs_per_second = (
                cfg.GRAPH_BATCH_SIZE * (step - window_start_step + 1) / elapsed
            )
            window_started = time.time()
            window_start_step = step

            writer.add_scalar("train/loss", float(loss), step)
            writer.add_scalar("train/bit_accuracy", float(accuracy.mean()), step)
            writer.add_scalar("train/graphs_per_second", graphs_per_second, step)
            postfix["loss"] = f"{float(loss):.4f}"
            postfix["acc"] = f"{float(accuracy.mean()):.4f}"
            postfix["g/s"] = f"{graphs_per_second:.0f}"
            progress.set_postfix(postfix, refresh=False)

        if step % cfg.EVAL_EVERY == 0:
            result = evaluate(model, root_values, rows_per_patch, device)
            probes = probe_references(model, rows_per_patch, device)

            writer.add_scalar("eval/bit_accuracy", result.bit_accuracy, step)
            writer.add_scalar("eval/mcc", result.mcc, step)
            writer.add_scalar(
                "eval/constant_target_fraction", result.constant_target_fraction, step
            )
            writer.add_scalar(
                "eval/live_trunk_nodes", result.mean_live_trunk_nodes, step
            )
            for rank, value in result.accuracy_by_rank.items():
                writer.add_scalar(f"eval/accuracy_by_rank/{rank:02d}", value, step)
                writer.add_scalar(
                    f"eval/outputs_at_rank/{rank:02d}",
                    result.outputs_by_rank[rank],
                    step,
                )
            for rank, value in result.mcc_by_rank.items():
                writer.add_scalar(f"eval/mcc_by_rank/{rank:02d}", value, step)
            for name, (probe_accuracy, probe_mcc) in probes.items():
                writer.add_scalar(
                    f"probe/{name}_adder_bit_accuracy", probe_accuracy, step
                )
                writer.add_scalar(f"probe/{name}_adder_mcc", probe_mcc, step)

            postfix["eval"] = f"{result.bit_accuracy:.4f}"
            postfix["mcc"] = f"{result.mcc:+.4f}"
            if probes:
                postfix["adder"] = "/".join(f"{mcc:+.3f}" for _, mcc in probes.values())
            progress.set_postfix(postfix, refresh=False)

            lines = [
                f"step {step:>8}  acc {result.bit_accuracy:.4f}  "
                f"mcc {result.mcc:+.4f}  "
                + "  ".join(
                    f"{name} {acc:.4f}/{mcc:+.4f}"
                    for name, (acc, mcc) in probes.items()
                )
                + "  "
                f"const {result.constant_target_fraction:.1%}  "
                f"gates {result.mean_live_trunk_nodes:.1f}",
                f"  cells: rank, accuracy, mcc, outputs -- past "
                f"r{cfg.MODEL.num_simulator_layers} exceeds the simulator's depth",
                *format_rank_ladder(result),
            ]
            progress.write("\n".join(lines))

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

    progress.close()
    writer.close()


if __name__ == "__main__":
    main()
