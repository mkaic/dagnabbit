import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder, TrainingStepLossReturnType
from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.scripts import config as cfg
from dagnabbit.scripts.logging_utils import (
    format_param_count,
    log_run_config,
    log_step_metrics,
    per_type_accuracies,
    step_preds_and_truth,
)


def combine_losses(
    losses: TrainingStepLossReturnType,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cc_mean = losses.condenser_node_classification_losses.mean()
    cr_mean = losses.condenser_node_reconstruction_losses.mean()
    pc_mean = losses.primary_node_classification_losses.mean()
    pr_mean = losses.primary_node_reconstruction_losses.mean()

    tf_cc_mean = losses.teacher_forced_condenser_node_classification_losses.mean()
    tf_cr_mean = losses.teacher_forced_condenser_node_reconstruction_losses.mean()
    tf_pc_mean = losses.teacher_forced_primary_node_classification_losses.mean()
    tf_pr_mean = losses.teacher_forced_primary_node_reconstruction_losses.mean()

    total = cfg.GLOBAL_LOSS_MULTIPLIER * (
        cfg.W_CONDENSER_DECODED_CLASSIFICATION * cc_mean
        + cfg.W_CONDENSER_RECONSTRUCTION * cr_mean
        + cfg.W_PRIMARY_DECODED_CLASSIFICATION * pc_mean
        + cfg.W_PRIMARY_RECONSTRUCTION * pr_mean
        + cfg.W_TF_CONDENSER_DECODED_CLASSIFICATION * tf_cc_mean
        + cfg.W_TF_CONDENSER_RECONSTRUCTION * tf_cr_mean
        + cfg.W_TF_PRIMARY_DECODED_CLASSIFICATION * tf_pc_mean
        + cfg.W_TF_PRIMARY_RECONSTRUCTION * tf_pr_mean
    )

    # Keep components as tensors; materialize to floats (a GPU sync) only on the
    # steps that actually log them, rather than every step.
    components = {
        "condenser_decoded_classification": cc_mean,
        "condenser_reconstruction": cr_mean,
        "primary_decoded_classification": pc_mean,
        "primary_reconstruction": pr_mean,
        "tf_condenser_decoded_classification": tf_cc_mean,
        "tf_condenser_reconstruction": tf_cr_mean,
        "tf_primary_decoded_classification": tf_pc_mean,
        "tf_primary_reconstruction": tf_pr_mean,
    }
    return total, components


LOSS_EMA_DECAY = 0.99


def save_checkpoint(
    path: Path,
    step: int,
    model: DagnabbitAutoEncoder,
    optimizer: torch.optim.Optimizer,
    loss: float,
) -> None:
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def dump_profile(
    prof: profile,
    output_dir: Path,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=30)
    print("\n==== top ops by", sort_key, "====")
    print(table)
    (output_dir / "op_table.txt").write_text(table)

    trace_path = output_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"chrome_trace={trace_path}  (open in chrome://tracing or ui.perfetto.dev)")

    if device.type == "cuda":
        snapshot_path = output_dir / "mem_snapshot.pickle"
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"memory_snapshot={snapshot_path}  (open at pytorch.org/memory_viz)")
        print(
            "max_memory_allocated_gb="
            f"{torch.cuda.max_memory_allocated() / 1e9:.3f}  "
            "max_memory_reserved_gb="
            f"{torch.cuda.max_memory_reserved() / 1e9:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DAG autoencoder.")
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Name for this TensorBoard run (creates runs/<name>/). "
            "Defaults to a timestamp."
        ),
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable TensorBoard logging.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Run a short torch.profiler + CUDA memory-snapshot pass instead of a "
            "full training run, write artifacts to --profile-output-dir, then exit."
        ),
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=8,
        help="Number of profiled (active) steps after warmup. Only used with --profile.",
    )
    parser.add_argument(
        "--profile-output-dir",
        default="profile_out",
        help="Directory for profiler artifacts (trace.json, mem_snapshot.pickle).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(cfg.SEED)

    device = torch.device(cfg.DEVICE)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}-{args.run_name}" if args.run_name else timestamp
    run_dir = Path(cfg.TENSORBOARD_LOG_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={run_dir}")

    writer: SummaryWriter | None = None
    if not args.no_log:
        writer = SummaryWriter(log_dir=str(run_dir))
        log_run_config(writer)
        print(f"tensorboard_log_dir={run_dir}")

    model = DagnabbitAutoEncoder(
        node_embedding_dim=cfg.NODE_EMBEDDING_DIM,
        trunk_node_type_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        condenser_node_type_in_degree=cfg.CONDENSER_NODE_TYPE_IN_DEGREE,
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        mlp_depth=cfg.MLP_DEPTH,
        mlp_expansion_factor=cfg.MLP_EXPANSION_FACTOR,
    ).to(device)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"trainable_parameters={format_param_count(num_trainable_params)} "
        f"({num_trainable_params})"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    prof: profile | None = None
    total_profile_steps = 0
    if args.profile:
        # wait: skip the very first (cold) step; warmup: let allocator/caches and
        # cudnn autotune settle so we measure steady-state; active: recorded steps.
        wait, warmup = 1, 2
        total_profile_steps = wait + warmup + args.profile_steps
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
            torch.cuda.memory._record_memory_history(max_entries=100_000)
        prof = profile(
            activities=activities,
            schedule=schedule(
                wait=wait, warmup=warmup, active=args.profile_steps, repeat=1
            ),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        prof.start()
        print(
            f"profiling: running {total_profile_steps} steps "
            f"(wait={wait}, warmup={warmup}, active={args.profile_steps})"
        )

    try:
        optimizer.zero_grad()
        window_preds: list[np.ndarray] = []
        window_truth: list[np.ndarray] = []
        condenser_window_preds: list[np.ndarray] = []
        condenser_window_truth: list[np.ndarray] = []
        tf_window_preds: list[np.ndarray] = []
        tf_window_truth: list[np.ndarray] = []
        tf_condenser_window_preds: list[np.ndarray] = []
        tf_condenser_window_truth: list[np.ndarray] = []
        loss_ema: float | None = None
        best_loss: float | None = None
        loss_window: deque[float] = deque(maxlen=cfg.CHECK_BEST_EVERY)
        progress = tqdm(range(cfg.NUM_STEPS), unit="step")
        for step in progress:
            graph = make_random_graph_description(
                num_root_nodes=cfg.NUM_ROOT_NODES,
                num_trunk_nodes=cfg.NUM_TRUNK_NODES,
                num_output_nodes=cfg.NUM_OUTPUT_NODES,
                trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
                num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
            )

            autocast_enabled = device.type == "cuda"
            # Standardize each linear's weight once for this step (in the autocast
            # compute dtype so the bf16 cast is also done once) and reuse it for
            # every node visit, instead of recomputing it per call. Backward stays
            # inside this context (it needs the cached weights) but outside the
            # autocast region, as recommended.
            cache_dtype = torch.bfloat16 if autocast_enabled else None
            with model.cached_standardized_weights(dtype=cache_dtype):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    losses = model.training_forward(graph)
                total, components = combine_losses(losses)

                scaled_loss = total / cfg.GRADIENT_ACCUMULATION_STEPS
                scaled_loss.backward()

            if (step + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                if cfg.GRADIENT_CLIP_MAX_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.GRADIENT_CLIP_MAX_NORM
                    )
                optimizer.step()
                optimizer.zero_grad()

            if writer is not None:
                step_preds, step_truth = step_preds_and_truth(
                    losses.primary_node_predicted_type_logits,
                    losses.primary_node_true_types,
                )
                window_preds.append(step_preds)
                window_truth.append(step_truth)

                condenser_step_preds, condenser_step_truth = step_preds_and_truth(
                    losses.condenser_node_predicted_type_logits,
                    losses.condenser_node_true_types,
                )
                condenser_window_preds.append(condenser_step_preds)
                condenser_window_truth.append(condenser_step_truth)

                # Teacher-forced predictions share the autoregressive true labels.
                tf_step_preds, tf_step_truth = step_preds_and_truth(
                    losses.teacher_forced_primary_node_predicted_type_logits,
                    losses.primary_node_true_types,
                )
                tf_window_preds.append(tf_step_preds)
                tf_window_truth.append(tf_step_truth)

                tf_condenser_step_preds, tf_condenser_step_truth = step_preds_and_truth(
                    losses.teacher_forced_condenser_node_predicted_type_logits,
                    losses.condenser_node_true_types,
                )
                tf_condenser_window_preds.append(tf_condenser_step_preds)
                tf_condenser_window_truth.append(tf_condenser_step_truth)

            loss_val = total.item()
            loss_window.append(loss_val)
            if loss_ema is None:
                loss_ema = loss_val
            else:
                loss_ema = LOSS_EMA_DECAY * loss_ema + (1 - LOSS_EMA_DECAY) * loss_val
            progress.set_postfix(loss_ema=f"{loss_ema:.4g}", refresh=False)

            if writer is not None and step % cfg.LOG_EVERY == 0:
                decoder_accuracies = per_type_accuracies(
                    np.concatenate(window_preds),
                    np.concatenate(window_truth),
                    num_classes=model.num_node_types,
                )
                condenser_decoder_accuracies = per_type_accuracies(
                    np.concatenate(condenser_window_preds),
                    np.concatenate(condenser_window_truth),
                    num_classes=model.num_node_types,
                )
                tf_decoder_accuracies = per_type_accuracies(
                    np.concatenate(tf_window_preds),
                    np.concatenate(tf_window_truth),
                    num_classes=model.num_node_types,
                )
                tf_condenser_decoder_accuracies = per_type_accuracies(
                    np.concatenate(tf_condenser_window_preds),
                    np.concatenate(tf_condenser_window_truth),
                    num_classes=model.num_node_types,
                )
                window_preds.clear()
                window_truth.clear()
                condenser_window_preds.clear()
                condenser_window_truth.clear()
                tf_window_preds.clear()
                tf_window_truth.clear()
                tf_condenser_window_preds.clear()
                tf_condenser_window_truth.clear()
                log_step_metrics(
                    writer,
                    step,
                    loss_val,
                    {name: value.item() for name, value in components.items()},
                    decoder_accuracies,
                    condenser_decoder_accuracies,
                    tf_decoder_accuracies,
                    tf_condenser_decoder_accuracies,
                )

            if (step + 1) % cfg.CHECK_BEST_EVERY == 0:
                avg_loss = sum(loss_window) / len(loss_window)
                if best_loss is None or avg_loss < best_loss:
                    best_loss = avg_loss
                    checkpoint_path = run_dir / "best.ckpt"
                    save_checkpoint(
                        checkpoint_path,
                        step,
                        model,
                        optimizer,
                        avg_loss,
                    )
                    progress.write(
                        f"step {step + 1}: new best avg loss {avg_loss:.4g} "
                        f"-> {checkpoint_path}"
                    )

            if prof is not None:
                prof.step()
                if step + 1 >= total_profile_steps:
                    break
    finally:
        if prof is not None:
            prof.stop()
            dump_profile(prof, Path(args.profile_output_dir), device)
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
