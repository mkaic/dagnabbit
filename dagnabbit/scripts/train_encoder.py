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
    accuracy_summary,
    format_param_count,
    log_decoder_accuracies,
    log_run_config,
    log_step_metrics,
    step_preds_and_truth,
)


def _safe_mean(t: torch.Tensor) -> torch.Tensor:
    return t.mean() if t.numel() else t.new_zeros(())


def combine_losses(
    losses: TrainingStepLossReturnType,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    tf_pc_mean = losses.teacher_forced_primary_node_classification_losses.mean()
    tf_pr_mean = _safe_mean(
        losses.teacher_forced_primary_node_parent_reconstruction_losses
    )
    tf_pc_cons_mean = _safe_mean(
        losses.teacher_forced_primary_node_parent_consistency_losses
    )
    single_pc_mean = losses.single_sample_primary_node_classification_losses.mean()

    total = cfg.GLOBAL_LOSS_MULTIPLIER * (
        cfg.W_TF_PRIMARY_DECODED_CLASSIFICATION * tf_pc_mean
        + cfg.W_PRIMARY_SINGLE_SAMPLE_CLASSIFICATION * single_pc_mean
        + cfg.W_TF_PRIMARY_PARENT_RECONSTRUCTION * tf_pr_mean
        + cfg.W_TF_PRIMARY_PARENT_CONSISTENCY * tf_pc_cons_mean
    )

    # Keep components as tensors; materialize to floats (a GPU sync) only on the
    # steps that actually log them, rather than every step.
    components = {
        "tf_primary_decoded_classification": tf_pc_mean,
        "single_sample_classification": single_pc_mean,
        "tf_primary_parent_reconstruction": tf_pr_mean,
        "tf_primary_parent_consistency": tf_pc_cons_mean,
    }

    # The aggregate (autoregressive-with-aggregation) stream is only computed when
    # enabled; its loss tensors are otherwise zeros and must not be scored or
    # logged. Fold it in only when the model actually ran it.
    if cfg.COMPUTE_AGGREGATE_DECODE_PASS:
        pc_mean = losses.primary_node_classification_losses.mean()
        pr_mean = _safe_mean(losses.primary_node_parent_reconstruction_losses)
        pc_cons_mean = _safe_mean(losses.primary_node_parent_consistency_losses)
        total = total + cfg.GLOBAL_LOSS_MULTIPLIER * (
            cfg.W_PRIMARY_DECODED_CLASSIFICATION * pc_mean
            + cfg.W_PRIMARY_PARENT_RECONSTRUCTION * pr_mean
            + cfg.W_PRIMARY_PARENT_CONSISTENCY * pc_cons_mean
        )
        components["primary_decoded_classification"] = pc_mean
        components["primary_parent_reconstruction"] = pr_mean
        components["primary_parent_consistency"] = pc_cons_mean

    return total, components


LOSS_EMA_DECAY = 0.99


def apply_torch_compile(model: DagnabbitAutoEncoder, device: torch.device) -> None:
    if not cfg.TORCH_COMPILE:
        return

    if device.type != "cuda":
        print(f"torch_compile=skipped device={device.type} (CUDA only)")
        return

    compile_kwargs = {"dynamic": cfg.TORCH_COMPILE_DYNAMIC}
    if cfg.TORCH_COMPILE_CUDAGRAPHS:
        compile_kwargs["mode"] = cfg.TORCH_COMPILE_MODE
    else:
        compile_kwargs["options"] = {
            "triton.cudagraphs": False,
            "triton.cudagraph_trees": False,
        }
    model.node_encoder.forward_batch = torch.compile(
        model.node_encoder.forward_batch,
        **compile_kwargs,
    )
    model.node_decoder.forward_batch = torch.compile(
        model.node_decoder.forward_batch,
        **compile_kwargs,
    )
    compile_mode = cfg.TORCH_COMPILE_MODE if cfg.TORCH_COMPILE_CUDAGRAPHS else "default"
    print(
        "torch_compile=enabled "
        f"mode={compile_mode} "
        f"dynamic={cfg.TORCH_COMPILE_DYNAMIC} "
        f"cudagraphs={cfg.TORCH_COMPILE_CUDAGRAPHS}"
    )


def save_checkpoint(
    path: Path,
    step: int,
    graphs_seen: int,
    model: DagnabbitAutoEncoder,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss: float,
    best_loss: float | None = None,
) -> None:
    checkpoint = {
        # ``step`` is retained as the zero-based loop index for compatibility
        # with existing checkpoints. The other counters are unambiguous.
        "step": step,
        "completed_steps": step + 1,
        "graphs_seen": graphs_seen,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        # The best avg-loss seen so far, tracked separately from this
        # checkpoint's own ``loss`` so a resume from latest.ckpt (which is not
        # necessarily the best) does not reset the best-loss tracker.
        "best_loss": best_loss,
    }
    if lr_scheduler is not None:
        checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    torch.save(checkpoint, path)


def make_lr_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    warmup_steps = cfg.LR_WARMUP_OPTIMIZER_STEPS
    if warmup_steps <= 0:
        return None

    def lr_lambda(optimizer_step_index: int) -> float:
        return min((optimizer_step_index + 1) / warmup_steps, 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


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
        "--resume",
        default=None,
        help=(
            "Resume training from a checkpoint. Pass a run directory (its "
            "latest.ckpt is used, falling back to best.ckpt) or a specific "
            ".ckpt file. Model, optimizer, LR-scheduler, step counter and "
            "best-loss are restored, and TensorBoard logs continue in the same "
            "run directory. Ignores --run-name."
        ),
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compile the repeated encoder/decoder forward_batch kernels on CUDA. "
            "Defaults to TORCH_COMPILE in config.py."
        ),
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
    if args.compile is not None:
        cfg.TORCH_COMPILE = args.compile

    torch.manual_seed(cfg.SEED)
    torch.set_float32_matmul_precision("high")

    device = torch.device(cfg.DEVICE)
    if cfg.GRAPH_BATCH_SIZE <= 0:
        raise ValueError("GRAPH_BATCH_SIZE must be positive")
    if cfg.GRADIENT_ACCUMULATION_STEPS <= 0:
        raise ValueError("GRADIENT_ACCUMULATION_STEPS must be positive")

    checkpoint_every_graphs = cfg.CHECKPOINT_EVERY_GRAPHS
    checkpoint_every_steps: int | None = None
    if checkpoint_every_graphs is not None:
        if checkpoint_every_graphs <= 0:
            raise ValueError("CHECKPOINT_EVERY_GRAPHS must be positive or None")
        if checkpoint_every_graphs % cfg.GRAPH_BATCH_SIZE != 0:
            raise ValueError(
                "CHECKPOINT_EVERY_GRAPHS must be divisible by GRAPH_BATCH_SIZE"
            )
        checkpoint_every_steps = checkpoint_every_graphs // cfg.GRAPH_BATCH_SIZE
        if checkpoint_every_steps % cfg.GRADIENT_ACCUMULATION_STEPS != 0:
            raise ValueError(
                "CHECKPOINT_EVERY_GRAPHS must land on an optimizer-update boundary"
            )

    resume_checkpoint_path: Path | None = None
    if args.resume is not None:
        resume_arg = Path(args.resume)
        if resume_arg.is_dir():
            # Prefer latest.ckpt (saved every CHECK_BEST_EVERY steps) so a resume
            # skips the dry spell since the last best; fall back to best.ckpt.
            latest = resume_arg / "latest.ckpt"
            resume_checkpoint_path = latest if latest.exists() else resume_arg / "best.ckpt"
        else:
            resume_checkpoint_path = resume_arg
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(
                f"--resume checkpoint not found: {resume_checkpoint_path}"
            )
        # Continue logging into the run the checkpoint lives in so TensorBoard
        # sees one uninterrupted run rather than two disjoint ones.
        run_dir = resume_checkpoint_path.parent
        print(f"resuming from {resume_checkpoint_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{timestamp}-{args.run_name}" if args.run_name else timestamp
        run_dir = Path(cfg.TENSORBOARD_LOG_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={run_dir}")

    writer: SummaryWriter | None = None
    if not args.no_log:
        # Reopening a SummaryWriter on the same log_dir appends a new event file;
        # TensorBoard merges them into a single continuous run.
        writer = SummaryWriter(log_dir=str(run_dir))
        if resume_checkpoint_path is None:
            log_run_config(writer)
        print(f"tensorboard_log_dir={run_dir}")

    model = DagnabbitAutoEncoder(
        node_embedding_dim=cfg.NODE_EMBEDDING_DIM,
        trunk_node_type_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        mlp_expansion_factor=cfg.MLP_EXPANSION_FACTOR,
        reconstruction_detach_target=cfg.RECONSTRUCTION_DETACH_TARGET,
        compute_reconstruction_loss=cfg.COMPUTE_RECONSTRUCTION_LOSS,
        compute_aggregate_decode_pass=cfg.COMPUTE_AGGREGATE_DECODE_PASS,
        transformer_num_layers=cfg.TRANSFORMER_NUM_LAYERS,
        transformer_mlp_depth=cfg.TRANSFORMER_MLP_DEPTH,
        transformer_num_register_tokens=cfg.TRANSFORMER_NUM_REGISTER_TOKENS,
        transformer_num_heads=cfg.TRANSFORMER_NUM_HEADS,
        transformer_dropout=cfg.TRANSFORMER_DROPOUT,
    ).to(device)

    apply_torch_compile(model, device)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"trainable_parameters={format_param_count(num_trainable_params)} "
        f"({num_trainable_params})"
    )

    optimizer = cfg.OPTIMIZER_CLASS(model.parameters(), **cfg.OPTIMIZER_KWARGS)
    lr_scheduler = make_lr_warmup_scheduler(optimizer)
    if lr_scheduler is None:
        print("lr_warmup_optimizer_steps=0")
    else:
        target_lrs = ", ".join(f"{lr:.4g}" for lr in lr_scheduler.base_lrs)
        print(
            f"lr_warmup_optimizer_steps={cfg.LR_WARMUP_OPTIMIZER_STEPS} "
            f"initial_lr={optimizer.param_groups[0]['lr']:.4g} "
            f"target_lr={target_lrs}"
        )

    start_step = 0
    resumed_best_loss: float | None = None
    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        # ``completed_steps`` is the next zero-based loop index to run. Fall back
        # to ``step + 1`` for older checkpoints that predate that field.
        start_step = checkpoint.get("completed_steps", checkpoint["step"] + 1)
        # Restore the best loss so a resumed run does not overwrite best.ckpt
        # with a worse checkpoint before it beats the prior best. Prefer the
        # dedicated ``best_loss`` field; fall back to ``loss`` for checkpoints
        # predating it (where best.ckpt's loss is itself the best).
        resumed_best_loss = checkpoint.get("best_loss")
        if resumed_best_loss is None:
            resumed_best_loss = checkpoint.get("loss")
        resumed_graphs_seen = start_step * cfg.GRAPH_BATCH_SIZE
        print(
            f"resumed: start_step={start_step} graphs_seen={resumed_graphs_seen} "
            f"best_loss={resumed_best_loss}"
        )

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
        tf_window_preds: list[np.ndarray] = []
        tf_window_truth: list[np.ndarray] = []
        single_window_preds: list[np.ndarray] = []
        single_window_truth: list[np.ndarray] = []
        loss_ema: float | None = None
        best_loss: float | None = resumed_best_loss
        loss_window: deque[float] = deque(maxlen=cfg.CHECK_BEST_EVERY)
        last_grad_norm: float | None = None
        last_grad_was_clipped: bool | None = None
        last_optimizer_lr = optimizer.param_groups[0]["lr"]
        progress = tqdm(
            range(start_step, cfg.NUM_STEPS),
            initial=start_step,
            total=cfg.NUM_STEPS,
            unit="step",
        )
        for step in progress:
            # TensorBoard's Step axis is the historical per-graph training
            # coordinate, not the optimizer/update index.
            tensorboard_step = step * cfg.GRAPH_BATCH_SIZE
            graphs = [
                make_random_graph_description(
                    num_root_nodes=cfg.NUM_ROOT_NODES,
                    num_trunk_nodes=cfg.NUM_TRUNK_NODES,
                    num_output_nodes=cfg.NUM_OUTPUT_NODES,
                    trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
                    num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
                )
                for _ in range(cfg.GRAPH_BATCH_SIZE)
            ]

            losses = model.training_forward_batch(graphs)
            total, components = combine_losses(losses)

            scaled_loss = total / cfg.GRADIENT_ACCUMULATION_STEPS
            scaled_loss.backward()

            if (step + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                clip_max_norm = cfg.GRADIENT_CLIP_MAX_NORM
                if clip_max_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_max_norm
                    )
                    last_grad_norm = grad_norm.item()
                    last_grad_was_clipped = last_grad_norm > clip_max_norm
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf")
                    )
                    last_grad_norm = grad_norm.item()
                    last_grad_was_clipped = None
                last_optimizer_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

            if writer is not None:
                # Aggregate stream is only populated when enabled; skip its
                # accuracy bookkeeping otherwise (its logits are all zeros).
                if cfg.COMPUTE_AGGREGATE_DECODE_PASS:
                    step_preds, step_truth = step_preds_and_truth(
                        losses.primary_node_predicted_type_logits,
                        losses.primary_node_true_types,
                    )
                    window_preds.append(step_preds)
                    window_truth.append(step_truth)

                # Teacher-forced predictions share the autoregressive true labels.
                tf_step_preds, tf_step_truth = step_preds_and_truth(
                    losses.teacher_forced_primary_node_predicted_type_logits,
                    losses.primary_node_true_types,
                )
                tf_window_preds.append(tf_step_preds)
                tf_window_truth.append(tf_step_truth)

                # Single-sample stream shares the same true labels.
                single_step_preds, single_step_truth = step_preds_and_truth(
                    losses.single_sample_primary_node_predicted_type_logits,
                    losses.primary_node_true_types,
                )
                single_window_preds.append(single_step_preds)
                single_window_truth.append(single_step_truth)

            loss_val = total.item()
            graphs_seen = (step + 1) * cfg.GRAPH_BATCH_SIZE
            loss_window.append(loss_val)
            if loss_ema is None:
                loss_ema = loss_val
            else:
                loss_ema = LOSS_EMA_DECAY * loss_ema + (1 - LOSS_EMA_DECAY) * loss_val
            progress.set_postfix(loss_ema=f"{loss_ema:.4g}", refresh=False)

            if writer is not None and step % cfg.LOG_EVERY == 0:
                decoder_accuracy = None
                decoder_supertype_accuracies = None
                if cfg.COMPUTE_AGGREGATE_DECODE_PASS:
                    (
                        decoder_accuracy,
                        decoder_supertype_accuracies,
                    ) = accuracy_summary(
                        np.concatenate(window_preds),
                        np.concatenate(window_truth),
                        num_classes=model.num_node_types,
                    )
                    window_preds.clear()
                    window_truth.clear()
                (
                    tf_decoder_accuracy,
                    tf_decoder_supertype_accuracies,
                ) = accuracy_summary(
                    np.concatenate(tf_window_preds),
                    np.concatenate(tf_window_truth),
                    num_classes=model.num_node_types,
                )
                (
                    single_decoder_accuracy,
                    single_decoder_supertype_accuracies,
                ) = accuracy_summary(
                    np.concatenate(single_window_preds),
                    np.concatenate(single_window_truth),
                    num_classes=model.num_node_types,
                )
                tf_window_preds.clear()
                tf_window_truth.clear()
                single_window_preds.clear()
                single_window_truth.clear()
                # Single-sample accuracy is the headline blind-decode-regime metric;
                # log it next to the autoregressive and tf curves.
                log_decoder_accuracies(
                    writer,
                    tensorboard_step,
                    single_decoder_accuracy,
                    single_decoder_supertype_accuracies,
                    mean_tag="accuracy/single/decoder_mean",
                    tag_prefix="accuracy/single",
                )
                log_step_metrics(
                    writer,
                    tensorboard_step,
                    loss_val,
                    {name: value.item() for name, value in components.items()},
                    decoder_accuracy,
                    decoder_supertype_accuracies,
                    tf_decoder_accuracy,
                    tf_decoder_supertype_accuracies,
                    grad_norm=last_grad_norm,
                    grad_was_clipped=last_grad_was_clipped,
                    learning_rate=last_optimizer_lr,
                )

            if (step + 1) % cfg.CHECK_BEST_EVERY == 0:
                avg_loss = sum(loss_window) / len(loss_window)
                if best_loss is None or avg_loss < best_loss:
                    best_loss = avg_loss
                    checkpoint_path = run_dir / "best.ckpt"
                    save_checkpoint(
                        checkpoint_path,
                        step,
                        graphs_seen,
                        model,
                        optimizer,
                        lr_scheduler,
                        avg_loss,
                        best_loss=best_loss,
                    )
                    progress.write(
                        f"step {step + 1}: new best avg loss {avg_loss:.4g} "
                        f"-> {checkpoint_path}"
                    )
                # Always refresh latest.ckpt so a resume can skip the dry spell
                # since the last best rather than replaying it.
                save_checkpoint(
                    run_dir / "latest.ckpt",
                    step,
                    graphs_seen,
                    model,
                    optimizer,
                    lr_scheduler,
                    avg_loss,
                    best_loss=best_loss,
                )

            if (
                checkpoint_every_steps is not None
                and (step + 1) % checkpoint_every_steps == 0
            ):
                checkpoint_dir = run_dir / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                checkpoint_path = checkpoint_dir / f"graphs-{graphs_seen:09d}.ckpt"
                save_checkpoint(
                    checkpoint_path,
                    step,
                    graphs_seen,
                    model,
                    optimizer,
                    lr_scheduler,
                    loss_val,
                )
                progress.write(
                    f"graphs {graphs_seen}: saved checkpoint -> {checkpoint_path}"
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
