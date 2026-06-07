import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
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
) -> tuple[torch.Tensor, dict[str, float]]:
    cc_mean = torch.stack(losses.condenser_node_classification_losses).mean()
    cs_mean = torch.stack(
        losses.condenser_node_predicted_embeddings_similarity_losses
    ).mean()
    pc_mean = torch.stack(losses.primary_node_classification_losses).mean()
    ps_mean = torch.stack(
        losses.primary_node_predicted_embeddings_similarity_losses
    ).mean()

    total = (
        cfg.W_CONDENSER_DECODED_CLASSIFICATION * cc_mean
        + cfg.W_CONDENSER_DECODED_SIMILARITY * cs_mean
        + cfg.W_PRIMARY_DECODED_CLASSIFICATION * pc_mean
        + cfg.W_PRIMARY_DECODED_SIMILARITY * ps_mean
    )

    components = {
        "condenser_decoded_classification": cc_mean.item(),
        "condenser_decoded_similarity": cs_mean.item(),
        "primary_decoded_classification": pc_mean.item(),
        "primary_decoded_similarity": ps_mean.item(),
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
    ).to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"trainable_parameters={format_param_count(num_trainable_params)} "
        f"({num_trainable_params})"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    try:
        optimizer.zero_grad()
        window_preds: list[np.ndarray] = []
        window_truth: list[np.ndarray] = []
        condenser_window_preds: list[np.ndarray] = []
        condenser_window_truth: list[np.ndarray] = []
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

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
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
                window_preds.clear()
                window_truth.clear()
                condenser_window_preds.clear()
                condenser_window_truth.clear()
                log_step_metrics(
                    writer,
                    step,
                    loss_val,
                    components,
                    decoder_accuracies,
                    condenser_decoder_accuracies,
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
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
