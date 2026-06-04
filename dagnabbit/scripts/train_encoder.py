import argparse
import logging
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder, TrainingStepLossReturnType
from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.scripts import config as cfg


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


def format_param_count(n: int) -> str:
    """Format a parameter count as a short human-readable string (e.g. 1.23M)."""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= threshold:
            return f"{n / threshold:.2f}{suffix}"
    return str(n)


def step_preds_and_truth(
    logits_per_node: list[torch.Tensor],
    true_types: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract argmax predictions and true class ids for one step."""
    preds = torch.stack(logits_per_node).detach().argmax(dim=-1).cpu().numpy()
    truth = np.asarray(true_types, dtype=np.int64)
    return preds, truth


def per_type_accuracies(
    preds: np.ndarray,
    truth: np.ndarray,
    num_classes: int,
) -> dict[int, float]:
    """Per-class accuracy (recall): argmax==c rate over nodes whose true type is c.

    NaN for classes with no nodes of that type in the accumulated window.
    """
    accuracies: dict[int, float] = {}
    for c in range(num_classes):
        mask = truth == c
        if not mask.any():
            accuracies[c] = float("nan")
        else:
            accuracies[c] = float((preds[mask] == c).mean())
    return accuracies


def node_type_class_label(cls: int) -> str:
    """Map a node-type class index to a human-readable metaclass label."""
    trunk_end = cfg.NUM_TRUNK_NODE_TYPES
    root_end = trunk_end + cfg.NUM_ROOT_NODES
    output_end = root_end + cfg.NUM_OUTPUT_NODES

    if cls < trunk_end:
        return f"trunk_class_{cls}"
    if cls < root_end:
        return f"root_class_{cls - trunk_end}"
    if cls < output_end:
        return f"output_class_{cls - root_end}"
    raise ValueError(f"unknown node type class index: {cls}")


def format_step_report(
    step: int,
    total: float,
    components: dict[str, float],
    decoder_accuracies: dict[int, float],
) -> str:
    """Compact multi-line summary of a training step's losses and accuracies."""

    def fmt_loss_row(prefix: str, label: str) -> str:
        return (
            f"  {label:<10} "
            f"dec_cls={components[f'{prefix}_decoded_classification']:>9.4g}  "
            f"dec_sim={components[f'{prefix}_decoded_similarity']:>9.4g}"
        )

    trunk_end = cfg.NUM_TRUNK_NODE_TYPES
    root_end = trunk_end + cfg.NUM_ROOT_NODES
    output_end = root_end + cfg.NUM_OUTPUT_NODES

    def fmt_acc_group(accuracies: dict[int, float], start: int, end: int) -> str:
        return "[" + ", ".join(
            "  nan" if np.isnan(accuracies[i]) else f"{accuracies[i]:.3f}"
            for i in range(start, end)
        ) + "]"

    def fmt_acc_row(label: str, start: int, end: int) -> str:
        return (
            f"  {label:<10} "
            f"dec={fmt_acc_group(decoder_accuracies, start, end)}"
        )

    return "\n".join(
        [
            f"step={step} total={total:.4g}",
            fmt_loss_row("condenser", "condenser"),
            fmt_loss_row("primary", "primary"),
            fmt_acc_row("acc_trunk", 0, trunk_end),
            fmt_acc_row("acc_root", trunk_end, root_end),
            fmt_acc_row("acc_output", root_end, output_end),
        ]
    )


def cfg_hparams() -> dict[str, bool | int | float | str]:
    """Build an ``add_hparams``-compatible dict from ``config.py``."""
    hparams: dict[str, bool | int | float | str] = {}
    for key, value in vars(cfg).items():
        if key.startswith("_"):
            continue
        if isinstance(value, (bool, int, float, str)):
            hparams[key] = value
        else:
            hparams[key] = str(value)
    return hparams


def log_run_config(writer: SummaryWriter) -> None:
    # Must be logged before scalars so TensorBoard associates them with this run.
    writer.add_hparams(cfg_hparams(), {"hparam/started": 0.0})

    config_text = "\n".join(
        f"{key}={value}"
        for key, value in vars(cfg).items()
        if not key.startswith("_")
    )
    writer.add_text("config", config_text, global_step=0)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    torch.manual_seed(cfg.SEED)

    device = torch.device(cfg.DEVICE)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}-{args.run_name}" if args.run_name else timestamp
    log_dir = f"{cfg.TENSORBOARD_LOG_DIR}/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    log_run_config(writer)
    logging.info("tensorboard_log_dir=%s", log_dir)

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
    logging.info(
        "trainable_parameters=%s (%d)",
        format_param_count(num_trainable_params),
        num_trainable_params,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    try:
        optimizer.zero_grad()
        window_preds: list[np.ndarray] = []
        window_truth: list[np.ndarray] = []
        for step in range(cfg.NUM_STEPS):
            graph = make_random_graph_description(
                num_root_nodes=cfg.NUM_ROOT_NODES,
                num_trunk_nodes=cfg.NUM_TRUNK_NODES,
                num_output_nodes=cfg.NUM_OUTPUT_NODES,
                trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
                num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
            )

            losses = model.training_forward(graph)
            total, components = combine_losses(losses)

            scaled_loss = total / cfg.GRADIENT_ACCUMULATION_STEPS
            scaled_loss.backward()

            if (step + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            step_preds, step_truth = step_preds_and_truth(
                losses.primary_node_predicted_type_logits,
                losses.primary_node_true_types,
            )
            window_preds.append(step_preds)
            window_truth.append(step_truth)

            if step % cfg.LOG_EVERY == 0:
                decoder_accuracies = per_type_accuracies(
                    np.concatenate(window_preds),
                    np.concatenate(window_truth),
                    num_classes=model.num_node_types,
                )
                window_preds.clear()
                window_truth.clear()
                logging.info(
                    "%s",
                    format_step_report(
                        step,
                        total.item(),
                        components,
                        decoder_accuracies,
                    ),
                )

                writer.add_scalar("loss/total", total.item(), step)
                for name, value in components.items():
                    writer.add_scalar(f"loss/{name}", value, step)

                valid_dec = [v for v in decoder_accuracies.values() if not np.isnan(v)]
                if valid_dec:
                    writer.add_scalar(
                        "accuracy/decoder_mean",
                        float(np.mean(valid_dec)),
                        step,
                    )

                for cls, acc in decoder_accuracies.items():
                    if not np.isnan(acc):
                        writer.add_scalar(
                            f"accuracy_per_class/{node_type_class_label(cls)}",
                            acc,
                            step,
                        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
