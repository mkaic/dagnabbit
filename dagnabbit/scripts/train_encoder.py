import logging

import mlflow
import numpy as np
import torch

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
    cec_mean = torch.stack(losses.condenser_node_encoded_classification_losses).mean()
    pec_mean = torch.stack(losses.primary_node_encoded_classification_losses).mean()

    total = (
        cfg.W_CONDENSER_DECODED_CLASSIFICATION * cc_mean
        + cfg.W_CONDENSER_DECODED_SIMILARITY * cs_mean
        + cfg.W_PRIMARY_DECODED_CLASSIFICATION * pc_mean
        + cfg.W_PRIMARY_DECODED_SIMILARITY * ps_mean
        + cfg.W_CONDENSER_ENCODED_CLASSIFICATION * cec_mean
        + cfg.W_PRIMARY_ENCODED_CLASSIFICATION * pec_mean
    )

    components = {
        "condenser_decoded_classification": cc_mean.item(),
        "condenser_decoded_similarity": cs_mean.item(),
        "condenser_encoded_classification": cec_mean.item(),
        "primary_decoded_classification": pc_mean.item(),
        "primary_decoded_similarity": ps_mean.item(),
        "primary_encoded_classification": pec_mean.item(),
    }
    return total, components


def format_param_count(n: int) -> str:
    """Format a parameter count as a short human-readable string (e.g. 1.23M)."""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= threshold:
            return f"{n / threshold:.2f}{suffix}"
    return str(n)


def per_type_accuracies(
    logits_per_node: list[torch.Tensor],
    true_types: list[int],
    num_classes: int,
) -> dict[int, float]:
    """Per-class accuracy (recall): argmax==c rate over nodes whose true type is c.

    NaN for classes with no nodes of that type in the batch.
    """
    preds = torch.stack(logits_per_node).detach().argmax(dim=-1).cpu().numpy()
    truth = np.asarray(true_types, dtype=np.int64)
    accuracies: dict[int, float] = {}
    for c in range(num_classes):
        mask = truth == c
        if not mask.any():
            accuracies[c] = float("nan")
        else:
            accuracies[c] = float((preds[mask] == c).mean())
    return accuracies


def format_step_report(
    step: int,
    total: float,
    components: dict[str, float],
    encoder_accuracies: dict[int, float],
    decoder_accuracies: dict[int, float],
) -> str:
    """Compact multi-line summary of a training step's losses and accuracies."""

    def fmt_loss_row(prefix: str, label: str) -> str:
        return (
            f"  {label:<10} "
            f"dec_cls={components[f'{prefix}_decoded_classification']:>9.4g}  "
            f"dec_sim={components[f'{prefix}_decoded_similarity']:>9.4g}  "
            f"enc_cls={components[f'{prefix}_encoded_classification']:>9.4g}"
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
            f"enc={fmt_acc_group(encoder_accuracies, start, end)}  "
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    torch.manual_seed(cfg.SEED)

    device = torch.device(cfg.DEVICE)

    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)

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

    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in vars(cfg).items() if not k.startswith("_")})

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

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            if step % cfg.LOG_EVERY == 0:
                decoder_accuracies = per_type_accuracies(
                    losses.primary_node_predicted_type_logits,
                    losses.primary_node_true_types,
                    num_classes=model.num_node_types,
                )
                encoder_accuracies = per_type_accuracies(
                    losses.primary_node_encoded_type_logits,
                    losses.primary_node_true_types,
                    num_classes=model.num_node_types,
                )
                logging.info(
                    "%s",
                    format_step_report(
                        step,
                        total.item(),
                        components,
                        encoder_accuracies,
                        decoder_accuracies,
                    ),
                )

                metrics: dict[str, float] = {"loss/total": total.item()}
                metrics.update({f"loss/{name}": v for name, v in components.items()})

                valid_enc = [v for v in encoder_accuracies.values() if not np.isnan(v)]
                valid_dec = [v for v in decoder_accuracies.values() if not np.isnan(v)]
                if valid_enc:
                    metrics["accuracy/encoder_mean"] = float(np.mean(valid_enc))
                if valid_dec:
                    metrics["accuracy/decoder_mean"] = float(np.mean(valid_dec))

                metrics.update({
                    f"accuracy_per_class/encoder_class_{cls}": acc
                    for cls, acc in encoder_accuracies.items()
                    if not np.isnan(acc)
                })
                metrics.update({
                    f"accuracy_per_class/decoder_class_{cls}": acc
                    for cls, acc in decoder_accuracies.items()
                    if not np.isnan(acc)
                })

                mlflow.log_metrics(metrics, step=step)


if __name__ == "__main__":
    main()
