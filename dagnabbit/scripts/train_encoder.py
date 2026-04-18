import logging

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder, TrainingStepLossReturnType
from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.scripts import config as cfg


def combine_losses(
    losses: TrainingStepLossReturnType,
) -> tuple[torch.Tensor, dict[str, float]]:
    cc_mean = torch.stack(losses.condenser_node_classification_losses).mean()
    cs_mean = torch.stack(losses.condenser_node_predicted_embeddings_similarity_losses).mean()
    pc_mean = torch.stack(losses.primary_node_classification_losses).mean()
    ps_mean = torch.stack(losses.primary_node_predicted_embeddings_similarity_losses).mean()

    total = (
        cfg.W_CONDENSER_CLASSIFICATION * cc_mean
        + cfg.W_CONDENSER_SIMILARITY * cs_mean
        + cfg.W_PRIMARY_CLASSIFICATION * pc_mean
        + cfg.W_PRIMARY_SIMILARITY * ps_mean
    )

    components = {
        "condenser_classification": cc_mean.item(),
        "condenser_similarity": cs_mean.item(),
        "primary_classification": pc_mean.item(),
        "primary_similarity": ps_mean.item(),
    }
    return total, components


def format_param_count(n: int) -> str:
    """Format a parameter count as a short human-readable string (e.g. 1.23M)."""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= threshold:
            return f"{n / threshold:.2f}{suffix}"
    return str(n)


def per_type_aucs(
    logits_per_node: list[torch.Tensor],
    true_types: list[int],
    num_classes: int,
) -> dict[int, float]:
    """One-vs-rest ROC-AUC per class. NaN for classes with only one label."""
    probs = torch.softmax(torch.stack(logits_per_node).detach(), dim=-1).cpu().numpy()
    truth = np.asarray(true_types, dtype=np.int64)
    aucs: dict[int, float] = {}
    for c in range(num_classes):
        labels = (truth == c).astype(np.int64)
        if labels.min() == labels.max():
            aucs[c] = float("nan")
        else:
            aucs[c] = float(roc_auc_score(labels, probs[:, c]))
    return aucs


def format_aucs_by_group(aucs: dict[int, float]) -> str:
    trunk_end = cfg.NUM_TRUNK_NODE_TYPES
    root_end = trunk_end + cfg.NUM_ROOT_NODES
    output_end = root_end + cfg.NUM_OUTPUT_NODES

    def fmt_group(start: int, end: int) -> str:
        return ", ".join(
            "nan" if np.isnan(aucs[i]) else f"{aucs[i]:.3f}"
            for i in range(start, end)
        )

    return (
        f"trunk=[{fmt_group(0, trunk_end)}] "
        f"root=[{fmt_group(trunk_end, root_end)}] "
        f"output=[{fmt_group(root_end, output_end)}]"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    torch.manual_seed(cfg.SEED)

    device = torch.device(cfg.DEVICE)

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
            aucs = per_type_aucs(
                losses.primary_node_predicted_type_logits,
                losses.primary_node_true_types,
                num_classes=model.num_node_types,
            )
            logging.info("step=%d total=%.4f %s", step, total.item(), components)
            logging.info("step=%d primary_auc %s", step, format_aucs_by_group(aucs))


if __name__ == "__main__":
    main()
