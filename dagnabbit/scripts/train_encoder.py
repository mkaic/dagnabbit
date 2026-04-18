import logging

import torch

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
    ).to(device)

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
            logging.info("step=%d total=%.4f %s", step, total.item(), components)


if __name__ == "__main__":
    main()
