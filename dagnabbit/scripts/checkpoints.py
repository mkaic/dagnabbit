"""Loading Phase 0 simulator checkpoints, shared by every downstream script."""

from pathlib import Path

import torch

from dagnabbit.dag.model import GraphSimulator


def load_simulator(
    path: Path, device: torch.device, num_trunk_node_types: int
) -> tuple[GraphSimulator, int]:
    """The saved model on ``device``, and the step it was saved at.

    ``num_trunk_node_types`` is the configured gate set's size, checked against
    the checkpoint's geometry because the trunk type ids are positional: a
    mismatch would silently make every gate mean something else.
    """
    # weights_only=False: the checkpoint carries the Geometry / SimulatorConfig
    # dataclasses alongside the tensors, and it is our own trusted artifact.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    geometry = checkpoint["geometry"]
    if geometry.num_trunk_node_types != num_trunk_node_types:
        raise ValueError(
            f"checkpoint was trained with {geometry.num_trunk_node_types} trunk "
            f"types but the configured gate set has {num_trunk_node_types}; the "
            "type ids would not mean the same thing"
        )
    model = GraphSimulator(geometry, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model, checkpoint["step"]
