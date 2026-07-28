"""Loading a trained autoencoder back out of a run directory.

Model geometry is read from the checkpoint's own state dict rather than from
``config.py``: the config tracks whatever run is current and drifts away from
older checkpoints, so trusting it silently builds the wrong model.
"""

import re
from pathlib import Path

import torch

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.scripts import config as cfg


def resolve_checkpoint(argument: str | Path) -> Path:
    """Accept a .ckpt path or a run directory (best.ckpt, else latest.ckpt)."""
    path = Path(argument)
    if path.is_dir():
        best = path / "best.ckpt"
        resolved = best if best.exists() else path / "latest.ckpt"
    else:
        resolved = path
    if not resolved.exists():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    return resolved


def strip_compile_prefix(state_dict: dict) -> dict:
    """Drop the ``_orig_mod.`` segments ``torch.compile`` inserts into keys.

    ``apply_torch_compile`` replaces ``model.compressor`` and ``model.decoder``
    with compiled wrappers, so a checkpoint saved mid-training carries keys
    like ``compressor._orig_mod.blocks.0...``. Loading those into an
    uncompiled model fails on unexpected keys, and the geometry inference below
    would see zero blocks and silently build a model with no compressor.
    """
    return {key.replace("._orig_mod.", "."): value for key, value in state_dict.items()}


def count_blocks(state_dict: dict, prefix: str) -> int:
    """Number of transformer blocks under ``prefix`` in a state dict."""
    pattern = re.compile(re.escape(prefix) + r"\.blocks\.(\d+)\.")
    indices = {
        int(match.group(1))
        for key in state_dict
        if (match := pattern.match(key)) is not None
    }
    return len(indices)


def build_model_from_checkpoint(
    state_dict: dict,
    device: torch.device,
) -> DagnabbitAutoEncoder:
    """Rebuild the model geometry a checkpoint was trained with.

    Everything the weights determine is read from them. Only the trunk/output
    node counts and the per-type in-degrees come from ``config.py``, because
    the tensors encoding those (position encodings, the pointer candidate mask)
    are non-persistent buffers and never reach the state dict. The in-degree
    list is cross-checked against the number of slot query projections, so a
    mismatch raises rather than loading a subtly wrong model.
    """
    state_dict = strip_compile_prefix(state_dict)
    node_embedding_dim = state_dict["mask_token"].shape[0]
    num_root_nodes = state_dict["root_node_embeddings.weight"].shape[0]
    num_trunk_node_types = state_dict["node_type_predictor.weight"].shape[0]
    num_slot_projs = len(
        {
            key.split(".")[1]
            for key in state_dict
            if key.startswith("pointer_slot_query_projs.")
        }
    )

    in_degrees = cfg.TRUNK_NODE_TYPE_IN_DEGREES
    if isinstance(in_degrees, int):
        in_degrees = [in_degrees] * num_trunk_node_types
    if max([1, *in_degrees]) != num_slot_projs:
        raise ValueError(
            f"checkpoint has {num_slot_projs} pointer slot projections, but "
            f"cfg.TRUNK_NODE_TYPE_IN_DEGREES={cfg.TRUNK_NODE_TYPE_IN_DEGREES} "
            f"implies a maximum in-degree of {max([1, *in_degrees])}"
        )

    model = DagnabbitAutoEncoder(
        node_embedding_dim=node_embedding_dim,
        trunk_node_type_in_degrees=in_degrees,
        num_trunk_node_types=num_trunk_node_types,
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=cfg.NUM_TRUNK_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        mlp_expansion_factor=cfg.MLP_EXPANSION_FACTOR,
        encoder_num_layers=count_blocks(
            state_dict, "node_encoder.sequence_transformer"
        ),
        compressor_num_layers=count_blocks(state_dict, "compressor"),
        decoder_num_layers=count_blocks(state_dict, "decoder"),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model(
    checkpoint: str | Path,
    device: torch.device,
) -> tuple[DagnabbitAutoEncoder, dict]:
    """Resolve, load, and rebuild a model. Returns ``(model, checkpoint_dict)``."""
    path = resolve_checkpoint(checkpoint)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    return build_model_from_checkpoint(loaded["model_state_dict"], device), loaded
