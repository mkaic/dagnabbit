import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams as hparams_summary

from dagnabbit.scripts import config as cfg


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


def log_decoder_accuracies(
    writer: SummaryWriter,
    step: int,
    decoder_accuracies: dict[int, float],
    *,
    mean_tag: str,
    tag_prefix: str,
    per_class_tag_prefix: str,
) -> None:
    valid_dec = [v for v in decoder_accuracies.values() if not np.isnan(v)]
    if valid_dec:
        writer.add_scalar(mean_tag, float(np.mean(valid_dec)), step)

    trunk_end = cfg.NUM_TRUNK_NODE_TYPES
    root_end = trunk_end + cfg.NUM_ROOT_NODES
    output_end = root_end + cfg.NUM_OUTPUT_NODES
    for group_name, start, end in (
        ("trunk", 0, trunk_end),
        ("root", trunk_end, root_end),
        ("output", root_end, output_end),
    ):
        group_vals = [
            decoder_accuracies[i]
            for i in range(start, end)
            if not np.isnan(decoder_accuracies[i])
        ]
        if group_vals:
            writer.add_scalar(
                f"{tag_prefix}/{group_name}",
                float(np.mean(group_vals)),
                step,
            )

    for cls, acc in decoder_accuracies.items():
        if not np.isnan(acc):
            writer.add_scalar(
                f"{per_class_tag_prefix}/{node_type_class_label(cls)}",
                acc,
                step,
            )


def log_step_metrics(
    writer: SummaryWriter,
    step: int,
    total: float,
    components: dict[str, float],
    decoder_accuracies: dict[int, float],
    condenser_decoder_accuracies: dict[int, float],
) -> None:
    writer.add_scalar("loss/total", total, step)
    for name, value in components.items():
        writer.add_scalar(f"loss/{name}", value, step)

    log_decoder_accuracies(
        writer,
        step,
        decoder_accuracies,
        mean_tag="accuracy/decoder_mean",
        tag_prefix="accuracy",
        per_class_tag_prefix="accuracy_per_class",
    )
    log_decoder_accuracies(
        writer,
        step,
        condenser_decoder_accuracies,
        mean_tag="accuracy/condenser/mean",
        tag_prefix="accuracy/condenser",
        per_class_tag_prefix="accuracy/condenser_per_class",
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
    # Log hparams on this writer (add_hparams opens a nested SummaryWriter subdir).
    exp, ssi, sei = hparams_summary(
        cfg_hparams(), {"hparam/started": 0.0}
    )
    writer.file_writer.add_summary(exp, 0)
    writer.file_writer.add_summary(ssi, 0)
    writer.file_writer.add_summary(sei, 0)

    config_text = "\n".join(
        f"{key}={value}"
        for key, value in vars(cfg).items()
        if not key.startswith("_")
    )
    writer.add_text("config", config_text, global_step=0)
