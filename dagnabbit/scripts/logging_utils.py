import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams as hparams_summary

from dagnabbit.dag.description import NodeSupertype, subtype_to_supertype
from dagnabbit.scripts import config as cfg


def format_param_count(n: int) -> str:
    """Format a parameter count as a short human-readable string (e.g. 1.23M)."""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= threshold:
            return f"{n / threshold:.2f}{suffix}"
    return str(n)


def step_preds_and_truth(
    logits_per_node: torch.Tensor,
    true_types: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract argmax predictions and true class ids for one step.

    ``logits_per_node`` is ``[N, num_types]`` and ``true_types`` is a 1-D label
    tensor aligned with it.
    """
    preds = logits_per_node.detach().argmax(dim=-1).cpu().numpy()
    truth = true_types.detach().cpu().numpy().astype(np.int64)
    return preds, truth


def accuracy_summary(
    preds: np.ndarray,
    truth: np.ndarray,
    num_classes: int,
) -> tuple[float, dict[NodeSupertype, float]]:
    """Overall and supertype accuracy over an accumulated logging window."""
    correct = preds == truth
    overall = float(correct.mean()) if truth.size else float("nan")

    class_supertypes = np.array(
        [subtype_to_supertype(cls) for cls in range(num_classes)],
        dtype=object,
    )
    truth_supertypes = class_supertypes[truth]
    by_supertype: dict[NodeSupertype, float] = {}
    for supertype in NodeSupertype:
        mask = truth_supertypes == supertype
        if mask.any():
            by_supertype[supertype] = float(correct[mask].mean())
    return overall, by_supertype


def log_decoder_accuracies(
    writer: SummaryWriter,
    step: int,
    overall_accuracy: float,
    supertype_accuracies: dict[NodeSupertype, float],
    *,
    mean_tag: str,
    tag_prefix: str,
) -> None:
    if not np.isnan(overall_accuracy):
        writer.add_scalar(mean_tag, overall_accuracy, step)

    for supertype, accuracy in supertype_accuracies.items():
        if not np.isnan(accuracy):
            writer.add_scalar(
                f"{tag_prefix}/{supertype.value}",
                accuracy,
                step,
            )


def log_step_metrics(
    writer: SummaryWriter,
    step: int,
    total: float,
    components: dict[str, float],
    decoder_accuracy: float,
    decoder_supertype_accuracies: dict[NodeSupertype, float],
    tf_decoder_accuracy: float | None = None,
    tf_decoder_supertype_accuracies: dict[NodeSupertype, float] | None = None,
    grad_norm: float | None = None,
    grad_was_clipped: bool | None = None,
) -> None:
    writer.add_scalar("loss/total", total, step)
    for name, value in components.items():
        writer.add_scalar(f"loss/{name}", value, step)

    if grad_norm is not None:
        writer.add_scalar("gradients/norm", grad_norm, step)
        if grad_was_clipped is not None:
            writer.add_scalar("gradients/was_clipped", float(grad_was_clipped), step)
            if cfg.GRADIENT_CLIP_MAX_NORM is not None:
                writer.add_scalar(
                    "gradients/norm_ratio",
                    grad_norm / cfg.GRADIENT_CLIP_MAX_NORM,
                    step,
                )

    log_decoder_accuracies(
        writer,
        step,
        decoder_accuracy,
        decoder_supertype_accuracies,
        mean_tag="accuracy/decoder_mean",
        tag_prefix="accuracy",
    )

    # Teacher-forced decode accuracies (logged under a parallel ``tf`` namespace
    # so they sit next to the autoregressive curves in TensorBoard).
    if tf_decoder_accuracy is not None and tf_decoder_supertype_accuracies is not None:
        log_decoder_accuracies(
            writer,
            step,
            tf_decoder_accuracy,
            tf_decoder_supertype_accuracies,
            mean_tag="accuracy/tf/decoder_mean",
            tag_prefix="accuracy/tf",
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
    exp, ssi, sei = hparams_summary(cfg_hparams(), {"hparam/started": 0.0})
    writer.file_writer.add_summary(exp, 0)
    writer.file_writer.add_summary(ssi, 0)
    writer.file_writer.add_summary(sei, 0)

    config_text = "\n".join(
        f"{key}={value}" for key, value in vars(cfg).items() if not key.startswith("_")
    )
    writer.add_text("config", config_text, global_step=0)
