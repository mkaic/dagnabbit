import subprocess

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
    """Extract flattened argmax predictions and true class ids for one step.

    ``logits_per_node`` may be ``[N, num_types]`` or ``[B, N, num_types]``;
    ``true_types`` is aligned with the leading node dimensions.
    """
    preds = logits_per_node.detach().argmax(dim=-1).reshape(-1).cpu().numpy()
    truth = true_types.detach().reshape(-1).cpu().numpy().astype(np.int64)
    return preds, truth


def step_pointer_stats(
    pointer_logits: torch.Tensor,
    true_positions: torch.Tensor,
    slot_mask: torch.Tensor,
    output_start: int,
    num_root_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-slot pointer correctness for one step, flattened over valid slots.

    ``pointer_logits`` is ``[B, N, S, N]``, ``true_positions`` and
    ``slot_mask`` are ``[B, N, S]``. Returns ``(correct, is_output,
    is_root_parent)`` boolean arrays over valid slots only: ``correct`` marks
    slots whose argmax matches the true parent position; ``is_output`` marks
    slots belonging to output positions (``>= output_start``), the rest
    belong to trunk positions; ``is_root_parent`` marks slots whose *true
    parent* is a root node (canonical position ``< num_root_nodes``).
    """
    predicted = pointer_logits.detach().argmax(dim=-1)
    correct = predicted == true_positions
    num_positions = slot_mask.shape[1]
    is_output = (
        torch.arange(num_positions, device=slot_mask.device)[None, :, None]
        >= output_start
    ).expand_as(slot_mask)
    is_root_parent = true_positions < num_root_nodes

    valid = slot_mask.detach().reshape(-1).cpu().numpy()
    correct_np = correct.reshape(-1).cpu().numpy()[valid]
    is_output_np = is_output.reshape(-1).cpu().numpy()[valid]
    is_root_parent_np = is_root_parent.reshape(-1).cpu().numpy()[valid]
    return correct_np, is_output_np, is_root_parent_np


def pointer_accuracy_summary(
    correct: np.ndarray,
    is_output: np.ndarray,
    is_root_parent: np.ndarray,
) -> tuple[float, dict[NodeSupertype, float], float]:
    """Pointer accuracies over a logging window.

    Unlike :func:`accuracy_summary`, the mean here is the plain fraction of
    valid slots pointed at the right parent (there is no class axis to
    balance). Groups split by consumer supertype: trunk vs output positions.

    The third return is the root-parent accuracy: among slots whose true
    parent is a root node, the fraction that select exactly that root in that
    slot. Root identity is no longer classified directly, but this measures
    the same recoverable information through the pointer head; it is logged
    as ``accuracy/root`` to continue the old curve's role. NaN when the
    window contains no root-parented slots.
    """
    mean = float(correct.mean()) if correct.size else float("nan")
    by_supertype: dict[NodeSupertype, float] = {}
    trunk_mask = ~is_output
    if trunk_mask.any():
        by_supertype[NodeSupertype.TRUNK] = float(correct[trunk_mask].mean())
    if is_output.any():
        by_supertype[NodeSupertype.OUTPUT] = float(correct[is_output].mean())
    root_parent_accuracy = (
        float(correct[is_root_parent].mean())
        if is_root_parent.any()
        else float("nan")
    )
    return mean, by_supertype, root_parent_accuracy


def accuracy_summary(
    preds: np.ndarray,
    truth: np.ndarray,
    num_classes: int,
) -> tuple[float, dict[NodeSupertype, float]]:
    """Per-class recall summary over an accumulated logging window.

    ``accuracy/decoder_mean`` historically logged the mean of per-class recalls,
    not node-weighted overall accuracy. Keep that definition so new runs remain
    comparable with older TensorBoard curves.
    """
    per_class: dict[int, float] = {}
    for cls in range(num_classes):
        mask = truth == cls
        if not mask.any():
            per_class[cls] = float("nan")
        else:
            per_class[cls] = float((preds[mask] == cls).mean())

    valid_class_accuracies = [v for v in per_class.values() if not np.isnan(v)]
    mean = (
        float(np.mean(valid_class_accuracies))
        if valid_class_accuracies
        else float("nan")
    )

    by_supertype: dict[NodeSupertype, float] = {}
    for supertype in NodeSupertype:
        group = [
            accuracy
            for cls, accuracy in per_class.items()
            if subtype_to_supertype(cls) is supertype and not np.isnan(accuracy)
        ]
        if group:
            by_supertype[supertype] = float(np.mean(group))

    return mean, by_supertype


def log_decoder_accuracies(
    writer: SummaryWriter,
    step: int,
    mean_accuracy: float,
    supertype_accuracies: dict[NodeSupertype, float],
    *,
    mean_tag: str,
    tag_prefix: str,
) -> None:
    # Mirror every accuracy under an ``error/`` tag (1 - accuracy) so progress is
    # legible on a log axis as it approaches zero. The accuracy tags all start
    # with ``accuracy``; swap that leading segment for ``error``.
    mean_error_tag = mean_tag.replace("accuracy", "error", 1)
    error_prefix = tag_prefix.replace("accuracy", "error", 1)

    if not np.isnan(mean_accuracy):
        writer.add_scalar(mean_tag, mean_accuracy, step)
        writer.add_scalar(mean_error_tag, 1.0 - mean_accuracy, step)

    for supertype, accuracy in supertype_accuracies.items():
        if not np.isnan(accuracy):
            writer.add_scalar(
                f"{tag_prefix}/{supertype.value}",
                accuracy,
                step,
            )
            writer.add_scalar(
                f"{error_prefix}/{supertype.value}",
                1.0 - accuracy,
                step,
            )


def log_step_metrics(
    writer: SummaryWriter,
    step: int,
    total: float,
    components: dict[str, float],
    decoder_accuracy: float | None = None,
    decoder_supertype_accuracies: dict[NodeSupertype, float] | None = None,
    pointer_accuracy: float | None = None,
    pointer_supertype_accuracies: dict[NodeSupertype, float] | None = None,
    grad_norm: float | None = None,
    grad_was_clipped: bool | None = None,
    learning_rate: float | None = None,
) -> None:
    writer.add_scalar("loss/total", total, step)
    for name, value in components.items():
        writer.add_scalar(f"loss/{name}", value, step)

    if learning_rate is not None:
        writer.add_scalar("optimizer/learning_rate", learning_rate, step)

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

    # Node-type classification accuracy of the reconstructed sequence. Reuses
    # the old scheme's canonical primary-decode tags (accuracy/decoder_mean,
    # accuracy/<supertype>, plus error/ mirrors) so new runs overlay old ones.
    if decoder_accuracy is not None and decoder_supertype_accuracies is not None:
        log_decoder_accuracies(
            writer,
            step,
            decoder_accuracy,
            decoder_supertype_accuracies,
            mean_tag="accuracy/decoder_mean",
            tag_prefix="accuracy",
        )

    # Parent-pointer accuracy is new to the sequence scheme; it gets its own
    # accuracy/pointer namespace (with error/ mirrors via the shared helper).
    if pointer_accuracy is not None and pointer_supertype_accuracies is not None:
        log_decoder_accuracies(
            writer,
            step,
            pointer_accuracy,
            pointer_supertype_accuracies,
            mean_tag="accuracy/pointer/mean",
            tag_prefix="accuracy/pointer",
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


def git_metadata() -> dict[str, str]:
    """Return Git metadata for TensorBoard run provenance."""

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        commit = _git("rev-parse", "HEAD")
        short_commit = _git("rev-parse", "--short", "HEAD")
        branch = _git("branch", "--show-current") or "DETACHED"
        dirty = bool(_git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "git_commit": "unknown",
            "git_commit_short": "unknown",
            "git_branch": "unknown",
            "git_dirty": "unknown",
        }

    return {
        "git_commit": commit,
        "git_commit_short": short_commit,
        "git_branch": branch,
        "git_dirty": str(dirty),
    }


def log_run_config(writer: SummaryWriter) -> None:
    metadata = git_metadata()
    hparams = cfg_hparams()
    hparams.update(metadata)

    # Log hparams on this writer (add_hparams opens a nested SummaryWriter subdir).
    exp, ssi, sei = hparams_summary(hparams, {"hparam/started": 0.0})
    writer.file_writer.add_summary(exp, 0)
    writer.file_writer.add_summary(ssi, 0)
    writer.file_writer.add_summary(sei, 0)

    config_items = [
        (key, value) for key, value in vars(cfg).items() if not key.startswith("_")
    ]
    config_items.extend(metadata.items())
    config_text = "\n".join(f"{key}={value}" for key, value in config_items)
    writer.add_text("config", config_text, global_step=0)
