"""What a run directory needs to hold to still mean something in a month.

A TensorBoard events file records what the run *did*. On its own it does not
record what the run *was*: which gate set, which sampling prior, which learning
rate, which code. Two curves in the same board are not comparable unless that is
written down, and reconstructing it from memory a few weeks later does not work.

Three things are captured, once, at startup:

``config.py``
    A verbatim copy of the configuration module's source, so the run carries the
    comments explaining *why* its numbers were chosen and not just the numbers.

``provenance.json``
    The git commit and branch, whether the tree was dirty, the exact command
    line, and the **resolved** value of every public constant in the config
    module. The resolved values are what close the gap the source copy leaves:
    the file is what was on disk, while these are what the run actually used
    after argparse overrides (``--steps``, ``--device``) and any runtime
    mutation. They are collected by reflection over the module's upper-case
    names, so a new config knob is captured without editing anything here.

``uncommitted.diff``
    Written only when the tree was dirty. A commit hash alone is misleading for
    a dirty run -- it names code that is not what ran -- and this is what makes
    such a run reproducible. Note the limit: ``git diff HEAD`` covers tracked
    changes only, so a brand-new untracked file is *listed* in the recorded
    ``git status`` but its contents are not saved.

Nothing here may take the run down. Provenance is bookkeeping, and a missing
``git`` binary or a read-only directory is not a reason to lose a night of
training, so every failure degrades to a warning and a recorded reason.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

# dagnabbit/scripts/provenance.py -> the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# A dirty tree can carry an arbitrarily large diff (a stray notebook output, a
# vendored file). Truncate rather than let provenance dominate the run
# directory; the commit plus the status listing still says what happened.
MAXIMUM_DIFF_BYTES = 4_000_000


def _git(*arguments: str, repo_root: Path = REPO_ROOT) -> str | None:
    """Run a git command, or return None if git cannot answer."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def git_state(repo_root: Path = REPO_ROOT) -> dict:
    """Commit, branch, dirtiness and diff. ``{"available": False}`` if not a repo."""
    commit = _git("rev-parse", "HEAD", repo_root=repo_root)
    if commit is None:
        return {"available": False}

    branch = _git("rev-parse", "--abbrev-ref", "HEAD", repo_root=repo_root)
    status = _git("status", "--porcelain", repo_root=repo_root) or ""
    diff = _git("diff", "HEAD", repo_root=repo_root) or ""
    truncated = len(diff) > MAXIMUM_DIFF_BYTES
    return {
        "available": True,
        "commit": commit.strip(),
        "branch": branch.strip() if branch else None,
        "dirty": bool(status.strip()),
        "status": status,
        "diff": diff[:MAXIMUM_DIFF_BYTES] if truncated else diff,
        "diff_truncated": truncated,
    }


def resolved_config(config_module: ModuleType) -> dict[str, str]:
    """Every public upper-case name in the config module, as ``repr``.

    Reflection rather than a hand-written list: the point of this file is to
    survive changes to the config, and a list of names to capture would be one
    more thing to forget to update. ``repr`` because the values are dataclasses,
    tuples and dtypes -- readable, and not required to round-trip.
    """
    return {
        name: repr(getattr(config_module, name))
        for name in sorted(dir(config_module))
        if name.isupper() and not name.startswith("_")
    }


def capture(
    run_directory: Path,
    config_module: ModuleType,
    command: list[str],
    writer=None,
) -> dict:
    """Write the run's provenance into ``run_directory``. Never raises.

    Also mirrors the config source and the git summary into ``writer`` as
    TensorBoard text, so they are readable next to the curves without going to
    the filesystem. The files are authoritative; the text tab truncates.
    """
    record: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(command),
        "run_directory": str(run_directory),
    }

    try:
        git = git_state()
        diff = git.pop("diff", "")
        record["git"] = git
        if git.get("available") and git.get("dirty") and diff:
            (run_directory / "uncommitted.diff").write_text(diff)
    except Exception as error:  # noqa: BLE001 - provenance must not kill a run
        record["git"] = {"available": False, "error": repr(error)}

    record["config_file"] = None
    try:
        config_path = Path(config_module.__file__)
        source = config_path.read_text()
        (run_directory / "config.py").write_text(source)
    except Exception as error:  # noqa: BLE001
        source = ""
        record["config_error"] = repr(error)
    else:
        # Kept separate from the copy above on purpose. ``relative_to`` raises
        # whenever the config lives outside the repo -- an installed package, a
        # second checkout -- and folding it in there once meant a perfectly
        # good copy still reported itself as unreadable and dropped the
        # TensorBoard text. The path is a label, so degrade to the absolute one.
        try:
            record["config_file"] = str(config_path.relative_to(REPO_ROOT))
        except ValueError:
            record["config_file"] = str(config_path)

    try:
        record["config"] = resolved_config(config_module)
    except Exception as error:  # noqa: BLE001
        record["config_error"] = repr(error)

    try:
        (run_directory / "provenance.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str)
        )
    except Exception as error:  # noqa: BLE001
        print(f"warning: could not write provenance.json: {error!r}")

    if writer is not None:
        try:
            _write_tensorboard_text(writer, record, source)
        except Exception as error:  # noqa: BLE001
            print(f"warning: could not log provenance to TensorBoard: {error!r}")

    return record


def _write_tensorboard_text(writer, record: dict, source: str) -> None:
    git = record.get("git", {})
    if git.get("available"):
        summary = (
            f"commit `{git['commit']}` on `{git['branch']}`"
            f"{' (**dirty** - see uncommitted.diff)' if git['dirty'] else ' (clean)'}"
        )
    else:
        summary = "no git information available"
    writer.add_text(
        "provenance/run",
        f"{summary}\n\nstarted `{record['started_at']}`\n\n"
        f"command: `{record['command']}`",
        0,
    )
    if source:
        writer.add_text("provenance/config", f"```python\n{source}\n```", 0)


def format_summary(record: dict) -> str:
    """One line for stdout at startup."""
    git = record.get("git", {})
    if not git.get("available"):
        return "provenance: no git information"
    state = "dirty" if git.get("dirty") else "clean"
    return f"provenance: {git['commit'][:10]} ({git.get('branch')}, {state})"
