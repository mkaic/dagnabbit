"""Tests for run provenance.

The property that matters is the one that is easiest to lose: provenance is
bookkeeping wrapped around the thing you actually care about, so it must never
be able to take a training run down. Most of these check a *failure* path --
no git, an unreadable config, an unwritable directory -- because those are the
ones that only ever run on the day something is already wrong.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from dagnabbit.scripts import provenance


@pytest.fixture
def fake_config(tmp_path):
    """A stand-in config module: a real file on disk plus public constants."""
    source = '"""Doc."""\nVALUE = 3\nNAME = "x"\n'
    path = tmp_path / "fake_config.py"
    path.write_text(source)
    module = SimpleNamespace(
        __file__=str(path),
        VALUE=3,
        NAME="x",
        _PRIVATE=1,
        lowercase=2,
    )
    return module, source


def test_capture_writes_config_and_provenance(tmp_path, fake_config):
    module, source = fake_config
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    record = provenance.capture(run_directory, module, ["train", "--steps", "1"])

    assert (run_directory / "config.py").read_text() == source
    written = json.loads((run_directory / "provenance.json").read_text())
    assert written["command"] == "train --steps 1"
    assert written["config"]["VALUE"] == "3"
    assert written["config"]["NAME"] == "'x'"
    # Reflection must not sweep up privates or lowercase module-level imports.
    assert "_PRIVATE" not in written["config"]
    assert "lowercase" not in written["config"]
    assert record["config"] == written["config"]
    # The diff never belongs in the json; it goes to its own file or nowhere.
    assert "diff" not in record["git"]


def test_resolved_config_captures_new_names_without_being_told():
    """The point of reflection: a new config knob needs no change here."""
    module = SimpleNamespace(EXISTING=1, BRAND_NEW_KNOB=(1, 2))
    resolved = provenance.resolved_config(module)
    assert resolved == {"EXISTING": "1", "BRAND_NEW_KNOB": "(1, 2)"}


def test_git_state_on_a_real_repository(tmp_path):
    """Commit, branch and cleanliness, against a repository built here."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*arguments):
        subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
        )

    git("init", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "T")
    (repo / "a.txt").write_text("one\n")
    git("add", "a.txt")
    git("commit", "-m", "first")

    clean = provenance.git_state(repo)
    assert clean["available"] and not clean["dirty"]
    assert clean["branch"] == "main"
    assert len(clean["commit"]) == 40
    assert clean["diff"] == ""

    (repo / "a.txt").write_text("two\n")
    dirty = provenance.git_state(repo)
    assert dirty["dirty"]
    assert dirty["commit"] == clean["commit"]
    assert "two" in dirty["diff"]


def test_git_state_outside_a_repository_is_not_an_error(tmp_path):
    assert provenance.git_state(tmp_path) == {"available": False}


def test_capture_survives_a_config_it_cannot_read(tmp_path):
    """An unreadable config must be recorded as such, not raised."""
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    module = SimpleNamespace(__file__=str(tmp_path / "gone.py"), VALUE=1)

    record = provenance.capture(run_directory, module, ["train"])

    assert record["config_file"] is None
    assert "config_error" in record
    # Everything that *could* be captured still was.
    assert record["config"]["VALUE"] == "1"
    assert (run_directory / "provenance.json").exists()
    assert not (run_directory / "config.py").exists()


def test_capture_survives_a_broken_writer(tmp_path, fake_config, capsys):
    """A TensorBoard writer that raises must not reach the caller."""
    module, _ = fake_config
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    class ExplodingWriter:
        def add_text(self, *arguments, **keywords):
            raise RuntimeError("boom")

    record = provenance.capture(
        run_directory, module, ["train"], writer=ExplodingWriter()
    )

    assert record["command"] == "train"
    assert "could not log provenance" in capsys.readouterr().out
    assert (run_directory / "provenance.json").exists()


def test_capture_survives_an_unwritable_directory(tmp_path, fake_config, capsys):
    """A night of training must not be lost to a read-only run directory."""
    module, _ = fake_config
    run_directory = tmp_path / "readonly"
    run_directory.mkdir()
    run_directory.chmod(0o500)
    try:
        record = provenance.capture(run_directory, module, ["train"])
    finally:
        run_directory.chmod(0o700)

    assert record["command"] == "train"
    assert "could not write provenance.json" in capsys.readouterr().out


def test_capture_logs_both_text_tags(tmp_path, fake_config):
    module, source = fake_config
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    logged: dict[str, str] = {}

    class RecordingWriter:
        def add_text(self, tag, body, step):
            logged[tag] = body

    provenance.capture(run_directory, module, ["train"], writer=RecordingWriter())

    assert set(logged) == {"provenance/run", "provenance/config"}
    # Fenced, or TensorBoard's markdown renderer mangles the source.
    assert logged["provenance/config"] == f"```python\n{source}\n```"


def test_format_summary_reports_dirtiness():
    clean = {
        "git": {"available": True, "commit": "a" * 40, "branch": "m", "dirty": False}
    }
    assert "clean" in provenance.format_summary(clean)
    dirty = {
        "git": {"available": True, "commit": "a" * 40, "branch": "m", "dirty": True}
    }
    assert "dirty" in provenance.format_summary(dirty)
    assert "no git" in provenance.format_summary({"git": {"available": False}})


def test_the_real_config_module_is_fully_capturable():
    """The names this repo actually configures must all survive ``repr``."""
    from dagnabbit.scripts import config as cfg

    resolved = provenance.resolved_config(cfg)
    for name in ("GATES", "GEOMETRY", "SAMPLING", "MODEL", "OPTIMIZER_KWARGS"):
        assert name in resolved, name
    assert Path(cfg.__file__).is_file()
