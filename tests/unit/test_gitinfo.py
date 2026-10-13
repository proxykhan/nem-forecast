from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from nem_forecast.gitinfo import git_sha


def git(cwd: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 - fixed test arguments
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_github_actions_sha_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef")

    assert git_sha() == "0123456789abcdef"


def test_outside_a_repository_it_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(Path.cwd().parent))

    assert git_sha() is None


def test_local_edits_mark_the_sha_dirty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("one")
    git(tmp_path, "add", "f.txt")
    git(tmp_path, "commit", "-qm", "init")

    clean = git_sha()
    (tmp_path / "f.txt").write_text("two")

    assert clean is not None
    assert len(clean) == 40
    assert git_sha() == f"{clean}-dirty"
