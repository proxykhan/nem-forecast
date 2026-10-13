"""Which code produced a result: the git commit, marked ``-dirty`` if there were local edits."""

from __future__ import annotations

import os
import subprocess


def git_sha() -> str | None:
    """The current commit, from GITHUB_SHA in Actions or from git locally; None if unknown."""
    if sha := os.getenv("GITHUB_SHA"):
        return sha
    try:
        head = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{head}-dirty" if dirty else head


def _git(*args: str) -> str:
    result = subprocess.run(  # noqa: S603 - fixed arguments, no user input
        ["git", *args],  # noqa: S607 - git is resolved from PATH on purpose
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return result.stdout.strip()
