from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from nem_forecast import __version__
from nem_forecast.cli import app

runner = CliRunner()


def test_help_lists_every_pipeline_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "migrate",
        "ingest",
        "backfill",
        "build-features",
        "backtest",
        "train",
        "serve",
        "monitor",
    ):
        assert command in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.parametrize(
    ("command", "phase"),
    [
        ("monitor", 6),
    ],
)
def test_stub_commands_are_not_implemented_yet(command: str, phase: int) -> None:
    result = runner.invoke(app, [command])

    assert isinstance(result.exception, NotImplementedError)
    assert f"Phase {phase}" in str(result.exception)


def test_migrate_applies_then_reports_up_to_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_widgets.sql").write_text("CREATE TABLE widgets (id INTEGER PRIMARY KEY)")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'cli.db'}")
    args = ["migrate", "--migrations-dir", str(migrations)]

    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == 0, first.output
    assert "Applied 001_widgets.sql" in first.output
    assert second.exit_code == 0, second.output
    assert "Database is up to date." in second.output


def test_serve_starts_the_api_on_the_requested_port(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("nem_forecast.serve.__main__.run", lambda **kwargs: calls.append(kwargs))

    result = runner.invoke(app, ["serve", "--port", "9000"])

    assert result.exit_code == 0, result.output
    assert calls == [{"host": "0.0.0.0", "port": 9000}]  # noqa: S104 - container interface
