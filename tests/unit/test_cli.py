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
        "score",
        "monitor",
        "calibrate-drift",
        "retrain",
        "grant-reader",
    ):
        assert command in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_monitoring_commands_need_the_served_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    for command in ("monitor", "retrain"):
        result = runner.invoke(app, [command])

        assert result.exit_code == 2
        assert "MODEL_REPO_ID" in result.output


def test_issues_need_a_github_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("nem_forecast.cli._served_model", lambda: None)

    result = runner.invoke(app, ["monitor", "--open-issues"])

    assert result.exit_code == 2
    assert "GITHUB_TOKEN" in result.output


def test_grant_reader_rejects_odd_role_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite://")

    result = runner.invoke(app, ["grant-reader", "--role", "x; DROP TABLE y"])

    assert result.exit_code == 2


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
