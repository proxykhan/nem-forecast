"""``nem-forecast`` command-line entrypoint.

Pipeline commands are stubs until their phase lands; each raises NotImplementedError naming it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from nem_forecast import __version__
from nem_forecast.config import get_settings
from nem_forecast.db import MIGRATIONS_DIR, apply_migrations, create_db_engine

app = typer.Typer(
    name="nem-forecast",
    help="Forecast VIC1 electricity demand from AEMO NEM data.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # locals can hold secrets from Settings
)


def _not_yet(phase: int, what: str) -> NotImplementedError:
    return NotImplementedError(f"{what} lands in Phase {phase}")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"nem-forecast {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    """Forecast VIC1 electricity demand from AEMO NEM data."""


@app.command()
def migrate(
    migrations_dir: Annotated[
        Path, typer.Option(help="Directory of numbered .sql files.")
    ] = MIGRATIONS_DIR,
) -> None:
    """Apply pending SQL migrations to DATABASE_URL."""
    engine = create_db_engine(get_settings().database_url.get_secret_value())
    try:
        applied = apply_migrations(engine, migrations_dir)
    finally:
        engine.dispose()
    if not applied:
        typer.echo("Database is up to date.")
    for migration in applied:
        typer.echo(f"Applied {migration.path.name}")


@app.command()
def ingest() -> None:
    """Ingest recent AEMO dispatch data into raw_dispatch."""
    raise _not_yet(1, "Ingestion")


@app.command()
def build_features() -> None:
    """Build leakage-safe features into the features table."""
    raise _not_yet(2, "Feature building")


@app.command()
def backtest() -> None:
    """Run the walk-forward backtest for baselines and models."""
    raise _not_yet(2, "Backtesting")


@app.command()
def train() -> None:
    """Train, track and register forecasting models."""
    raise _not_yet(3, "Training")


@app.command()
def serve() -> None:
    """Start the FastAPI inference service."""
    raise _not_yet(4, "Serving")


@app.command()
def monitor() -> None:
    """Score served forecasts and check for drift."""
    raise _not_yet(6, "Monitoring")


if __name__ == "__main__":
    app()
