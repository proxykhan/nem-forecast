"""``nem-forecast`` command-line entrypoint.

Commands for later phases are stubs that raise NotImplementedError naming their phase.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import Engine

from nem_forecast import __version__
from nem_forecast.config import get_settings
from nem_forecast.db import MIGRATIONS_DIR, apply_migrations, create_db_engine
from nem_forecast.ingest.backfill import MonthStatus, default_months
from nem_forecast.ingest.backfill import backfill as run_backfill
from nem_forecast.ingest.loader import coverage
from nem_forecast.ingest.nemweb import DEFAULT_CACHE_DIR, NemwebClient
from nem_forecast.ingest.recent import ingest_dispatch_is
from nem_forecast.schemas import NEM_TZ, Region

app = typer.Typer(
    name="nem-forecast",
    help="Forecast VIC1 electricity demand from AEMO NEM data.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # locals can hold secrets from Settings
)


_TIME_FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"]
CacheDir = Annotated[Path, typer.Option(help="Local response cache for downloaded AEMO files.")]


def _not_yet(phase: int, what: str) -> NotImplementedError:
    return NotImplementedError(f"{what} lands in Phase {phase}")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"nem-forecast {__version__}")
        raise typer.Exit


@contextmanager
def _database() -> Iterator[Engine]:
    engine = create_db_engine(get_settings().database_url.get_secret_value())
    try:
        yield engine
    finally:
        engine.dispose()


def _market_time(value: datetime) -> datetime:
    """CLI times are AEMO market time (UTC+10) unless they carry an explicit offset."""
    return value if value.tzinfo else value.replace(tzinfo=NEM_TZ)


def _echo_coverage(engine: Engine, region: Region) -> None:
    rows, first, last = coverage(engine, region)
    if first is None or last is None:
        typer.echo(f"raw_dispatch {region}: no rows")
        return
    typer.echo(
        f"raw_dispatch {region}: {rows:,} rows, "
        f"{first.astimezone(NEM_TZ):%Y-%m-%d %H:%M} to {last.astimezone(NEM_TZ):%Y-%m-%d %H:%M} "
        "(market time)"
    )


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Forecast VIC1 electricity demand from AEMO NEM data."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


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
def ingest(
    start: Annotated[
        datetime | None,
        typer.Option(
            formats=_TIME_FORMATS,
            help="Exclusive start, market time (UTC+10). Default: --hours before --end.",
        ),
    ] = None,
    end: Annotated[
        datetime | None,
        typer.Option(formats=_TIME_FORMATS, help="Inclusive end, market time. Default: now."),
    ] = None,
    hours: Annotated[
        int, typer.Option(min=1, max=72, help="Window length when --start is not given.")
    ] = 24,
    region: Annotated[
        list[Region] | None, typer.Option(help="Region to keep; repeat for several. Default: VIC1.")
    ] = None,
    cache_dir: CacheDir = DEFAULT_CACHE_DIR,
) -> None:
    """Ingest recent 5-minute dispatch data (NEMWeb DispatchIS) into raw_dispatch."""
    end_at = _market_time(end) if end else datetime.now(NEM_TZ)
    start_at = _market_time(start) if start else end_at - timedelta(hours=hours)
    regions = region or [Region.VIC1]

    with _database() as engine, NemwebClient(cache_dir=cache_dir) as client:
        report = ingest_dispatch_is(engine, client, start=start_at, end=end_at, regions=regions)
        typer.echo(
            f"{report.files} files ({report.files_failed} failed), {report.records} records: "
            f"{report.load.inserted} inserted, {report.load.updated} revised, "
            f"{report.load.unchanged} unchanged; {report.skipped_rows} rows skipped, "
            f"{report.missing_intervals} intervals missing"
        )
        for each in regions:
            _echo_coverage(engine, each)
    if report.files == 0 or report.files_failed == report.files:
        raise typer.Exit(code=1)


@app.command()
def backfill(
    region: Annotated[Region, typer.Option(help="Region to backfill.")] = Region.VIC1,
    years: Annotated[
        int, typer.Option(min=1, max=25, help="Years of history when --from is not given.")
    ] = 3,
    from_month: Annotated[
        datetime | None, typer.Option("--from", formats=["%Y-%m"], help="First month, YYYY-MM.")
    ] = None,
    to_month: Annotated[
        datetime | None,
        typer.Option("--to", formats=["%Y-%m"], help="Last month, YYYY-MM. Default: this month."),
    ] = None,
    force: Annotated[bool, typer.Option(help="Reload months that are already complete.")] = False,
    cache_dir: CacheDir = DEFAULT_CACHE_DIR,
) -> None:
    """Backfill history month by month from AEMO's price-and-demand files. Resumable."""
    first, last = default_months(datetime.now(NEM_TZ).date(), years)
    first = from_month.date() if from_month else first
    last = to_month.date() if to_month else last

    with _database() as engine, NemwebClient(cache_dir=cache_dir) as client:
        outcomes = run_backfill(
            engine, client, region=region, first_month=first, last_month=last, force=force
        )
        for outcome in outcomes:
            detail = outcome.error or ""
            if outcome.load is not None:
                detail = (
                    f"{outcome.records} records: {outcome.load.inserted} inserted, "
                    f"{outcome.load.updated} revised, {outcome.load.unchanged} unchanged, "
                    f"{outcome.missing_intervals} missing"
                )
            typer.echo(f"{outcome.month:%Y-%m}  {outcome.status:<11}  {detail}")
        _echo_coverage(engine, region)
    if any(o.status is MonthStatus.FAILED for o in outcomes):
        raise typer.Exit(code=1)


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
