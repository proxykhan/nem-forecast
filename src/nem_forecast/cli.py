"""``nem-forecast`` command-line entrypoint.

Commands for later phases are stubs that raise NotImplementedError naming their phase.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import mlflow
import polars as pl
import typer
from huggingface_hub import HfApi
from sqlalchemy import Engine, text

from nem_forecast import __version__
from nem_forecast.backtest.baselines import BASELINES, Forecaster
from nem_forecast.backtest.harness import BacktestConfig, plan_folds, run_backtest, save_results
from nem_forecast.backtest.metrics import ALL_BUCKETS, ALL_HORIZONS
from nem_forecast.backtest.report import CHART_NAME, verdict, write_report
from nem_forecast.config import get_settings
from nem_forecast.db import MIGRATIONS_DIR, apply_migrations, create_db_engine
from nem_forecast.features.build import FEATURE_VERSION, one_step_features, write_features
from nem_forecast.features.series import load_dispatch, span, to_half_hourly
from nem_forecast.gitinfo import git_sha
from nem_forecast.ingest.backfill import MonthStatus, default_months
from nem_forecast.ingest.backfill import backfill as run_backfill
from nem_forecast.ingest.loader import coverage
from nem_forecast.ingest.nemweb import DEFAULT_CACHE_DIR, NemwebClient
from nem_forecast.ingest.recent import ingest_dispatch_is
from nem_forecast.models.base import METADATA_FILE, DirectModel
from nem_forecast.models.catalog import MODELS
from nem_forecast.models.lgbm import LightGBMDirect, LightGBMQuantile
from nem_forecast.models.registry import (
    CARD_FILES,
    latest_backtest,
    push_to_hub,
    render_model_card,
    repo_name,
)
from nem_forecast.models.ridge import RidgeForecaster
from nem_forecast.models.tuning import (
    MAX_TRIALS,
    VALIDATION_DAYS,
    load_params,
    params_path,
    save_params,
    tune_lightgbm,
    validation_split,
)
from nem_forecast.schemas import NEM_TZ, Region
from nem_forecast.tracking.client import configure_mlflow, experiment_id
from nem_forecast.tracking.runs import MlflowBacktestObserver, MlflowTuningLogger, model_params

log = logging.getLogger(__name__)

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
    for noisy in ("httpx", "mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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
def build_features(
    region: Annotated[Region, typer.Option(help="Region to build features for.")] = Region.VIC1,
) -> None:
    """Build leakage-checked horizon-1 features into the features table."""
    with _database() as engine:
        frame = one_step_features(to_half_hourly(load_dispatch(engine, region)))
        rows = write_features(engine, region, frame)
    typer.echo(f"Wrote {rows:,} {FEATURE_VERSION} feature rows for {region}.")


def _forecasters(region: Region, *, with_models: bool) -> list[Forecaster]:
    forecasters: list[Forecaster] = [baseline() for baseline in BASELINES]
    if with_models:
        params = load_params(params_path(region))
        if params is None:
            log.warning("no tuned parameters at %s; using defaults", params_path(region))
        forecasters += [RidgeForecaster(), LightGBMDirect(params), LightGBMQuantile(params)]
    return forecasters


def _make_model(name: str, region: Region) -> DirectModel:
    if name not in MODELS:
        raise typer.BadParameter(f"choose one of {', '.join(MODELS)}", param_hint="--model")
    cls = MODELS[name]
    if issubclass(cls, LightGBMDirect | LightGBMQuantile):
        return cls(load_params(params_path(region)))
    return cls()


@app.command()
def tune(
    region: Annotated[Region, typer.Option(help="Region to tune for.")] = Region.VIC1,
    trials: Annotated[int, typer.Option(min=1, max=MAX_TRIALS, help="Optuna trials.")] = MAX_TRIALS,
    validation_days: Annotated[
        int, typer.Option(min=7, help="Validation length, ending where the backtest starts.")
    ] = VALIDATION_DAYS,
    test_days: Annotated[int, typer.Option(min=1, help="Backtest period to stay clear of.")] = 365,
) -> None:
    """Tune LightGBM with Optuna on a validation split before the backtest period."""
    configure_mlflow(get_settings())
    sha = git_sha()
    with _database() as engine:
        series = to_half_hourly(load_dispatch(engine, region))
    split = validation_split(series, BacktestConfig(test_days=test_days), validation_days)
    with MlflowTuningLogger(region=region, split=split, git_sha=sha, trials=trials) as logger:
        result = tune_lightgbm(series, split, trials=trials, on_trial=logger)
        logger.finish(result)
    path = params_path(region)
    save_params(path, result, git_sha=sha)
    typer.echo(
        f"Validation {split.train_end:%Y-%m-%d} to {split.valid_end:%Y-%m-%d}: best MAE "
        f"{result.best_mae:.1f} MW after {len(result.trials)} trials."
    )
    typer.echo(f"Parameters: {path}")


@app.command()
def backtest(
    region: Annotated[Region, typer.Option(help="Region to backtest.")] = Region.VIC1,
    test_days: Annotated[
        int, typer.Option(min=1, help="Length of the walk-forward evaluation period.")
    ] = 365,
    refit_every_days: Annotated[int, typer.Option(min=1, help="Days between refits.")] = 30,
    models: Annotated[
        bool, typer.Option(help="Include Ridge and LightGBM, or run the baselines only.")
    ] = True,
    report_dir: Annotated[Path, typer.Option(help="Where the markdown report goes.")] = Path(
        "docs/backtest"
    ),
    save: Annotated[bool, typer.Option(help="Record runs and metrics in model_runs.")] = True,
    track: Annotated[bool, typer.Option(help="Log runs to MLflow.")] = True,
) -> None:
    """Walk-forward backtest of the baselines and models; writes model_runs and a report."""
    started = datetime.now(UTC)
    config = BacktestConfig(test_days=test_days, refit_every_days=refit_every_days)
    sha = git_sha()
    if track:
        configure_mlflow(get_settings())
    forecasters = _forecasters(region, with_models=models)
    with _database() as engine:
        series = to_half_hourly(load_dispatch(engine, region))
        observer = (
            MlflowBacktestObserver(
                region=region,
                config=config,
                git_sha=sha,
                data_span=span(series),
                eval_start=plan_folds(series, config)[0].cutoff,
            )
            if track
            else None
        )
        result = run_backtest(series, forecasters, config, observer=observer)
        mlflow_runs = observer.parent_runs if observer else None
        run_ids = (
            save_results(
                engine,
                result,
                region=region,
                git_sha=sha,
                started_at=started,
                mlflow_run_ids=mlflow_runs,
                params={f.name: model_params(f) for f in forecasters},
            )
            if save
            else None
        )
    path = write_report(
        result,
        report_dir,
        region=region,
        git_sha=sha,
        generated_at=started,
        run_ids=run_ids,
        mlflow_runs=mlflow_runs,
    )
    if observer:
        observer.log_report([path, path.parent / CHART_NAME])

    overall = result.metrics.filter(
        pl.col("horizon") == ALL_HORIZONS, pl.col("tod_bucket") == ALL_BUCKETS
    ).pivot(on="metric", index="model", values="value")
    typer.echo(f"{'model':<24}{'MAE':>8}{'RMSE':>8}{'MAPE%':>8}{'pinball':>9}{'cover%':>8}")
    for row in overall.iter_rows(named=True):
        typer.echo(
            f"{row['model']:<24}{row['mae']:>8.0f}{row['rmse']:>8.0f}{row['mape']:>8.1f}"
            f"{row['pinball_avg']:>9.1f}{row['coverage_80']:>8.1f}"
        )
    for name, kind in result.kinds.items():
        if kind == "model":
            typer.echo(f"{name}: {verdict(result, name).replace('**', '')}")
    typer.echo(f"Report: {path}")


@app.command()
def train(
    model: Annotated[str, typer.Option(help=f"One of: {', '.join(MODELS)}.")] = "lightgbm_quantile",
    region: Annotated[Region, typer.Option(help="Region to train for.")] = Region.VIC1,
    output_dir: Annotated[Path, typer.Option(help="Where the trained model is saved.")] = Path(
        "artifacts"
    ),
    promote: Annotated[
        bool, typer.Option(help="Push the model and its card to Hugging Face Hub.")
    ] = False,
) -> None:
    """Train a model on all data so far; optionally promote it to Hugging Face Hub."""
    settings = get_settings()
    sha = git_sha()
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    if promote and (not sha or sha.endswith("-dirty")):
        raise typer.BadParameter("commit your changes first: versions are git SHAs")
    if promote and token is None:
        raise typer.BadParameter("HF_TOKEN is not set")
    forecaster = _make_model(model, region)
    configure_mlflow(settings)
    started = datetime.now(UTC)
    repo_id = f"{HfApi(token=token).whoami()['name']}/{repo_name(region)}" if token else None
    with _database() as engine:
        series = to_half_hourly(load_dispatch(engine, region))
        forecaster.fit(series)
        directory = output_dir / f"{region.value.lower()}-{model}"
        forecaster.save(directory, version=sha or "unknown")
        metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        card = render_model_card(
            metadata,
            latest_backtest(engine, region),
            region=region,
            repo_id=repo_id or f"<user>/{repo_name(region)}",
        )
        for name in CARD_FILES:
            (directory / name).write_text(card, encoding="utf-8", newline="\n")

        url = None
        with mlflow.start_run(
            experiment_id=experiment_id(f"{region.value} training"),
            run_name=f"{model} final",
            tags={"git_sha": sha or "unknown", "region": region.value},
        ) as run:
            mlflow.log_params(
                model_params(forecaster)
                | {
                    "features": ",".join(metadata["features"]),
                    "data_start": metadata["trained_on"][0],
                    "data_end": metadata["trained_on"][1],
                }
            )
            mlflow.log_artifact(str(directory / METADATA_FILE))
            mlflow.log_artifact(str(directory / CARD_FILES[0]))
            if promote and token is not None and sha is not None:
                url = push_to_hub(directory, token=token, region=region, version=sha)
                mlflow.set_tag("hf_model_url", url)
        _record_training(engine, metadata, region, sha, started, run.info.run_id, promoted=url)
    typer.echo(f"Trained {model} on {metadata['trained_on'][0]} to {metadata['trained_on'][1]}.")
    typer.echo(f"Saved to {directory}")
    if url:
        typer.echo(f"Promoted: {url}")


def _record_training(
    engine: Engine,
    metadata: dict[str, Any],
    region: Region,
    sha: str | None,
    started: datetime,
    mlflow_run_id: str,
    *,
    promoted: str | None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO model_runs (model_name, model_kind, run_type, region, git_sha, "
                "mlflow_run_id, hf_model_version, feature_version, params, train_start, "
                "train_end, status, started_at, finished_at) VALUES (:name, 'model', 'train', "
                ":region, :sha, :mlflow, :hf, :feature_version, CAST(:params AS JSONB), :start, "
                ":end, 'succeeded', :started, now())"
            ),
            {
                "name": metadata["name"],
                "region": region.value,
                "sha": sha,
                "mlflow": mlflow_run_id,
                "hf": (sha or "")[:12] if promoted else None,
                "feature_version": metadata["feature_version"],
                "params": json.dumps(metadata["state"].get("params", {})),
                "start": metadata["trained_on"][0],
                "end": metadata["trained_on"][1],
                "started": started,
            },
        )


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
