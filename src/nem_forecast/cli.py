"""``nem-forecast`` command-line entrypoint.

Ingestion and serving work with the core install. Commands that train, track or report import
their heavy tools (MLflow, Optuna, matplotlib, Evidently) inside the command, from the ``train``
and ``monitor`` extras, so the scheduled ingest job and the serving image never need them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import polars as pl
import typer
from sqlalchemy import Engine, text

from nem_forecast import __version__
from nem_forecast.backtest.baselines import BASELINES, Forecaster
from nem_forecast.backtest.harness import (
    BacktestConfig,
    forecaster_params,
    plan_folds,
    prune_backtests,
    run_backtest,
    save_results,
)
from nem_forecast.backtest.metrics import ALL_BUCKETS, ALL_HORIZONS
from nem_forecast.config import get_settings
from nem_forecast.db import MIGRATIONS_DIR, apply_migrations, create_db_engine
from nem_forecast.features.build import FEATURE_VERSION, one_step_features, write_features
from nem_forecast.features.series import load_half_hourly, span
from nem_forecast.gitinfo import git_sha
from nem_forecast.ingest.backfill import MonthStatus, default_months
from nem_forecast.ingest.backfill import backfill as run_backfill
from nem_forecast.ingest.loader import coverage
from nem_forecast.ingest.nemweb import DEFAULT_CACHE_DIR, NemwebClient
from nem_forecast.ingest.recent import ingest_dispatch_is
from nem_forecast.models.base import METADATA_FILE, DirectModel
from nem_forecast.models.catalog import MODELS
from nem_forecast.models.lgbm import LightGBMDirect, LightGBMQuantile
from nem_forecast.models.params import MAX_TRIALS, VALIDATION_DAYS, load_params, params_path
from nem_forecast.models.ridge import RidgeForecaster
from nem_forecast.schemas import NEM_TZ, Region

if TYPE_CHECKING:
    from nem_forecast.monitor.alerts import GitHubIssues
    from nem_forecast.serve.model_store import LoadedModel

log = logging.getLogger(__name__)

app = typer.Typer(
    name="nem-forecast",
    help="Forecast VIC1 electricity demand from AEMO NEM data.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # locals can hold secrets from Settings
)


_TIME_FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"]
CacheDir = Annotated[Path, typer.Option(help="Local response cache for downloaded AEMO files.")]


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
    since_latest: Annotated[
        bool,
        typer.Option(
            help="Start half an hour before the latest interval already loaded, but no more "
            "than --hours back: a late scheduled run catches up instead of leaving a gap."
        ),
    ] = False,
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
        if since_latest and start is None:
            latest = [coverage(engine, each)[2] for each in regions]
            loaded = [ts for ts in latest if ts is not None]
            if len(loaded) == len(regions):  # a region with no data reads the full --hours
                start_at = max(start_at, min(loaded) - timedelta(minutes=30))
            typer.echo(f"Reading from {start_at.astimezone(NEM_TZ):%Y-%m-%d %H:%M} (market time)")
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
        frame = one_step_features(load_half_hourly(engine, region))
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
    from nem_forecast.models.tuning import save_params, tune_lightgbm, validation_split
    from nem_forecast.tracking.client import configure_mlflow
    from nem_forecast.tracking.runs import MlflowTuningLogger

    configure_mlflow(get_settings())
    sha = git_sha()
    with _database() as engine:
        series = load_half_hourly(engine, region)
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
    fold_artifacts: Annotated[
        bool, typer.Option(help="Attach the report to every fold run, not just each model's.")
    ] = True,
    retain_days: Annotated[
        int | None,
        typer.Option(min=1, help="After saving, delete backtest runs older than this many days."),
    ] = None,
    readme: Annotated[
        Path | None, typer.Option(help="Rewrite the results section of this README.")
    ] = None,
) -> None:
    """Walk-forward backtest of the baselines and models; writes model_runs and a report."""
    from nem_forecast.backtest.report import (
        CHART_NAME,
        render_results_block,
        update_readme,
        verdict,
        write_report,
    )
    from nem_forecast.tracking.client import configure_mlflow
    from nem_forecast.tracking.runs import MlflowBacktestObserver

    started = datetime.now(UTC)
    config = BacktestConfig(test_days=test_days, refit_every_days=refit_every_days)
    sha = git_sha()
    if track:
        configure_mlflow(get_settings())
    forecasters = _forecasters(region, with_models=models)
    with _database() as engine:
        series = load_half_hourly(engine, region)
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
                params={f.name: forecaster_params(f) for f in forecasters},
            )
            if save
            else None
        )
        if save and retain_days is not None:
            pruned = prune_backtests(engine, region, keep_days=retain_days)
            typer.echo(f"Pruned {pruned} backtest runs older than {retain_days} days.")
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
        observer.log_report([path, path.parent / CHART_NAME], include_folds=fold_artifacts)
    if readme is not None:
        update_readme(readme, render_results_block(result, generated_at=started))

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
    import mlflow
    from huggingface_hub import HfApi

    from nem_forecast.models.registry import (
        CARD_FILES,
        latest_backtest,
        push_to_hub,
        render_model_card,
        repo_name,
    )
    from nem_forecast.tracking.client import configure_mlflow, experiment_id

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
        series = load_half_hourly(engine, region)
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
                forecaster_params(forecaster)
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
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "0.0.0.0",  # noqa: S104
    port: Annotated[
        int | None, typer.Option(help="Port; defaults to $PORT (set by Render) or 8000.")
    ] = None,
) -> None:
    """Start the forecast API (FastAPI under uvicorn)."""
    from nem_forecast.serve.__main__ import run

    run(host=host, port=port)


def _served_model() -> LoadedModel:
    """The model the API serves, from MODEL_REPO_ID at MODEL_REVISION (or MODEL_DIR)."""
    from nem_forecast.serve.model_store import fetch_model

    settings = get_settings()
    if settings.model_repo_id is None and settings.model_dir is None:
        raise typer.BadParameter(
            "set MODEL_REPO_ID and MODEL_REVISION to the served model (see render.yaml)"
        )
    return fetch_model(settings)


def _issues(enabled: bool) -> GitHubIssues | None:
    from nem_forecast.monitor.alerts import GitHubIssues

    if not enabled:
        return None
    settings = get_settings()
    if settings.github_token is None or not settings.github_repository:
        raise typer.BadParameter("GITHUB_TOKEN and GITHUB_REPOSITORY are needed to open issues")
    return GitHubIssues(settings.github_repository, settings.github_token.get_secret_value())


def _write_summary(path: Path | None, summary: dict[str, Any]) -> None:
    if path is not None:
        path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")


@app.command()
def score(
    region: Annotated[Region, typer.Option(help="Region to score.")] = Region.VIC1,
    days: Annotated[int, typer.Option(min=1, help="Window for realised error.")] = 30,
    model: Annotated[
        str, typer.Option(help="Backtested model the served one was trained as.")
    ] = "lightgbm_quantile",
    rescore_days: Annotated[
        int, typer.Option(min=1, help="Re-check scores this recent for revised actuals.")
    ] = 35,
) -> None:
    """Score served forecasts whose actuals have landed; compare their error with the backtest."""
    from nem_forecast.models.registry import latest_backtest
    from nem_forecast.monitor.scoring import compare_with_backtest, load_scores, score_predictions

    now = datetime.now(UTC)
    with _database() as engine:
        report = score_predictions(engine, region, now=now, rescore_days=rescore_days)
        scores = load_scores(engine, region, start=now - timedelta(days=days), end=now)
        backtest = latest_backtest(engine, region)
    typer.echo(
        f"Scored {report.scored:,} forecasts, rescored {report.rescored:,} after revisions, "
        f"{report.unchanged:,} unchanged, {report.pending:,} awaiting actuals."
    )
    if scores.is_empty():
        typer.echo("No scored forecasts in the window yet.")
        return
    typer.echo(
        f"Last {days} days: {scores.height:,} forecasts from "
        f"{scores['forecast_origin'].n_unique():,} origins, "
        f"models {', '.join(sorted(scores['model_version'].unique()))}."
    )
    table = compare_with_backtest(scores, backtest, model=model)
    typer.echo(f"{'horizons':<10}{'n':>8}{'realised MAE':>14}{'backtest MAE':>14}{'ratio':>8}")
    for row in table.iter_rows(named=True):
        typer.echo(
            f"{row['group']:<10}{row['n']:>8,}{row['realised_mae']:>14,.1f}"
            f"{row['backtest_mae']:>14,.1f}{row['ratio']:>8.2f}"
        )


@app.command()
def monitor(
    region: Annotated[Region, typer.Option(help="Region to monitor.")] = Region.VIC1,
    site_dir: Annotated[Path, typer.Option(help="Where the published site is written.")] = Path(
        "site"
    ),
    site_url: Annotated[
        str | None, typer.Option(help="Public URL the site is published at, for links.")
    ] = None,
    run_url: Annotated[str | None, typer.Option(help="URL of this CI run, for links.")] = None,
    open_issues: Annotated[
        bool, typer.Option(help="Open a GitHub issue for each breached check.")
    ] = False,
    summary: Annotated[
        Path | None, typer.Option(help="Write the outcome as JSON here (for the workflow).")
    ] = None,
) -> None:
    """Score served forecasts, check drift, evaluate alerts and write the monitoring site."""
    from nem_forecast.monitor.drift import Thresholds, thresholds_path
    from nem_forecast.monitor.jobs import run_monitor

    served = _served_model()
    issues = _issues(open_issues)
    report_url = f"{site_url.rstrip('/')}/drift_report.html" if site_url else None
    with _database() as engine:
        outcome = run_monitor(
            engine,
            served,
            region=region,
            now=datetime.now(UTC),
            site_dir=site_dir,
            report_url=report_url,
            run_url=run_url,
            thresholds=Thresholds.load(thresholds_path(region)),
            issues=issues,
            git_sha=git_sha(),
        )
    for check in outcome.checks:
        typer.echo(f"{check.status:<18} {check.title}: {check.detail}")
    for url in outcome.issues:
        typer.echo(f"Issue: {url}")
    typer.echo(f"Site: {site_dir / 'index.html'}")
    _write_summary(
        summary,
        {
            "breached": [c.name for c in outcome.breached],
            "retrain": outcome.retrain,
            "issues": outcome.issues,
        },
    )


@app.command()
def calibrate_drift(
    region: Annotated[Region, typer.Option(help="Region to calibrate.")] = Region.VIC1,
    weeks: Annotated[int, typer.Option(min=20, help="Past weeks to score.")] = 104,
    alarm_rate: Annotated[
        float, typer.Option(min=0.001, max=0.5, help="Share of past weeks that should alert.")
    ] = 0.05,
) -> None:
    """Set drift thresholds from history: score every past week against its seasonal reference."""
    from nem_forecast.monitor.drift import calibrate, thresholds_path, weekly_scores

    with _database() as engine:
        series = load_half_hourly(engine, region)
    scores = weekly_scores(series, weeks=weeks)
    result = {
        "region": region.value,
        "computed_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        **calibrate(scores, alarm_rate=alarm_rate),
    }
    path = thresholds_path(region)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    typer.echo(
        f"{result['weeks']} weeks to {result['last_week_end'][:10]}: Evidently's default threshold "
        f"would have alerted in {result['default_alarm_rate']:.0%} of them; at the "
        f"{result['quantile']:.3f} quantile, {result['achieved_alarm_rate']:.0%}."
    )
    typer.echo(f"{'column':<22}{'median':>9}{'threshold':>11}")
    for column, threshold in result["thresholds"].items():
        typer.echo(f"{column:<22}{result['median_scores'][column]:>9.3f}{threshold:>11.3f}")
    typer.echo(f"Thresholds: {path}")


@app.command()
def retrain(
    region: Annotated[Region, typer.Option(help="Region to retrain.")] = Region.VIC1,
    trigger: Annotated[str, typer.Option(help="Why: 'manual' or the alert that fired.")] = "manual",
    holdout_days: Annotated[
        int, typer.Option(min=7, help="Holdout length (never before the incumbent's data ends).")
    ] = 14,
    promote: Annotated[
        bool, typer.Option(help="Promote to Hugging Face Hub if the candidate wins.")
    ] = False,
    open_issues: Annotated[
        bool, typer.Option(help="Open a GitHub issue if the candidate is rejected.")
    ] = False,
    output_dir: Annotated[Path, typer.Option(help="Where the new model is saved.")] = Path(
        "artifacts"
    ),
    summary: Annotated[
        Path | None, typer.Option(help="Write the outcome as JSON here (for the workflow).")
    ] = None,
) -> None:
    """Retrain the served model's type; promote only if it beats the incumbent at every horizon.

    Without --promote this is a dry run: the comparison is printed and nothing is recorded.
    """
    from nem_forecast.models.registry import HubPublisher
    from nem_forecast.monitor.jobs import run_retrain

    settings = get_settings()
    incumbent = _served_model()
    publisher = None
    if promote:
        if settings.hf_token is None:
            raise typer.BadParameter("HF_TOKEN is not set")
        publisher = HubPublisher(
            settings.hf_token.get_secret_value(), region, cache_dir=settings.model_cache_dir
        )
    with _database() as engine:
        outcome = run_retrain(
            engine,
            incumbent,
            lambda: _make_model(incumbent.name, region),
            region=region,
            started=datetime.now(UTC),
            trigger=trigger,
            output_dir=output_dir,
            publisher=publisher,
            issues=_issues(open_issues),
            git_sha=git_sha(),
            holdout_days=holdout_days,
        )
    if outcome.comparison is not None:
        typer.echo(outcome.comparison.markdown(incumbent=incumbent.version))
    typer.echo(f"Decision: {outcome.decision}: {outcome.reason}")
    if outcome.url:
        typer.echo(f"Promoted: {outcome.url}")
    if outcome.issue:
        typer.echo(f"Issue: {outcome.issue}")
    _write_summary(summary, outcome.to_dict())


READER_TABLES = (
    "raw_dispatch",
    "predictions",
    "prediction_scores",
    "monitor_runs",
    "retrain_runs",
    "model_runs",
    "model_run_metrics",
)


@app.command()
def grant_reader(
    role: Annotated[str, typer.Option(help="Login role to create, or rotate.")] = "dashboard",
) -> None:
    """Create a read-only database login for the dashboard and print its connection string.

    Run it yourself: the password is generated here, printed once, and stored nowhere else.
    Running it again for the same role rotates the password.
    """
    import secrets

    from psycopg import sql
    from sqlalchemy.engine import make_url

    if not role.isidentifier():
        raise typer.BadParameter("use letters, digits and underscores", param_hint="--role")
    password = secrets.token_urlsafe(24)
    url = make_url(get_settings().database_url.get_secret_value())
    name = sql.Identifier(role)
    with _database() as engine, engine.begin() as conn:
        exists = bool(
            conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role}).scalar()
        )
        verb = "ALTER" if exists else "CREATE"
        statements = [
            sql.SQL(verb + " ROLE {} WITH LOGIN PASSWORD {}").format(name, sql.Literal(password)),
            sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(name),
            sql.SQL("ALTER ROLE {} SET statement_timeout = '15s'").format(name),
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(str(url.database)), name
            ),
            sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(name),
            sql.SQL("GRANT SELECT ON {} TO {}").format(
                sql.SQL(", ").join(sql.Identifier(t) for t in READER_TABLES), name
            ),
        ]
        raw = conn.connection.driver_connection
        if raw is None:
            raise RuntimeError("no open database connection")
        for statement in statements:
            raw.execute(statement)
    reader = url.set(drivername="postgresql", username=role, password=password)
    typer.echo(
        f"{'Rotated' if exists else 'Created'} read-only role {role}: SELECT on "
        f"{', '.join(READER_TABLES)}."
    )
    typer.echo("Connection string, shown once; paste it into Vercel as DATABASE_URL:")
    typer.echo(reader.render_as_string(hide_password=False))


if __name__ == "__main__":
    app()
