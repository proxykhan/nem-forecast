"""Walk-forward backtesting.

Expanding window: at each refit point the forecaster is fitted on every half-hour up to that
point, then forecasts 48 half-hours ahead from every half-hourly origin until the next refit,
30 days later. Origins cover every time of day, so error by horizon is not confounded with
time of day (a single daily origin would make horizon 12 mean "noon" and nothing else).

Two guarantees, both enforced on every fold rather than assumed:

* ``fit`` only ever receives history ending at the fold's first origin.
* ``predict`` only ever receives features that pass :func:`assert_no_leakage`, without the label.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import polars as pl
from sqlalchemy import Engine, text

from nem_forecast.backtest.baselines import PREDICTIONS, Forecaster
from nem_forecast.backtest.metrics import score
from nem_forecast.features.build import (
    DEFAULT_HORIZONS,
    FEATURE_VERSION,
    KEYS,
    LABEL,
    assert_no_leakage,
    build_features,
)
from nem_forecast.features.series import TIME_ZONE, span
from nem_forecast.schemas import NEM_TZ, Region

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    test_days: int = 365
    refit_every_days: int = 30
    origin_every_minutes: int = 30
    horizons: tuple[int, ...] = DEFAULT_HORIZONS


@dataclass(frozen=True)
class Fold:
    number: int
    cutoff: datetime  # training data ends here; also the fold's first origin
    end: datetime  # origins strictly before this belong to the fold


@dataclass(frozen=True)
class BacktestResult:
    config: BacktestConfig
    folds: list[Fold]
    predictions: pl.DataFrame
    metrics: pl.DataFrame
    train_start: datetime
    eval_start: datetime
    eval_end: datetime
    descriptions: dict[str, str] = field(default_factory=dict)


class HarnessError(RuntimeError):
    pass


def plan_folds(series: pl.DataFrame, config: BacktestConfig) -> list[Fold]:
    """Refit points every ``refit_every_days`` from market midnight ``test_days`` before the end."""
    first, last = span(series)
    start_local = (last - timedelta(days=config.test_days)).astimezone(NEM_TZ)
    cutoff = start_local.replace(hour=0, minute=0, second=0, microsecond=0)
    if cutoff <= first:
        raise HarnessError(f"series starts {first}; no training data before the test period")
    folds: list[Fold] = []
    while cutoff < last:
        end = min(cutoff + timedelta(days=config.refit_every_days), last)
        folds.append(Fold(len(folds), cutoff, end))
        cutoff = end
    return folds


def walk_forward(
    series: pl.DataFrame, forecaster: Forecaster, config: BacktestConfig
) -> pl.DataFrame:
    """Predictions for every fold: keys, actual demand, half_hour, p10/p50/p90, model, fold."""
    outputs = []
    for fold in plan_folds(series, config):
        started = time.perf_counter()
        forecaster.fit(series.filter(pl.col("period_end") <= fold.cutoff))

        origins = pl.datetime_range(
            fold.cutoff,
            fold.end,
            interval=f"{config.origin_every_minutes}m",
            closed="left",
            time_zone=TIME_ZONE,
            eager=True,
        )
        frame = build_features(series, origins, config.horizons)
        assert_no_leakage(frame)
        predicted = forecaster.predict(frame.drop(LABEL))
        _check_predictions(predicted, frame, forecaster.name)

        outputs.append(
            frame.select(*KEYS, LABEL, "half_hour")
            .join(predicted.select(*KEYS, *PREDICTIONS), on=list(KEYS), how="left")
            .with_columns(model=pl.lit(forecaster.name), fold=pl.lit(fold.number, pl.Int16))
        )
        log.info(
            "%s fold %d: trained to %s, %d origins, %.1fs",
            forecaster.name,
            fold.number,
            fold.cutoff.isoformat(),
            len(origins),
            time.perf_counter() - started,
        )
    return pl.concat(outputs).filter(pl.col(LABEL).is_not_null())


def run_backtest(
    series: pl.DataFrame, forecasters: Sequence[Forecaster], config: BacktestConfig
) -> BacktestResult:
    folds = plan_folds(series, config)
    predictions = pl.concat([walk_forward(series, f, config) for f in forecasters])
    eval_end = predictions["target"].max()
    if not isinstance(eval_end, datetime):
        raise HarnessError("the backtest produced no scored forecasts")
    return BacktestResult(
        config=config,
        folds=folds,
        predictions=predictions,
        metrics=score(predictions),
        train_start=span(series)[0],
        eval_start=folds[0].cutoff,
        eval_end=eval_end,
        descriptions={f.name: getattr(f, "description", "") for f in forecasters},
    )


def save_results(
    engine: Engine,
    result: BacktestResult,
    *,
    region: Region,
    git_sha: str | None,
    started_at: datetime,
    model_kind: str = "baseline",
) -> dict[str, uuid.UUID]:
    """One model_runs row per model plus its tidy metrics, in one transaction."""
    run_ids: dict[str, uuid.UUID] = {}
    with engine.begin() as conn:
        for model in result.metrics["model"].unique(maintain_order=True):
            run_id = uuid.uuid4()
            params = {**asdict(result.config), "description": result.descriptions.get(model, "")}
            conn.execute(
                text(
                    "INSERT INTO model_runs (run_id, model_name, model_kind, run_type, region, "
                    "git_sha, feature_version, params, train_start, train_end, eval_start, "
                    "eval_end, status, started_at, finished_at) VALUES (:run_id, :model, :kind, "
                    "'backtest', :region, :git_sha, :feature_version, CAST(:params AS JSONB), "
                    ":train_start, :train_end, :eval_start, :eval_end, 'succeeded', :started_at, "
                    "now())"
                ),
                {
                    "run_id": run_id,
                    "model": model,
                    "kind": model_kind,
                    "region": region.value,
                    "git_sha": git_sha,
                    "feature_version": FEATURE_VERSION,
                    "params": json.dumps(params),
                    "train_start": result.train_start,
                    "train_end": result.folds[-1].cutoff,
                    "eval_start": result.eval_start,
                    "eval_end": result.eval_end,
                    "started_at": started_at,
                },
            )
            rows = result.metrics.filter(pl.col("model") == model).drop("model")
            conn.execute(
                text(
                    "INSERT INTO model_run_metrics (run_id, metric, horizon, tod_bucket, value, "
                    "n_obs) VALUES (:run_id, :metric, :horizon, :tod_bucket, :value, :n_obs)"
                ),
                [{"run_id": run_id, **row} for row in rows.iter_rows(named=True)],
            )
            run_ids[model] = run_id
    return run_ids


def _check_predictions(predicted: pl.DataFrame, frame: pl.DataFrame, name: str) -> None:
    missing = {*KEYS, *PREDICTIONS} - set(predicted.columns)
    if missing:
        raise HarnessError(f"{name}.predict() did not return {sorted(missing)}")
    if predicted.height != frame.height or predicted.select(KEYS).is_duplicated().any():
        raise HarnessError(
            f"{name}.predict() returned {predicted.height} rows for {frame.height} inputs"
        )
