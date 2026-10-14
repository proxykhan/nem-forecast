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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Protocol

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
    kinds: dict[str, str] = field(default_factory=dict)  # model -> "baseline" | "model"


class HarnessError(RuntimeError):
    pass


class BacktestObserver(Protocol):
    """Hooks for experiment tracking; the harness itself knows nothing about MLflow."""

    def start_model(self, forecaster: Forecaster) -> None: ...

    def end_fold(self, forecaster: Forecaster, fold: Fold, predictions: pl.DataFrame) -> None: ...

    def end_model(self, forecaster: Forecaster, predictions: pl.DataFrame) -> None: ...

    def abort_model(self, forecaster: Forecaster) -> None: ...


FoldCallback = Callable[[Fold, pl.DataFrame], None]


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
    series: pl.DataFrame,
    forecaster: Forecaster,
    config: BacktestConfig,
    on_fold: FoldCallback | None = None,
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

        scored = (
            frame.select(*KEYS, LABEL, "half_hour")
            .join(predicted.select(*KEYS, *PREDICTIONS), on=list(KEYS), how="left")
            .with_columns(model=pl.lit(forecaster.name), fold=pl.lit(fold.number, pl.Int16))
            .filter(pl.col(LABEL).is_not_null())
        )
        if on_fold is not None:
            on_fold(fold, scored)
        outputs.append(scored)
        log.info(
            "%s fold %d: trained to %s, %d origins, %.1fs",
            forecaster.name,
            fold.number,
            fold.cutoff.isoformat(),
            len(origins),
            time.perf_counter() - started,
        )
    return pl.concat(outputs)


def run_backtest(
    series: pl.DataFrame,
    forecasters: Sequence[Forecaster],
    config: BacktestConfig,
    *,
    observer: BacktestObserver | None = None,
) -> BacktestResult:
    folds = plan_folds(series, config)
    per_model = []
    for forecaster in forecasters:
        if observer is None:
            per_model.append(walk_forward(series, forecaster, config))
            continue
        observer.start_model(forecaster)
        try:
            predictions = walk_forward(
                series, forecaster, config, on_fold=partial(observer.end_fold, forecaster)
            )
        except BaseException:
            observer.abort_model(forecaster)
            raise
        observer.end_model(forecaster, predictions)
        per_model.append(predictions)
    predictions = pl.concat(per_model)
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
        kinds={f.name: getattr(f, "kind", "model") for f in forecasters},
    )


def save_results(
    engine: Engine,
    result: BacktestResult,
    *,
    region: Region,
    git_sha: str | None,
    started_at: datetime,
    mlflow_run_ids: dict[str, str] | None = None,
    params: dict[str, dict[str, object]] | None = None,
) -> dict[str, uuid.UUID]:
    """One model_runs row per model plus its tidy metrics, in one transaction."""
    run_ids: dict[str, uuid.UUID] = {}
    with engine.begin() as conn:
        for model in result.metrics["model"].unique(maintain_order=True):
            run_id = uuid.uuid4()
            run_params = {
                **asdict(result.config),
                "description": result.descriptions.get(model, ""),
                **(params or {}).get(model, {}),
            }
            conn.execute(
                text(
                    "INSERT INTO model_runs (run_id, model_name, model_kind, run_type, region, "
                    "git_sha, feature_version, params, train_start, train_end, eval_start, "
                    "eval_end, status, started_at, finished_at, mlflow_run_id) VALUES (:run_id, "
                    ":model, :kind, 'backtest', :region, :git_sha, :feature_version, "
                    "CAST(:params AS JSONB), :train_start, :train_end, :eval_start, :eval_end, "
                    "'succeeded', :started_at, now(), :mlflow_run_id)"
                ),
                {
                    "run_id": run_id,
                    "model": model,
                    "kind": result.kinds.get(model, "model"),
                    "region": region.value,
                    "git_sha": git_sha,
                    "feature_version": FEATURE_VERSION,
                    "params": json.dumps(run_params),
                    "train_start": result.train_start,
                    "train_end": result.folds[-1].cutoff,
                    "eval_start": result.eval_start,
                    "eval_end": result.eval_end,
                    "started_at": started_at,
                    "mlflow_run_id": (mlflow_run_ids or {}).get(model),
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


def forecaster_params(forecaster: Forecaster) -> dict[str, Any]:
    """Plain description of a forecaster, for run records."""
    params: dict[str, Any] = {
        "model": forecaster.name,
        "model_kind": getattr(forecaster, "kind", "model"),
        "description": getattr(forecaster, "description", ""),
    }
    if rule := getattr(forecaster, "point", None):
        params["rule_feature"] = rule
    for key, value in (getattr(forecaster, "params", None) or {}).items():
        params[f"lgbm.{key}"] = value
    if groups := getattr(forecaster, "horizon_groups", None):
        params["horizon_groups"] = "|".join(f"{g[0]}-{g[-1]}" for g in groups)
    return params


def prune_backtests(engine: Engine, region: Region, *, keep_days: int) -> int:
    """Delete backtest runs (and their metrics) older than ``keep_days``; returns how many.

    A nightly backtest writes about 16,000 metric rows; without pruning that would outgrow a
    0.5 GB free database within a year. Training runs, which record promoted models, are kept.
    """
    with engine.begin() as conn:
        deleted = conn.execute(
            text(
                "DELETE FROM model_runs WHERE run_type = 'backtest' AND region = :region "
                "AND started_at < now() - make_interval(days => :days)"
            ),
            {"region": region.value, "days": keep_days},
        ).rowcount
    return int(deleted or 0)


def _check_predictions(predicted: pl.DataFrame, frame: pl.DataFrame, name: str) -> None:
    missing = {*KEYS, *PREDICTIONS} - set(predicted.columns)
    if missing:
        raise HarnessError(f"{name}.predict() did not return {sorted(missing)}")
    if predicted.height != frame.height or predicted.select(KEYS).is_duplicated().any():
        raise HarnessError(
            f"{name}.predict() returned {predicted.height} rows for {frame.height} inputs"
        )
