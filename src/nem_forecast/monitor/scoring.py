"""Score served forecasts against actual demand as it lands.

The API logs every forecast it serves in ``predictions``. Once a target half-hour has all six of
its 5-minute intervals in ``raw_dispatch``, the forecast for it gets a row in
``prediction_scores``: the forecast, the actual, and the error.

* **One score per forecast, not per request.** A forecast is (region, model version, origin,
  target). Clients often ask for the same forecast more than once, and counting each request
  would weight popular origins; the first-served values are scored.
* **Actuals are the training target.** The half-hour mean of six 5-minute values, from
  :func:`~nem_forecast.features.series.load_half_hourly`, so live error and backtest error
  measure the same thing.
* **Revised actuals are rescored, not overwritten.** AEMO revises data, and the monthly archive
  can restate a whole month after it ends. Every run re-checks the last ``rescore_days`` of
  scores; when an actual has changed, the old score is marked superseded and a new one added.

Metrics reuse the backtest's :func:`~nem_forecast.backtest.metrics.score`, so a realised MAE and
a backtest MAE are computed by the same code.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import polars as pl
from sqlalchemy import Engine, text

from nem_forecast.backtest.metrics import ALL_BUCKETS, score, tod_bucket
from nem_forecast.features.build import LABEL, with_calendar
from nem_forecast.features.series import PERIOD, TIME_ZONE, load_half_hourly
from nem_forecast.models.base import HORIZON_GROUPS, horizon_group_label
from nem_forecast.schemas import Region

log = logging.getLogger(__name__)

RESCORE_DAYS = 35  # the monthly archive can restate a month after it ends
TOLERANCE_MW = 1e-6  # demand is stored to 0.01 MW, so any real revision moves a mean by > 0.001
LIVE = "live"  # the model label for realised metrics pooled across every served version

_TS = pl.Datetime("us", TIME_ZONE)
_KEY = ["model_version", "forecast_origin", "target_period"]

_EARLIEST_UNSCORED = """
SELECT min(p.target_period)
FROM predictions p
WHERE p.region = :region AND p.target_period <= :now
  AND NOT EXISTS (
      SELECT 1 FROM prediction_scores s
      WHERE s.superseded_at IS NULL AND s.region = p.region
        AND s.model_version = p.model_version AND s.forecast_origin = p.forecast_origin
        AND s.target_period = p.target_period
  )
"""

# The first-served row of each forecast whose target lies in the window.
_FORECASTS = """
SELECT DISTINCT ON (model_version, forecast_origin, target_period)
       id AS prediction_id, model_version, forecast_origin, target_period, horizon,
       predicted_mw, lower_mw, upper_mw
FROM predictions
WHERE region = :region AND target_period >= :start AND target_period <= :now
ORDER BY model_version, forecast_origin, target_period, served_at, id
"""

_CURRENT = """
SELECT id AS score_id, model_version, forecast_origin, target_period, actual_mw AS scored_actual
FROM prediction_scores
WHERE region = :region AND superseded_at IS NULL AND target_period >= :start
"""

_SUPERSEDE = """
UPDATE prediction_scores SET superseded_at = :now WHERE id = ANY(:ids) AND superseded_at IS NULL
"""

_INSERT = """
INSERT INTO prediction_scores (
    prediction_id, region, model_version, forecast_origin, target_period, horizon,
    predicted_mw, lower_mw, upper_mw, actual_mw, scored_at
) VALUES (
    :prediction_id, :region, :model_version, :forecast_origin, :target_period, :horizon,
    :predicted_mw, :lower_mw, :upper_mw, :actual_mw, :now
)
"""

_SCORES = """
SELECT model_version, forecast_origin, target_period, horizon,
       predicted_mw, lower_mw, upper_mw, actual_mw
FROM prediction_scores
WHERE region = :region AND superseded_at IS NULL
  AND target_period > :start AND target_period <= :end
"""

_SERVED = """
SELECT DISTINCT ON (forecast_origin, target_period) forecast_origin, horizon, predicted_mw
FROM predictions
WHERE region = :region AND model_version = :version
  AND forecast_origin > :start AND forecast_origin <= :end
ORDER BY forecast_origin, target_period, served_at, id
"""

_FORECAST_SCHEMA = pl.Schema(
    {
        "prediction_id": pl.Int64(),
        "model_version": pl.String(),
        "forecast_origin": _TS,
        "target_period": _TS,
        "horizon": pl.Int16(),
        "predicted_mw": pl.Float64(),
        "lower_mw": pl.Float64(),
        "upper_mw": pl.Float64(),
    }
)
_CURRENT_SCHEMA = pl.Schema(
    {
        "score_id": pl.Int64(),
        "model_version": pl.String(),
        "forecast_origin": _TS,
        "target_period": _TS,
        "scored_actual": pl.Float64(),
    }
)
SCORES_SCHEMA = pl.Schema(
    {
        "model_version": pl.String(),
        "forecast_origin": _TS,
        "target_period": _TS,
        "horizon": pl.Int16(),
        "predicted_mw": pl.Float64(),
        "lower_mw": pl.Float64(),
        "upper_mw": pl.Float64(),
        "actual_mw": pl.Float64(),
    }
)


@dataclass(frozen=True)
class ScoreReport:
    scored: int  # forecasts scored for the first time
    rescored: int  # forecasts whose actual was revised since they were scored
    unchanged: int  # previously scored forecasts re-checked and still right
    pending: int  # target has passed but its half-hour is not complete in raw_dispatch yet
    window_start: datetime | None  # earliest target period examined


def score_predictions(
    engine: Engine,
    region: Region = Region.VIC1,
    *,
    now: datetime,
    rescore_days: int = RESCORE_DAYS,
) -> ScoreReport:
    """Score every forecast whose actual has landed; rescore any whose actual was revised."""
    now = now.astimezone(UTC)
    params: dict[str, object] = {"region": region.value, "now": now}
    with engine.connect() as conn:
        earliest = conn.execute(text(_EARLIEST_UNSCORED), params).scalar()
    start = _floor_to_period(now - timedelta(days=rescore_days))
    if earliest is not None:
        start = min(start, earliest.astimezone(UTC))
    params["start"] = start

    forecasts = _read(engine, _FORECASTS, params, _FORECAST_SCHEMA)
    if forecasts.is_empty():
        return ScoreReport(0, 0, 0, 0, None)

    # The half-hour ending at `start` needs the intervals after start - 30 min.
    actuals = load_half_hourly(engine, region, start=start - PERIOD, end=now).select(
        pl.col("period_end").alias("target_period"), pl.col("demand").alias("actual_mw")
    )
    joined = forecasts.join(actuals, on="target_period", how="left")
    landed = joined.filter(pl.col("actual_mw").is_not_null())
    pending = joined.height - landed.height

    current = _read(engine, _CURRENT, params, _CURRENT_SCHEMA)
    checked = landed.join(current, on=_KEY, how="left")
    new = checked.filter(pl.col("score_id").is_null())
    revised = checked.filter(
        pl.col("score_id").is_not_null(),
        (pl.col("actual_mw") - pl.col("scored_actual")).abs() > TOLERANCE_MW,
    )
    unchanged = checked.height - new.height - revised.height

    rows = [
        {**row, "region": region.value, "now": now}
        for row in pl.concat([new, revised])
        .select(*_FORECAST_SCHEMA.names(), "actual_mw")
        .iter_rows(named=True)
    ]
    with engine.begin() as conn:
        if revised.height:
            conn.execute(text(_SUPERSEDE), {"now": now, "ids": revised["score_id"].to_list()})
        if rows:
            conn.execute(text(_INSERT), rows)
    report = ScoreReport(new.height, revised.height, unchanged, pending, start)
    log.info(
        "scored %d forecasts, rescored %d after revisions, %d unchanged, %d awaiting actuals",
        report.scored,
        report.rescored,
        report.unchanged,
        report.pending,
    )
    return report


def load_scores(
    engine: Engine, region: Region = Region.VIC1, *, start: datetime, end: datetime
) -> pl.DataFrame:
    """Current scores for target periods in ``(start, end]``."""
    params = {"region": region.value, "start": start, "end": end}
    return _read(engine, _SCORES, params, SCORES_SCHEMA)


def load_served(
    engine: Engine,
    region: Region = Region.VIC1,
    *,
    model_version: str,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """First-served forecasts of one model from origins in ``(start, end]``.

    Columns: forecast_origin, horizon, predicted_mw.
    """
    params = {"region": region.value, "version": model_version, "start": start, "end": end}
    return _read(
        engine,
        _SERVED,
        params,
        pl.Schema({"forecast_origin": _TS, "horizon": pl.Int16(), "predicted_mw": pl.Float64()}),
    )


def realised_metrics(scores: pl.DataFrame, *, model: str = LIVE) -> pl.DataFrame:
    """Tidy metrics of scored forecasts, in the backtest's format (``model`` labels the rows)."""
    rows = with_calendar(
        scores.select(
            pl.lit(model).alias("model"),
            pl.col("target_period").alias("target"),
            "horizon",
            pl.col("actual_mw").alias(LABEL),
            pl.col("lower_mw").alias("p10"),
            pl.col("predicted_mw").alias("p50"),
            pl.col("upper_mw").alias("p90"),
        )
    )
    return score(rows)


def compare_with_backtest(
    scores: pl.DataFrame, backtest: pl.DataFrame, *, model: str
) -> pl.DataFrame:
    """Realised MAE by horizon group, against what the backtest predicts for the same forecasts.

    Error depends heavily on horizon and time of day, and live scores do not cover them evenly
    (a day-old forecast has all 48 horizons landed, this morning's only a few). So the backtest
    figure is the backtest's MAE for each (horizon, time-of-day bucket) cell, weighted by how
    many live forecasts fell in that cell: the error the backtest says these very forecasts
    should have had.

    ``backtest`` is :func:`~nem_forecast.models.registry.latest_backtest` output; ``model`` names
    the backtested model the served one was trained as. Columns: group, n, realised_mae,
    backtest_mae, ratio. The last row, group ``all``, pools every horizon.
    """
    reference = backtest.filter(
        pl.col("model") == model,
        pl.col("metric") == "mae",
        pl.col("horizon") > 0,
        pl.col("tod_bucket") != ALL_BUCKETS,
    ).select(
        pl.col("horizon").cast(pl.Int16),
        "tod_bucket",
        pl.col("value").alias("backtest_ae"),
    )
    if reference.is_empty():
        raise ValueError(f"no recorded backtest MAE for {model}")
    group_of = {h: horizon_group_label(group) for group in HORIZON_GROUPS for h in group}
    rows = (
        with_calendar(scores.rename({"target_period": "target"}))
        .with_columns(
            tod_bucket=tod_bucket(pl.col("half_hour")),
            group=pl.col("horizon").replace_strict(
                group_of, default="other", return_dtype=pl.String
            ),
            abs_error=(pl.col("predicted_mw") - pl.col("actual_mw")).abs(),
        )
        .join(reference, on=["horizon", "tod_bucket"], how="left")
    )
    order = [horizon_group_label(group) for group in HORIZON_GROUPS]
    by_group = (
        rows.group_by("group")
        .agg(
            n=pl.len(),
            realised_mae=pl.col("abs_error").mean(),
            backtest_mae=pl.col("backtest_ae").mean(),
        )
        .with_columns(
            pl.col("group")
            .replace_strict(order, list(range(len(order))), default=99)
            .alias("_order")
        )
        .sort("_order")
        .drop("_order")
    )
    pooled = rows.select(
        group=pl.lit("all"),
        n=pl.len(),
        realised_mae=pl.col("abs_error").mean(),
        backtest_mae=pl.col("backtest_ae").mean(),
    )
    return (
        pl.concat([by_group, pooled], how="vertical_relaxed")
        .with_columns(
            pl.col("n").cast(pl.Int64), ratio=pl.col("realised_mae") / pl.col("backtest_mae")
        )
        .filter(pl.col("n") > 0)
    )


def _read(
    engine: Engine, query: str, params: Mapping[str, object], schema: pl.Schema
) -> pl.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).all()
    return pl.DataFrame([tuple(r) for r in rows], schema=schema, orient="row")


def _floor_to_period(ts: datetime) -> datetime:
    utc = ts.astimezone(UTC)
    return utc.replace(minute=utc.minute - utc.minute % 30, second=0, microsecond=0)
