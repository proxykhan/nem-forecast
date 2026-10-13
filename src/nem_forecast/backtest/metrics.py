"""Forecast accuracy metrics in tidy form.

One row per (model, metric, horizon, tod_bucket), matching the model_run_metrics table:
``horizon = 0`` means all horizons pooled and ``tod_bucket = 'all'`` all times of day. Metrics
come out per horizon and per time-of-day bucket because a one-step forecast and a
24-hour-ahead forecast fail differently, and so do the overnight trough and the evening peak.

* mae, rmse: of the median (p50), in MW
* mape: of the median, in percent
* pinball_q10, pinball_q50, pinball_q90: quantile loss of each predicted quantile, in MW
* pinball_avg: mean of the three; a single score for the whole predictive distribution
* coverage_80: percent of actuals inside [p10, p90]; 80 when the interval is calibrated
"""

from __future__ import annotations

import polars as pl

from nem_forecast.features.build import LABEL

ALL_HORIZONS = 0
ALL_BUCKETS = "all"
METRICS = (
    "mae",
    "rmse",
    "mape",
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "pinball_avg",
    "coverage_80",
)

# Melbourne wall-clock time of the target period's start. Chosen to separate the regimes
# that drive VIC1 demand, not tuned: overnight base load, the morning ramp, the midday
# rooftop-solar trough, the evening peak, and the late-evening fall-off.
TOD_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("00-06 overnight", 0, 12),
    ("06-10 morning ramp", 12, 20),
    ("10-16 solar trough", 20, 32),
    ("16-21 evening peak", 32, 42),
    ("21-24 late evening", 42, 48),
)


def tod_bucket(half_hour: pl.Expr) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.String)
    for name, start, end in reversed(TOD_BUCKETS):
        expr = (
            pl.when(half_hour.is_between(start, end, closed="left"))
            .then(pl.lit(name))
            .otherwise(expr)
        )
    return expr


def pinball(actual: pl.Expr, predicted: pl.Expr, q: float) -> pl.Expr:
    diff = actual - predicted
    return pl.max_horizontal(q * diff, (q - 1) * diff)


def score(predictions: pl.DataFrame) -> pl.DataFrame:
    """Tidy metrics from rows of (model, horizon, half_hour, demand, p10, p50, p90)."""
    actual = pl.col(LABEL)
    rows = predictions.drop_nulls([LABEL, "p10", "p50", "p90"]).with_columns(
        tod_bucket=tod_bucket(pl.col("half_hour")),
        _abs=(actual - pl.col("p50")).abs(),
        _sq=(actual - pl.col("p50")) ** 2,
        _ape=(actual - pl.col("p50")).abs() / actual.abs(),
        _pin10=pinball(actual, pl.col("p10"), 0.1),
        _pin50=pinball(actual, pl.col("p50"), 0.5),
        _pin90=pinball(actual, pl.col("p90"), 0.9),
        _covered=actual.is_between(pl.col("p10"), pl.col("p90")),
    )
    aggregates = [
        pl.col("_abs").mean().alias("mae"),
        pl.col("_sq").mean().sqrt().alias("rmse"),
        (pl.col("_ape").mean() * 100).alias("mape"),
        pl.col("_pin10").mean().alias("pinball_q10"),
        pl.col("_pin50").mean().alias("pinball_q50"),
        pl.col("_pin90").mean().alias("pinball_q90"),
        (pl.col("_covered").mean() * 100).alias("coverage_80"),
        pl.len().alias("n_obs"),
    ]
    groupings = (
        ("model", "horizon", "tod_bucket"),
        ("model", "horizon"),
        ("model", "tod_bucket"),
        ("model",),
    )
    frames = []
    for keys in groupings:
        grouped = rows.group_by(keys).agg(aggregates)
        if "horizon" not in keys:
            grouped = grouped.with_columns(horizon=pl.lit(ALL_HORIZONS))
        if "tod_bucket" not in keys:
            grouped = grouped.with_columns(tod_bucket=pl.lit(ALL_BUCKETS))
        frames.append(
            grouped.with_columns(
                pinball_avg=pl.mean_horizontal("pinball_q10", "pinball_q50", "pinball_q90")
            ).select("model", pl.col("horizon").cast(pl.Int16), "tod_bucket", *METRICS, "n_obs")
        )
    return (
        pl.concat(frames)
        .unpivot(index=["model", "horizon", "tod_bucket", "n_obs"], variable_name="metric")
        .select("model", "metric", "horizon", "tod_bucket", "value", pl.col("n_obs").cast(pl.Int64))
        .sort("model", "metric", "horizon", "tod_bucket")
    )
