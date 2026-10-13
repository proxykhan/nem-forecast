"""Leakage-safe features for forecasting half-hourly demand from a forecast origin.

A forecast is made at an *origin* (the end of the latest half-hour with data) for horizons
h = 1..48 half-hours ahead; the *target* period ends at ``origin + h x 30 min``. Every feature
must be computable at the origin, which is stricter than "computable before the target": for
h = 5, the demand at t-1 does not exist yet when the forecast is made.

Features, with the doc's names at h = 1 and the latest data each one reads:

* ``demand_at_origin``: demand of the period ending at the origin (lag t-1). Reads: origin.
* ``same_slot_prev_day``: the same half-hour on the latest fully known day,
  t - 48 x ceil(h / 48) (lag t-48). Reads: target - 1 day.
* ``same_slot_prev_week``: the same half-hour a week earlier (lag t-336). Reads: target - 7 days.
* ``same_slot_mean_4w``: mean of the same half-hour on the last 4 same weekdays, null unless
  all 4 exist. Reads: target - 7 days.
* ``rolling_mean_24h``, ``rolling_std_24h``, ``rolling_mean_7d``: the 48 or 336 periods ending
  at the origin (24h mean, 24h std, 7d mean). Reads: origin.
* ``half_hour``, ``day_of_week``, ``month``, ``is_public_holiday``: calendar of the target
  period, known in advance.

Calendar features use Melbourne wall-clock time, because people's routines follow the clock
on the wall and shift with daylight saving. Lags use market time: "the same instant 24 hours
ago".

**Provenance.** Each data-derived feature ``x`` comes with ``x_asof``, the timestamp of the
latest observation it used. Values are fetched by joining the series on that very timestamp,
so provenance records what was actually used rather than what was intended.
:func:`assert_no_leakage` checks every ``_asof`` against the row's cutoff, and fails closed on
any column it cannot account for.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import date, datetime

import holidays
import polars as pl
import psycopg
from sqlalchemy import Engine, text

from nem_forecast.features.series import PERIOD, TIME_ZONE
from nem_forecast.schemas import Region

log = logging.getLogger(__name__)

FEATURE_VERSION = "v1"
LOCAL_TIME_ZONE = "Australia/Melbourne"
PERIODS_PER_DAY = 48
PERIODS_PER_WEEK = 336
DEFAULT_HORIZONS = tuple(range(1, PERIODS_PER_DAY + 1))
ASOF = "_asof"

KEYS = ("origin", "target", "horizon")
LABEL = "demand"
CALENDAR_FEATURES = ("half_hour", "day_of_week", "month", "is_public_holiday")
DATA_FEATURES = (
    "demand_at_origin",
    "same_slot_prev_day",
    "same_slot_prev_week",
    "same_slot_mean_4w",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_mean_7d",
)
FEATURES = ("horizon", *CALENDAR_FEATURES, *DATA_FEATURES)

_ROLLING_WINDOWS = {
    "rolling_mean_24h": ("24h", PERIODS_PER_DAY, "mean"),
    "rolling_std_24h": ("24h", PERIODS_PER_DAY, "std"),
    "rolling_mean_7d": ("7d", PERIODS_PER_WEEK, "mean"),
}


class LeakageError(AssertionError):
    """A feature used data that would not have existed at its forecast origin."""


def build_features(
    series: pl.DataFrame,
    origins: Iterable[datetime] | pl.Series,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    *,
    with_label: bool = True,
) -> pl.DataFrame:
    """One row per (origin, horizon), with provenance columns and optionally the label.

    ``series`` is the half-hourly frame from :func:`~nem_forecast.features.series.to_half_hourly`.
    It may extend beyond the origins (labels need it); features never read past their origin.
    """
    if not horizons or min(horizons) < 1 or max(horizons) > PERIODS_PER_WEEK:
        raise ValueError(f"horizons must lie in 1..{PERIODS_PER_WEEK}")

    origin_frame = pl.DataFrame({"origin": origins}).cast({"origin": pl.Datetime("us", TIME_ZONE)})
    frame = origin_frame.join(
        pl.DataFrame({"horizon": list(horizons)}, schema={"horizon": pl.Int16}), how="cross"
    ).with_columns(target=pl.col("origin") + pl.col("horizon").cast(pl.Int64) * PERIOD)

    days_back = (pl.col("horizon").cast(pl.Int64) + PERIODS_PER_DAY - 1) // PERIODS_PER_DAY
    frame = frame.with_columns(
        demand_at_origin_asof=pl.col("origin"),
        same_slot_prev_day_asof=pl.col("target") - pl.duration(days=days_back),
        same_slot_prev_week_asof=pl.col("target") - pl.duration(days=7),
        same_slot_mean_4w_asof=pl.col("target") - pl.duration(days=7),
        rolling_mean_24h_asof=pl.col("origin"),
        rolling_std_24h_asof=pl.col("origin"),
        rolling_mean_7d_asof=pl.col("origin"),
    )

    stats = _with_rolling_stats(series)
    frame = _lookup(frame, stats, "demand", "demand_at_origin")
    frame = _lookup(frame, stats, "demand", "same_slot_prev_day")
    frame = _lookup(frame, stats, "demand", "same_slot_prev_week")
    for name in _ROLLING_WINDOWS:
        frame = _lookup(frame, stats, name, name)

    weekly = [f"_week_{k}" for k in range(1, 5)]
    for k, column in enumerate(weekly, start=1):
        frame = frame.with_columns(
            (pl.col("target") - pl.duration(days=7 * k)).alias(f"{column}{ASOF}")
        )
        frame = _lookup(frame, stats, "demand", column).drop(f"{column}{ASOF}")
    frame = frame.with_columns(
        same_slot_mean_4w=pl.mean_horizontal(weekly, ignore_nulls=False)
    ).drop(weekly)

    frame = _with_calendar(frame)
    if with_label:
        frame = frame.join(
            series.select(pl.col("period_end").alias("target"), pl.col("demand").alias(LABEL)),
            on="target",
            how="left",
        )
    label = [LABEL] if with_label else []
    asof = [f"{name}{ASOF}" for name in DATA_FEATURES]
    return frame.select(*KEYS, *label, *CALENDAR_FEATURES, *DATA_FEATURES, *asof).sort(
        "origin", "horizon"
    )


def assert_no_leakage(features: pl.DataFrame, cutoff: str | datetime = "origin") -> None:
    """Raise :class:`LeakageError` if any row used data timestamped after its cutoff.

    ``cutoff`` is a column name (each row's own cutoff, normally its origin) or one timestamp
    for every row. Every column must be a key, the label, a calendar feature (known in
    advance), or a data feature with an ``_asof`` provenance column; anything else fails,
    because a feature that cannot be traced cannot be trusted.
    """
    known = {*KEYS, LABEL, *CALENDAR_FEATURES}
    asof_columns = [c for c in features.columns if c.endswith(ASOF)]
    traced = {c.removesuffix(ASOF) for c in asof_columns}
    untraced = [c for c in features.columns if c not in known | traced and c not in asof_columns]
    if untraced:
        raise LeakageError(f"no provenance for columns {untraced}; cannot verify them")

    limit = pl.col(cutoff) if isinstance(cutoff, str) else pl.lit(cutoff)
    for column in asof_columns:
        late = features.filter(pl.col(column) > limit)
        if late.height:
            row = late.row(0, named=True)
            raise LeakageError(
                f"{column.removesuffix(ASOF)}: {late.height} rows use data after their cutoff, "
                f"e.g. target {row.get('target')} with cutoff "
                f"{row[cutoff] if isinstance(cutoff, str) else cutoff} used data from {row[column]}"
            )


def one_step_features(series: pl.DataFrame) -> pl.DataFrame:
    """Horizon-1 features for every period: the doc's per-period table (lags t-1, t-48, t-336)."""
    origins = series["period_end"].head(len(series) - 1)
    return build_features(series, origins, horizons=(1,))


def write_features(engine: Engine, region: Region, frame: pl.DataFrame) -> int:
    """Replace this region's rows for FEATURE_VERSION in the features table; returns row count."""
    assert_no_leakage(frame)
    values = frame.select(
        "target",
        "origin",
        pl.struct(
            *[
                pl.col(c).fill_nan(None) if frame.schema[c].is_float() else pl.col(c)
                for c in FEATURES
            ]
        )
        .struct.json_encode()
        .alias("feature_values"),
    )
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM features WHERE region = :region AND feature_version = :version"),
            {"region": region.value, "version": FEATURE_VERSION},
        )
        driver = conn.connection.driver_connection
        if not isinstance(driver, psycopg.Connection):
            raise TypeError("write_features needs a PostgreSQL connection through psycopg 3")
        with (
            driver.cursor() as cursor,
            cursor.copy(
                "COPY features (settlement_date, region, feature_version, data_cutoff, "
                "feature_values) FROM STDIN"
            ) as copy,
        ):
            for target, origin, feature_values in values.iter_rows():
                copy.write_row((target, region.value, FEATURE_VERSION, origin, feature_values))
    log.info("wrote %d %s feature rows for %s", values.height, FEATURE_VERSION, region)
    return values.height


def vic_public_holidays(years: Iterable[int]) -> list[date]:
    return sorted(holidays.country_holidays("AU", subdiv="VIC", years=list(years)))


def _with_rolling_stats(series: pl.DataFrame) -> pl.DataFrame:
    """Each row's statistics cover the window ending at (and including) its own period."""
    exprs = []
    for name, (window, periods, kind) in _ROLLING_WINDOWS.items():
        if kind == "mean":
            expr = pl.col("demand").rolling_mean_by("period_end", window, min_samples=periods)
        else:
            expr = pl.col("demand").rolling_std_by("period_end", window, min_samples=periods)
        exprs.append(expr.alias(name))
    return series.sort("period_end").with_columns(exprs)


def _lookup(frame: pl.DataFrame, stats: pl.DataFrame, source: str, feature: str) -> pl.DataFrame:
    """Join ``stats[source]`` at the timestamp in ``feature_asof``: value and provenance agree."""
    key = f"{feature}{ASOF}"
    return frame.join(
        stats.select(pl.col("period_end").alias(key), pl.col(source).alias(feature)),
        on=key,
        how="left",
    )


def _with_calendar(frame: pl.DataFrame) -> pl.DataFrame:
    start_local = (pl.col("target") - PERIOD).dt.convert_time_zone(LOCAL_TIME_ZONE)
    frame = frame.with_columns(_local_start=start_local)
    years = frame.select(pl.col("_local_start").dt.year().unique())["_local_start"].to_list()
    holiday_dates = vic_public_holidays(years)
    return frame.with_columns(
        half_hour=(
            pl.col("_local_start").dt.hour() * 2 + pl.col("_local_start").dt.minute() // 30
        ).cast(pl.Int8),
        day_of_week=(pl.col("_local_start").dt.weekday() - 1).cast(pl.Int8),
        month=pl.col("_local_start").dt.month().cast(pl.Int8),
        is_public_holiday=pl.col("_local_start").dt.date().is_in(holiday_dates),
    ).drop("_local_start")
