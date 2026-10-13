"""The half-hourly demand series every feature and model starts from.

raw_dispatch holds 5-minute intervals. A half-hour period ending at HH:30 covers the six
intervals ending HH:05 .. HH:30, and its demand is their mean. A period with fewer than six
intervals is left null rather than averaged from a partial hour.

Timestamps stay in UTC inside polars; AEMO market time is a fixed UTC+10, so half-hour
boundaries are identical in both.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import polars as pl
from sqlalchemy import Engine, text

from nem_forecast.schemas import Region

log = logging.getLogger(__name__)

PERIOD = timedelta(minutes=30)
INTERVALS_PER_PERIOD = 6
TIME_ZONE = "UTC"

SERIES_SCHEMA = pl.Schema({"period_end": pl.Datetime("us", TIME_ZONE), "demand": pl.Float64()})


def span(series: pl.DataFrame) -> tuple[datetime, datetime]:
    """First and last period_end of a non-empty half-hourly series."""
    first, last = series["period_end"].min(), series["period_end"].max()
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        raise ValueError("series is empty")
    return first, last


def load_dispatch(engine: Engine, region: Region) -> pl.DataFrame:
    """All 5-minute demand for a region: columns settlement_date (UTC), demand_mw."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT settlement_date, demand_mw FROM raw_dispatch "
                "WHERE region = :region ORDER BY settlement_date"
            ),
            {"region": region.value},
        ).all()
    return pl.DataFrame(
        rows,
        schema={"settlement_date": pl.Datetime("us", TIME_ZONE), "demand_mw": pl.Float64},
        orient="row",
    )


def to_half_hourly(dispatch: pl.DataFrame) -> pl.DataFrame:
    """Mean demand per half-hour period on a complete grid; incomplete periods are null."""
    if dispatch.is_empty():
        return pl.DataFrame(schema=SERIES_SCHEMA)
    periods = (
        dispatch.sort("settlement_date")
        .group_by_dynamic("settlement_date", every="30m", closed="right", label="right")
        .agg(pl.col("demand_mw").mean().alias("demand"), pl.len().alias("intervals"))
        .rename({"settlement_date": "period_end"})
    )
    first, last = span(periods)
    grid = pl.DataFrame(
        {
            "period_end": pl.datetime_range(
                first,
                last,
                interval="30m",
                time_zone=TIME_ZONE,
                eager=True,
            )
        }
    )
    series = grid.join(periods, on="period_end", how="left").select(
        "period_end",
        pl.when(pl.col("intervals") == INTERVALS_PER_PERIOD)
        .then(pl.col("demand"))
        .otherwise(None)
        .alias("demand"),
    )
    # A partial final period is still in progress, not missing: drop it.
    if series["demand"][-1] is None:
        series = series.head(-1)
    incomplete = series["demand"].null_count()
    if incomplete:
        log.warning("%d of %d half-hour periods incomplete; left null", incomplete, len(series))
    return series.cast(SERIES_SCHEMA)
