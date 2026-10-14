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


def load_dispatch(
    engine: Engine,
    region: Region,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    """5-minute demand for a region in ``(start, end]`` (all of it by default).

    Columns: settlement_date (UTC), demand_mw.
    """
    conditions = ["region = :region"]
    params: dict[str, object] = {"region": region.value}
    if start is not None:
        conditions.append("settlement_date > :start")
        params["start"] = start
    if end is not None:
        conditions.append("settlement_date <= :end")
        params["end"] = end
    query = (
        "SELECT settlement_date, demand_mw FROM raw_dispatch "  # noqa: S608 - fixed fragments
        f"WHERE {' AND '.join(conditions)} ORDER BY settlement_date"
    )
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).all()
    return pl.DataFrame(
        rows,
        schema={"settlement_date": pl.Datetime("us", TIME_ZONE), "demand_mw": pl.Float64},
        orient="row",
    )


def to_half_hourly(dispatch: pl.DataFrame) -> pl.DataFrame:
    """Mean demand per half-hour period on a complete grid; incomplete periods are null.

    The reference implementation, from 5-minute rows in memory. :func:`load_half_hourly` does the
    same in the database and must agree with it (tests/integration/test_backtest_db.py).
    """
    if dispatch.is_empty():
        return pl.DataFrame(schema=SERIES_SCHEMA)
    periods = (
        dispatch.sort("settlement_date")
        .group_by_dynamic("settlement_date", every="30m", closed="right", label="right")
        .agg(pl.col("demand_mw").mean().alias("demand"), pl.len().alias("intervals"))
        .rename({"settlement_date": "period_end"})
    )
    return _on_grid(periods)


def load_half_hourly(
    engine: Engine,
    region: Region,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    """The half-hourly series for intervals in ``(start, end]``, averaged inside Postgres.

    Sends one row per half-hour instead of six, which matters over a long link to the database:
    the API's 29-day history is about 1,400 rows instead of 8,400.
    """
    conditions = ["region = :region"]
    params: dict[str, object] = {"region": region.value}
    if start is not None:
        conditions.append("settlement_date > :start")
        params["start"] = start
    if end is not None:
        conditions.append("settlement_date <= :end")
        params["end"] = end
    # date_bin labels a bin by its start; shifting by a microsecond makes each half-hour own the
    # interval ending exactly on its boundary, i.e. periods are (start, end] like AEMO's.
    query = (
        "SELECT date_bin('30 minutes', settlement_date - interval '1 microsecond', "  # noqa: S608
        "TIMESTAMPTZ '2000-01-01 00:00:00+00') + interval '30 minutes' AS period_end, "
        "avg(demand_mw) AS demand, count(*) AS intervals FROM raw_dispatch "
        f"WHERE {' AND '.join(conditions)} GROUP BY 1 ORDER BY 1"
    )
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).all()
    periods = pl.DataFrame(
        rows,
        schema={
            "period_end": pl.Datetime("us", TIME_ZONE),
            "demand": pl.Float64,
            "intervals": pl.UInt32,
        },
        orient="row",
    )
    if periods.is_empty():
        return pl.DataFrame(schema=SERIES_SCHEMA)
    return _on_grid(periods)


def _on_grid(periods: pl.DataFrame) -> pl.DataFrame:
    """Complete half-hour grid from (period_end, demand, intervals); partial periods are null."""
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
