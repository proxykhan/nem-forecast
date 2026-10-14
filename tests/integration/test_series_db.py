"""The half-hourly series computed in Postgres must equal the in-memory reference exactly.

Training and serving both read through load_half_hourly; if it ever drifted from to_half_hourly,
features (and so forecasts) would silently differ from what the backtest evaluated.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest
from sqlalchemy import Engine, text

from nem_forecast.db import apply_migrations
from nem_forecast.features.series import load_dispatch, load_half_hourly, to_half_hourly
from nem_forecast.ingest.loader import upsert_intervals
from nem_forecast.schemas import DispatchInterval, Region
from tests.synthetic import START, dispatch, series

pytestmark = pytest.mark.integration


@pytest.fixture
def engine(pg_engine: Engine) -> Engine:
    apply_migrations(pg_engine)
    rows = dispatch(series(days=3, noise_sd=40.0))
    records = [
        DispatchInterval(settlement_date=ts, region=Region.VIC1, demand_mw=mw, source_file="t")
        for ts, mw in rows.iter_rows()
    ]
    with pg_engine.begin() as conn:
        upsert_intervals(conn, records)
        # One interval missing from a period, a whole period missing, and a period in progress.
        conn.execute(
            text(
                "DELETE FROM raw_dispatch "
                "WHERE settlement_date IN (:a, :b1, :b2, :b3, :b4, :b5, :b6)"
            ),
            {
                "a": START + timedelta(hours=5, minutes=15),
                **{f"b{k}": START + timedelta(hours=20, minutes=5 * k) for k in range(1, 7)},
            },
        )
        conn.execute(
            text("DELETE FROM raw_dispatch WHERE settlement_date > :cut"),
            {"cut": START + timedelta(days=3) - timedelta(minutes=20)},
        )
    return pg_engine


def assert_same(left: pl.DataFrame, right: pl.DataFrame) -> None:
    assert left["period_end"].to_list() == right["period_end"].to_list()
    assert left["demand"].is_null().to_list() == right["demand"].is_null().to_list()
    assert ((left["demand"] - right["demand"]).abs().fill_null(0) < 1e-9).all()


def test_database_and_in_memory_series_agree(engine: Engine) -> None:
    in_db = load_half_hourly(engine, Region.VIC1)
    reference = to_half_hourly(load_dispatch(engine, Region.VIC1))

    assert_same(in_db, reference)
    assert in_db["demand"].null_count() == 2  # the gappy period and the missing one


def test_bounded_windows_agree(engine: Engine) -> None:
    start, end = START + timedelta(hours=10), START + timedelta(days=2, hours=3)

    in_db = load_half_hourly(engine, Region.VIC1, start=start, end=end)
    reference = to_half_hourly(load_dispatch(engine, Region.VIC1, start=start, end=end))

    assert_same(in_db, reference)
    assert in_db["period_end"][0] == start + timedelta(minutes=30)
    assert in_db["period_end"][-1] == end


def test_no_rows_gives_an_empty_series(engine: Engine) -> None:
    assert load_half_hourly(engine, Region.NSW1).is_empty()
