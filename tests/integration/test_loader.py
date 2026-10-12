"""raw_dispatch upserts against real PostgreSQL: idempotency and revision handling."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import Engine, text

from nem_forecast.db import apply_migrations
from nem_forecast.ingest.loader import Batch, LoadResult, Source, load_batch
from nem_forecast.ingest.parser import parse_file
from nem_forecast.schemas import NEM_TZ, DispatchInterval, Region
from tests.aemo_files import (
    DISPATCH_IS_ZIP,
    DST_DAY_CSV,
    VIC1_DEMAND,
    dispatch_is_text,
    replace_once,
    with_vic1_demand,
    zipped,
)

pytestmark = pytest.mark.integration

INTERVAL_END = datetime(2026, 10, 13, 4, 40, tzinfo=NEM_TZ)
REVISED_FILE = "PUBLIC_DISPATCHIS_202610130440_0000000542597999.zip"


@pytest.fixture
def engine(pg_engine: Engine) -> Engine:
    apply_migrations(pg_engine)
    return pg_engine


def batch(records: list[DispatchInterval], *, source: Source = Source.DISPATCH_IS) -> Batch:
    return Batch(
        source=source,
        regions=sorted({r.region for r in records}),
        period_start=INTERVAL_END - timedelta(minutes=5),
        period_end=INTERVAL_END,
        records=records,
        files=1,
        rows_skipped=0,
        missing_intervals=0,
        complete=True,
    )


def load_text(engine: Engine, csv_text: str, name: str = DISPATCH_IS_ZIP) -> LoadResult:
    return load_batch(engine, batch(parse_file(name, zipped(csv_text)).records))


def rows(engine: Engine, sql: str, **params: Any) -> list[Any]:
    with engine.connect() as conn:
        return list(conn.execute(text(sql), params).all())


def snapshot(engine: Engine) -> list[Any]:
    return rows(engine, "SELECT * FROM raw_dispatch ORDER BY region")


def test_ingesting_the_same_file_twice_changes_nothing(engine: Engine) -> None:
    first = load_text(engine, dispatch_is_text())
    before = snapshot(engine)

    second = load_text(engine, dispatch_is_text())

    assert first == LoadResult(inserted=5, updated=0, unchanged=0)
    assert second == LoadResult(inserted=0, updated=0, unchanged=5)
    assert snapshot(engine) == before  # not even updated_at or source_file moves
    assert rows(engine, "SELECT count(*) FROM raw_dispatch") == [(5,)]
    assert rows(engine, "SELECT count(*) FROM raw_dispatch_history") == [(0,)]


def test_revised_demand_bumps_revision_and_keeps_the_old_value(engine: Engine) -> None:
    load_text(engine, dispatch_is_text())

    result = load_text(engine, with_vic1_demand(dispatch_is_text(), "4702.15"), REVISED_FILE)

    assert result == LoadResult(inserted=0, updated=1, unchanged=4)
    assert rows(
        engine,
        "SELECT revision, demand_mw, source_file FROM raw_dispatch WHERE region = 'VIC1'",
    ) == [(2, 4702.15, REVISED_FILE)]
    assert rows(
        engine,
        "SELECT revision, demand_mw, source_file, superseded_by_file FROM raw_dispatch_history",
    ) == [(1, float(VIC1_DEMAND), DISPATCH_IS_ZIP, REVISED_FILE)]
    assert rows(engine, "SELECT count(*) FROM raw_dispatch WHERE revision > 1") == [(1,)]


def test_each_further_revision_adds_one_history_row(engine: Engine) -> None:
    load_text(engine, dispatch_is_text())
    load_text(engine, with_vic1_demand(dispatch_is_text(), "4702.15"), REVISED_FILE)
    load_text(engine, with_vic1_demand(dispatch_is_text(), "4710.00"), REVISED_FILE)
    load_text(engine, with_vic1_demand(dispatch_is_text(), "4710.00"), REVISED_FILE)

    assert rows(engine, "SELECT revision FROM raw_dispatch WHERE region = 'VIC1'") == [(3,)]
    assert rows(
        engine, "SELECT revision, demand_mw FROM raw_dispatch_history ORDER BY revision"
    ) == [(1, float(VIC1_DEMAND)), (2, 4702.15)]


def test_revised_price_is_a_revision_too(engine: Engine) -> None:
    text_ = dispatch_is_text()
    load_text(engine, text_)
    vic1_price = next(
        line
        for line in text_.splitlines()
        if line.startswith("D,DISPATCH,PRICE") and ",VIC1," in line
    )

    result = load_text(
        engine, replace_once(text_, vic1_price, vic1_price.replace(",0,0.01,", ",0,95.5,", 1))
    )

    assert result.updated == 1
    assert rows(engine, "SELECT rrp, revision FROM raw_dispatch WHERE region = 'VIC1'") == [
        (95.5, 2)
    ]


def test_missing_price_never_erases_a_known_one(engine: Engine) -> None:
    text_ = dispatch_is_text()
    load_text(engine, text_)
    without_prices = "\n".join(
        line for line in text_.splitlines() if not line.startswith("D,DISPATCH,PRICE")
    )

    result = load_text(engine, without_prices)

    assert result == LoadResult(inserted=0, updated=0, unchanged=5)
    assert rows(engine, "SELECT rrp FROM raw_dispatch WHERE region = 'VIC1'") == [(0.01,)]


def test_duplicate_keys_in_one_batch_keep_the_last(engine: Engine) -> None:
    original = parse_file(DISPATCH_IS_ZIP, zipped(dispatch_is_text()), regions={Region.VIC1})
    revised = parse_file(
        REVISED_FILE,
        zipped(with_vic1_demand(dispatch_is_text(), "4702.15")),
        regions={Region.VIC1},
    )

    result = load_batch(engine, batch([*original.records, *revised.records]))

    assert result == LoadResult(inserted=1, updated=0, unchanged=0)
    assert rows(engine, "SELECT demand_mw, revision FROM raw_dispatch") == [(4702.15, 1)]


def test_every_batch_is_logged_with_its_counts(engine: Engine) -> None:
    load_text(engine, dispatch_is_text())
    load_text(engine, with_vic1_demand(dispatch_is_text(), "4702.15"), REVISED_FILE)

    assert rows(
        engine,
        "SELECT source, regions, files, rows_parsed, rows_inserted, rows_updated, rows_unchanged "
        "FROM ingest_log ORDER BY id",
    ) == [
        ("dispatch_is", "NSW1,QLD1,SA1,TAS1,VIC1", 1, 5, 5, 0, 0),
        ("dispatch_is", "NSW1,QLD1,SA1,TAS1,VIC1", 1, 5, 0, 1, 4),
    ]


def test_empty_batch_is_still_logged(engine: Engine) -> None:
    assert load_batch(engine, batch([])) == LoadResult()
    assert rows(engine, "SELECT rows_parsed FROM ingest_log") == [(0,)]


def test_real_monthly_file_across_daylight_saving_loads_every_interval(engine: Engine) -> None:
    records = parse_file(DST_DAY_CSV.name, DST_DAY_CSV.read_bytes()).records

    result = load_batch(engine, batch(records, source=Source.PRICE_AND_DEMAND))

    assert result.inserted == 289
    assert rows(
        engine,
        "SELECT count(DISTINCT settlement_date), "
        "max(settlement_date) - min(settlement_date) FROM raw_dispatch",
    ) == [(289, timedelta(hours=24))]


def test_switching_feed_is_not_mistaken_for_a_revision(engine: Engine) -> None:
    """Found on real data: DispatchIS prices carry 5 decimals, the monthly file 2.

    Before prices were normalised to cents, 57 of 230 overlapping VIC1 intervals in a real
    24-hour run were recorded as revisions that were only rounding.
    """
    load_text(engine, dispatch_is_text())
    monthly = (
        "REGION,SETTLEMENTDATE,TOTALDEMAND,RRP,PERIODTYPE\n"
        f"VIC1,2026/10/13 04:40:00,{VIC1_DEMAND}0,0.01,TRADE\n"
        "QLD1,2026/10/13 04:40:00,5954.92,152.67,TRADE\n"
    )

    result = load_batch(
        engine,
        batch(parse_file("PRICE_AND_DEMAND_202610_VIC1.csv", monthly.encode()).records),
    )

    assert result == LoadResult(inserted=0, updated=0, unchanged=2)
    assert rows(engine, "SELECT count(*) FROM raw_dispatch_history") == [(0,)]
