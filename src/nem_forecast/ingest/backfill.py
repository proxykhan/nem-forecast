"""Backfill raw_dispatch from AEMO's monthly price-and-demand files, one month at a time.

Each month is fetched, parsed and committed (data plus its ingest_log row) in one transaction,
so an interrupted run loses at most the month in flight. Re-running skips every month that
ingest_log marks complete: the month had ended and no interval was missing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum

from sqlalchemy import Engine, text

from nem_forecast.ingest.loader import Batch, LoadResult, Source, load_batch
from nem_forecast.ingest.nemweb import NemwebClient, NemwebError, price_and_demand_url
from nem_forecast.ingest.parser import ParseError, find_missing_intervals, parse_file
from nem_forecast.schemas import NEM_TZ, Region

log = logging.getLogger(__name__)

# AEMO appends to the current month's file daily. A few days after month end the file is
# final, and only then is it safe to serve from the local cache.
_FINAL_AFTER = timedelta(days=3)

_COMPLETED_MONTHS = """
SELECT DISTINCT period_start FROM ingest_log
WHERE source = :source AND regions = :region AND complete
"""


class MonthStatus(StrEnum):
    LOADED = "loaded"
    SKIPPED = "skipped"  # already complete in the database
    UNAVAILABLE = "unavailable"  # AEMO has not published it yet
    FAILED = "failed"


@dataclass(frozen=True)
class MonthOutcome:
    month: date
    status: MonthStatus
    records: int = 0
    missing_intervals: int = 0
    load: LoadResult | None = None
    error: str | None = None


def default_months(today: date, years: int) -> tuple[date, date]:
    """First and last month start covering ``years`` full years up to the current month."""
    last = today.replace(day=1)
    return last.replace(year=last.year - years), last


def month_starts(first: date, last: date) -> list[date]:
    months: list[date] = []
    month = first.replace(day=1)
    while month <= last:
        months.append(month)
        month = _next_month(month)
    return months


def month_period(month: date) -> tuple[datetime, datetime]:
    """``(start, end]`` of a month in market time; the first interval ends at 00:05 on day 1."""
    following = _next_month(month)
    return (
        datetime(month.year, month.month, 1, tzinfo=NEM_TZ),
        datetime(following.year, following.month, 1, tzinfo=NEM_TZ),
    )


def completed_months(engine: Engine, region: Region) -> set[datetime]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(_COMPLETED_MONTHS),
            {"source": Source.PRICE_AND_DEMAND.value, "region": region.value},
        ).scalars()
        return set(rows)


def backfill(
    engine: Engine,
    client: NemwebClient,
    *,
    region: Region,
    first_month: date,
    last_month: date,
    force: bool = False,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> list[MonthOutcome]:
    done = set() if force else completed_months(engine, region)
    outcomes: list[MonthOutcome] = []
    for month in month_starts(first_month, last_month):
        period_start, _ = month_period(month)
        if period_start in done:
            log.info("%s %s: already complete, skipping", region, f"{month:%Y-%m}")
            outcomes.append(MonthOutcome(month, MonthStatus.SKIPPED))
            continue
        outcomes.append(_load_month(engine, client, region, month, now()))
    return outcomes


def _load_month(
    engine: Engine, client: NemwebClient, region: Region, month: date, now: datetime
) -> MonthOutcome:
    period_start, period_end = month_period(month)
    if period_start >= now:
        return MonthOutcome(month, MonthStatus.UNAVAILABLE, error="month has not started")
    month_over = now >= period_end
    url = price_and_demand_url(month, region)
    name = url.rsplit("/", 1)[-1]

    try:
        content = client.get(url, use_cache=now >= period_end + _FINAL_AFTER)
        parsed = parse_file(name, content, regions=[region])
    except NemwebError as exc:
        if exc.status_code == 404 and not month_over:
            log.info("%s: not published yet", name)
            return MonthOutcome(month, MonthStatus.UNAVAILABLE, error=str(exc))
        log.error("%s: %s", name, exc)
        return MonthOutcome(month, MonthStatus.FAILED, error=str(exc))
    except ParseError as exc:
        log.error("%s", exc)
        return MonthOutcome(month, MonthStatus.FAILED, error=str(exc))

    records = [r for r in parsed.records if period_start < r.settlement_date <= period_end]
    latest = max((r.settlement_date for r in records), default=period_start)
    missing = find_missing_intervals(
        (r.settlement_date for r in records),
        period_start,
        period_end if month_over else latest,
    )
    if missing:
        log.warning("%s: %d missing intervals, first %s", name, len(missing), missing[0])

    load = load_batch(
        engine,
        Batch(
            source=Source.PRICE_AND_DEMAND,
            regions=[region],
            period_start=period_start,
            period_end=period_end,
            records=records,
            files=1,
            rows_skipped=parsed.skipped_rows,
            missing_intervals=len(missing),
            complete=month_over and not missing,
        ),
    )
    return MonthOutcome(
        month,
        MonthStatus.LOADED,
        records=len(records),
        missing_intervals=len(missing),
        load=load,
    )


def _next_month(month: date) -> date:
    return date(month.year + month.month // 12, month.month % 12 + 1, 1)
