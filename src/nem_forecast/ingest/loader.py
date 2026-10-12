"""Idempotent upsert of DispatchInterval records into raw_dispatch, keeping revision history.

For each incoming (settlement_date, region):

* new                       -> inserted at revision 1
* same demand and price     -> untouched (re-running a file changes nothing)
* demand or price differs   -> old row copied to raw_dispatch_history, new value written,
                               revision incremented

A missing price (``rrp`` NULL) in the incoming row means "not reported", not "price removed",
so it never triggers a revision or overwrites a known price.

Each batch commits atomically together with its ingest_log row.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import psycopg
from sqlalchemy import Connection, Engine, text

from nem_forecast.schemas import DispatchInterval, Region

log = logging.getLogger(__name__)

# Serialises concurrent loaders (scheduled ingest vs. a manual backfill) so revision
# detection and the history insert see a consistent table.
_LOCK_KEY = 7_201_993_502

_CREATE_STAGING = """
CREATE TEMP TABLE IF NOT EXISTS _incoming_dispatch (
    settlement_date TIMESTAMPTZ      NOT NULL,
    region          TEXT             NOT NULL,
    demand_mw       DOUBLE PRECISION NOT NULL,
    rrp             DOUBLE PRECISION,
    source_file     TEXT             NOT NULL,
    ingested_at     TIMESTAMPTZ      NOT NULL
) ON COMMIT DROP
"""

_COPY_STAGING = (
    "COPY _incoming_dispatch (settlement_date, region, demand_mw, rrp, source_file, ingested_at) "
    "FROM STDIN"
)


def _changed(stored: str, incoming: str) -> str:
    """SQL predicate shared by the history insert and the upsert, so they always agree."""
    return (
        f"({stored}.demand_mw IS DISTINCT FROM {incoming}.demand_mw "
        f"OR ({incoming}.rrp IS NOT NULL AND {stored}.rrp IS DISTINCT FROM {incoming}.rrp))"
    )


_ARCHIVE_SUPERSEDED = f"""
INSERT INTO raw_dispatch_history (
    raw_dispatch_id, settlement_date, region, revision, demand_mw, rrp, source_file,
    valid_from, superseded_by_file
)
SELECT r.id, r.settlement_date, r.region, r.revision, r.demand_mw, r.rrp, r.source_file,
       r.updated_at, i.source_file
FROM raw_dispatch AS r
JOIN _incoming_dispatch AS i USING (settlement_date, region)
WHERE {_changed("r", "i")}
"""  # noqa: S608 - constant SQL, no user input

_UPSERT = f"""
INSERT INTO raw_dispatch AS r (
    settlement_date, region, demand_mw, rrp, source_file, ingested_at, updated_at
)
SELECT settlement_date, region, demand_mw, rrp, source_file, ingested_at, ingested_at
FROM _incoming_dispatch
ON CONFLICT (settlement_date, region) DO UPDATE SET
    demand_mw   = EXCLUDED.demand_mw,
    rrp         = COALESCE(EXCLUDED.rrp, r.rrp),
    source_file = EXCLUDED.source_file,
    revision    = r.revision + 1,
    updated_at  = now()
WHERE {_changed("r", "EXCLUDED")}
RETURNING (xmax = 0) AS inserted
"""  # noqa: S608 - constant SQL, no user input

_INSERT_LOG = """
INSERT INTO ingest_log (
    source, regions, period_start, period_end, files, rows_parsed, rows_skipped,
    rows_inserted, rows_updated, rows_unchanged, missing_intervals, complete
) VALUES (
    :source, :regions, :period_start, :period_end, :files, :rows_parsed, :rows_skipped,
    :rows_inserted, :rows_updated, :rows_unchanged, :missing_intervals, :complete
)
"""


class Source(StrEnum):
    DISPATCH_IS = "dispatch_is"
    PRICE_AND_DEMAND = "price_and_demand"


@dataclass(frozen=True)
class LoadResult:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0


@dataclass(frozen=True)
class Batch:
    """Records from one ingest unit (a backfill month, or one scheduled run) plus provenance."""

    source: Source
    regions: Sequence[Region]
    period_start: datetime
    period_end: datetime
    records: Sequence[DispatchInterval]
    files: int
    rows_skipped: int
    missing_intervals: int
    complete: bool  # the period has ended and every interval in it arrived


def load_batch(engine: Engine, batch: Batch) -> LoadResult:
    """Upsert a batch and record it in ingest_log, in one transaction."""
    with engine.begin() as conn:
        result = upsert_intervals(conn, batch.records)
        conn.execute(
            text(_INSERT_LOG),
            {
                "source": batch.source.value,
                "regions": ",".join(sorted(r.value for r in batch.regions)),
                "period_start": batch.period_start,
                "period_end": batch.period_end,
                "files": batch.files,
                "rows_parsed": len(batch.records),
                "rows_skipped": batch.rows_skipped,
                "rows_inserted": result.inserted,
                "rows_updated": result.updated,
                "rows_unchanged": result.unchanged,
                "missing_intervals": batch.missing_intervals,
                "complete": batch.complete,
            },
        )
    log.info(
        "%s %s..%s: %d inserted, %d revised, %d unchanged",
        batch.source.value,
        batch.period_start.isoformat(),
        batch.period_end.isoformat(),
        result.inserted,
        result.updated,
        result.unchanged,
    )
    return result


def coverage(engine: Engine, region: Region) -> tuple[int, datetime | None, datetime | None]:
    """Row count and first/last settlement_date held for a region."""
    with engine.connect() as conn:
        count, first, last = conn.execute(
            text(
                "SELECT count(*), min(settlement_date), max(settlement_date) "
                "FROM raw_dispatch WHERE region = :region"
            ),
            {"region": region.value},
        ).one()
    return int(count), first, last


def upsert_intervals(conn: Connection, records: Iterable[DispatchInterval]) -> LoadResult:
    """Upsert into raw_dispatch on an open PostgreSQL transaction (the caller commits)."""
    rows = _latest_per_key(records)
    if not rows:
        return LoadResult()

    conn.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _LOCK_KEY})
    conn.execute(text(_CREATE_STAGING))
    conn.execute(text("TRUNCATE _incoming_dispatch"))
    _copy_to_staging(conn, rows)
    conn.execute(text(_ARCHIVE_SUPERSEDED))
    inserted_flags = conn.execute(text(_UPSERT)).scalars().all()

    inserted = sum(1 for flag in inserted_flags if flag)
    updated = len(inserted_flags) - inserted
    return LoadResult(inserted=inserted, updated=updated, unchanged=len(rows) - inserted - updated)


def _latest_per_key(records: Iterable[DispatchInterval]) -> list[DispatchInterval]:
    """One row per (settlement_date, region), last one wins; ON CONFLICT can't touch a row twice."""
    latest: dict[tuple[datetime, Region], DispatchInterval] = {}
    for record in records:
        latest[(record.settlement_date, record.region)] = record
    return list(latest.values())


def _copy_to_staging(conn: Connection, rows: Sequence[DispatchInterval]) -> None:
    driver = conn.connection.driver_connection
    if not isinstance(driver, psycopg.Connection):
        raise TypeError("the loader needs a PostgreSQL connection through psycopg 3")
    with driver.cursor() as cursor, cursor.copy(_COPY_STAGING) as copy:
        for row in rows:
            copy.write_row(
                (
                    row.settlement_date,
                    row.region.value,
                    row.demand_mw,
                    row.rrp,
                    row.source_file,
                    row.ingested_at,
                )
            )
