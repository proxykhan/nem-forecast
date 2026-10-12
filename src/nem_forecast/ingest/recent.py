"""Ingest recent 5-minute DispatchIS files from NEMWeb's Current folder.

Current keeps roughly the last 48 hours. Anything older comes from the monthly files via
``nem-forecast backfill``, which also repairs gaps if the scheduled job was down for a while.
"""

from __future__ import annotations

import logging
import re
import zipfile
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Engine

from nem_forecast.ingest.loader import Batch, LoadResult, Source, load_batch
from nem_forecast.ingest.nemweb import CURRENT_DISPATCH_IS, NemwebClient, NemwebError
from nem_forecast.ingest.parser import ParseError, find_missing_intervals, parse_file
from nem_forecast.schemas import DISPATCH_INTERVAL, NEM_TZ, DispatchInterval, Region

log = logging.getLogger(__name__)

# The timestamp in the name is the interval-ending SETTLEMENTDATE, in market time.
DISPATCH_IS_FILENAME = re.compile(r"PUBLIC_DISPATCHIS_(\d{12})_\d+\.zip$", re.IGNORECASE)


@dataclass(frozen=True)
class IngestReport:
    files: int
    files_failed: int
    records: int
    skipped_rows: int
    missing_intervals: int
    load: LoadResult


def interval_end_from_filename(name: str) -> datetime:
    match = DISPATCH_IS_FILENAME.search(name)
    if match is None:
        raise ValueError(f"not a DispatchIS file name: {name}")
    return datetime.strptime(match.group(1), "%Y%m%d%H%M").replace(tzinfo=NEM_TZ)


def ingest_dispatch_is(
    engine: Engine,
    client: NemwebClient,
    *,
    start: datetime,
    end: datetime,
    regions: Collection[Region],
) -> IngestReport:
    """Load every Current DispatchIS interval ending in ``(start, end]`` for ``regions``."""
    if end <= start:
        raise ValueError("end must be after start")

    available = sorted(
        (interval_end_from_filename(url), url)
        for url in client.list_links(CURRENT_DISPATCH_IS, DISPATCH_IS_FILENAME)
    )
    selected = [(ts, url) for ts, url in available if start < ts <= end]
    if available and start + DISPATCH_INTERVAL < available[0][0]:
        log.warning(
            "NEMWeb Current starts at %s; run `nem-forecast backfill` for anything earlier",
            available[0][0].isoformat(),
        )
    log.info("%d DispatchIS files in (%s, %s]", len(selected), start.isoformat(), end.isoformat())

    records: list[DispatchInterval] = []
    skipped = failed = 0
    for _, url in selected:
        name = url.rsplit("/", 1)[-1]
        try:
            # Files in Current never change once published, so the cache is always safe here.
            parsed = parse_file(name, client.get(url), regions=regions)
        except (NemwebError, ParseError, zipfile.BadZipFile) as exc:
            failed += 1
            log.error("%s: %s", name, exc)
            continue
        records.extend(parsed.records)
        skipped += parsed.skipped_rows

    missing = 0
    if available:
        window_start = max(start, available[0][0] - DISPATCH_INTERVAL)
        window_end = min(end, available[-1][0])
        for region in regions:
            gaps = find_missing_intervals(
                (r.settlement_date for r in records if r.region is region), window_start, window_end
            )
            if gaps:
                log.warning(
                    "%s: %d missing intervals, first %s", region, len(gaps), gaps[0].isoformat()
                )
            missing += len(gaps)

    load = load_batch(
        engine,
        Batch(
            source=Source.DISPATCH_IS,
            regions=sorted(regions),
            period_start=start,
            period_end=end,
            records=records,
            files=len(selected) - failed,
            rows_skipped=skipped,
            missing_intervals=missing,
            complete=bool(available)
            and failed == missing == 0
            and available[0][0] - DISPATCH_INTERVAL <= start
            and end <= available[-1][0],
        ),
    )
    return IngestReport(
        files=len(selected),
        files_failed=failed,
        records=len(records),
        skipped_rows=skipped,
        missing_intervals=missing,
        load=load,
    )
