"""Parse AEMO files into DispatchInterval records.

Two formats reach us:

* **MMS report CSV** (NEMWeb, e.g. DispatchIS). Every line starts with a record type:
  ``C`` comment, ``I`` column header for the table named by fields 2-4 (report, subtype,
  version), ``D`` data row for that table. One file interleaves several tables with different
  widths, and AEMO adds columns between versions, so columns are always looked up by name from
  the most recent ``I`` row, never by position. Demand comes from ``DISPATCH,REGIONSUM`` and
  price from ``DISPATCH,PRICE``.
* **Aggregated price-and-demand CSV** (aemo.com.au monthly files): a plain header row, then
  ``REGION,SETTLEMENTDATE,TOTALDEMAND,RRP,PERIODTYPE``.

Timestamps are AEMO market time, a fixed UTC+10 with no daylight saving (verified against the
October 2025 file in tests/unit/test_parser.py), and mark the *end* of each 5-minute interval.

Demand and price are normalised to two decimal places, the precision of AEMO's aggregated data.
DispatchIS publishes prices to five (31.64139 where the monthly file has 31.64), and without a
single canonical precision every interval seen by both feeds would look like a revision.

Malformed rows are logged and skipped; one bad row never fails the file.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from pydantic import ValidationError

from nem_forecast.schemas import DISPATCH_INTERVAL, NEM_TZ, DispatchInterval, Region

log = logging.getLogger(__name__)

AEMO_TIME_FORMAT = "%Y/%m/%d %H:%M:%S"
_DEMAND_TABLE = ("DISPATCH", "REGIONSUM")
_PRICE_TABLE = ("DISPATCH", "PRICE")
_PRICE_AND_DEMAND_COLUMNS = frozenset({"REGION", "SETTLEMENTDATE", "TOTALDEMAND", "RRP"})
# The physical (intervention) run duplicates each interval; prices come from the pricing run.
_PRICING_RUN = "0"
_CENT = Decimal("0.01")

_Key = tuple[datetime, Region]


class ParseError(ValueError):
    """The file as a whole is unusable (not a format we recognise)."""


@dataclass(frozen=True)
class ParseResult:
    records: list[DispatchInterval]
    skipped_rows: int = 0


def parse_aemo_time(value: str) -> datetime:
    return datetime.strptime(value, AEMO_TIME_FORMAT).replace(tzinfo=NEM_TZ)


def parse_amount(value: str | None) -> float:
    """MW or $/MWh at two decimal places, rounded half-up from the published text."""
    if value is None or not value.strip():
        raise ValueError("empty value")
    try:
        return float(Decimal(value.strip()).quantize(_CENT, rounding=ROUND_HALF_UP))
    except InvalidOperation as exc:
        raise ValueError(f"not a number: {value!r}") from exc


def parse_file(
    source_file: str,
    content: bytes,
    *,
    regions: Collection[Region] | None = None,
    ingested_at: datetime | None = None,
) -> ParseResult:
    """Parse a zipped or bare AEMO CSV, keeping only ``regions`` (all regions if None)."""
    ingested_at = ingested_at or datetime.now(UTC)
    records: list[DispatchInterval] = []
    skipped = 0
    for member, text in _csv_texts(source_file, content):
        result = parse_csv_text(text, source_file, regions=regions, ingested_at=ingested_at)
        log.debug("%s: %d records, %d skipped", member, len(result.records), result.skipped_rows)
        records.extend(result.records)
        skipped += result.skipped_rows
    return ParseResult(records, skipped)


def parse_csv_text(
    text: str,
    source_file: str,
    *,
    regions: Collection[Region] | None = None,
    ingested_at: datetime | None = None,
) -> ParseResult:
    ingested_at = ingested_at or datetime.now(UTC)
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    if first_line.startswith(("C,", "I,")):
        demand, price, skipped = _read_mms_report(text, source_file)
    elif "SETTLEMENTDATE" in first_line.upper():
        demand, price, skipped = _read_price_and_demand(text, source_file)
    else:
        raise ParseError(f"{source_file}: unrecognised format, first line {first_line[:80]!r}")

    records: list[DispatchInterval] = []
    for (settlement_date, region), demand_mw in sorted(demand.items()):
        if regions is not None and region not in regions:
            continue
        try:
            records.append(
                DispatchInterval(
                    settlement_date=settlement_date,
                    region=region,
                    demand_mw=demand_mw,
                    rrp=price.get((settlement_date, region)),
                    source_file=source_file,
                    ingested_at=ingested_at,
                )
            )
        except ValidationError as exc:
            skipped += 1
            log.warning(
                "%s: skipped %s %s: %s",
                source_file,
                region,
                settlement_date.isoformat(),
                exc.errors()[0]["msg"],
            )
    return ParseResult(records, skipped)


def find_missing_intervals(
    timestamps: Iterable[datetime], start: datetime, end: datetime
) -> list[datetime]:
    """Interval-ending times in ``(start, end]`` that have no record.

    ``start`` need not sit on the 5-minute grid (a window of "the last 24 hours" usually doesn't);
    expected intervals are always grid points.
    """
    present = set(timestamps)
    missing: list[datetime] = []
    since_midnight = start - start.replace(hour=0, minute=0, second=0, microsecond=0)
    expected = start - since_midnight % DISPATCH_INTERVAL + DISPATCH_INTERVAL
    while expected <= end:
        if expected not in present:
            missing.append(expected)
        expected += DISPATCH_INTERVAL
    return missing


def _csv_texts(source_file: str, content: bytes) -> Iterator[tuple[str, str]]:
    if not zipfile.is_zipfile(io.BytesIO(content)):
        yield source_file, _decode(content)
        return
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.infolist():
            if member.filename.lower().endswith(".csv"):
                yield member.filename, _decode(archive.read(member))
            else:
                log.warning("%s: ignoring non-CSV member %s", source_file, member.filename)


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8-sig", errors="replace")


def _read_mms_report(
    text: str, source_file: str
) -> tuple[dict[_Key, float], dict[_Key, float], int]:
    headers: dict[tuple[str, str], list[str]] = {}
    demand: dict[_Key, float] = {}
    price: dict[_Key, float] = {}
    skipped = 0

    for line_no, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        if not row or row[0] == "C":
            continue
        if row[0] == "I" and len(row) > 4:
            headers[(row[1], row[2])] = row[4:]
            continue
        if row[0] != "D" or len(row) < 4:
            skipped += 1
            log.warning("%s:%d: unknown record type %r, skipped", source_file, line_no, row[0])
            continue

        table = (row[1], row[2])
        if table not in (_DEMAND_TABLE, _PRICE_TABLE):
            continue
        columns = headers.get(table)
        if columns is None or len(row) - 4 != len(columns):
            skipped += 1
            expected = "no I row yet" if columns is None else f"{len(columns)} columns"
            log.warning(
                "%s:%d: %s row has %d columns (%s), skipped",
                source_file,
                line_no,
                ",".join(table),
                len(row) - 4,
                expected,
            )
            continue

        values = dict(zip(columns, row[4:], strict=True))
        try:
            if values.get("INTERVENTION", _PRICING_RUN) != _PRICING_RUN:
                continue
            key = (parse_aemo_time(values["SETTLEMENTDATE"]), Region(values["REGIONID"]))
            if table == _DEMAND_TABLE:
                demand[key] = parse_amount(values["TOTALDEMAND"])
            else:
                price[key] = parse_amount(values["RRP"])
        except (KeyError, ValueError) as exc:
            skipped += 1
            log.warning("%s:%d: %s, skipped", source_file, line_no, exc)

    return demand, price, skipped


def _read_price_and_demand(
    text: str, source_file: str
) -> tuple[dict[_Key, float], dict[_Key, float], int]:
    reader = csv.DictReader(io.StringIO(text))
    missing_columns = _PRICE_AND_DEMAND_COLUMNS - set(reader.fieldnames or ())
    if missing_columns:
        raise ParseError(f"{source_file}: missing columns {sorted(missing_columns)}")

    demand: dict[_Key, float] = {}
    price: dict[_Key, float] = {}
    skipped = 0
    for row in reader:
        try:
            key = (parse_aemo_time(row["SETTLEMENTDATE"]), Region(row["REGION"]))
            demand[key] = parse_amount(row["TOTALDEMAND"])
            if row["RRP"]:
                price[key] = parse_amount(row["RRP"])
        except (KeyError, TypeError, ValueError) as exc:
            skipped += 1
            log.warning("%s:%d: %s, skipped", source_file, reader.line_num, exc)
    return demand, price, skipped
