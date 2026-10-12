"""Builders for AEMO-format test files, based on the real trimmed fixtures in tests/fixtures."""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

from nem_forecast.ingest.backfill import month_period
from nem_forecast.ingest.nemweb import CURRENT_DISPATCH_IS, NemwebClient
from nem_forecast.schemas import DISPATCH_INTERVAL, NEM_TZ

FIXTURES = Path(__file__).parent / "fixtures" / "nemweb"
DISPATCH_IS_NAME = "PUBLIC_DISPATCHIS_202610130440_0000000542597921"
DISPATCH_IS_ZIP = f"{DISPATCH_IS_NAME}.zip"
DISPATCH_IS_TIME = "2026/10/13 04:40:00"
VIC1_DEMAND = "4660.9"
DST_DAY_CSV = FIXTURES / "PRICE_AND_DEMAND_202510_VIC1_dst_day.csv"


def dispatch_is_text() -> str:
    """Real DispatchIS report for the interval ending 2026-10-13 04:40, trimmed to 18 lines."""
    return (FIXTURES / f"{DISPATCH_IS_NAME}.CSV").read_text(encoding="utf-8")


def zipped(text: str, member: str = f"{DISPATCH_IS_NAME}.CSV") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return buffer.getvalue()


def replace_once(text: str, old: str, new: str) -> str:
    """str.replace that fails loudly if the fixture no longer contains ``old``."""
    assert text.count(old) == 1, f"expected exactly one {old!r} in fixture"
    return text.replace(old, new)


def with_vic1_demand(text: str, demand: str) -> str:
    line = next(
        line
        for line in text.splitlines()
        if line.startswith("D,DISPATCH,REGIONSUM") and ",VIC1," in line
    )
    return replace_once(text, line, line.replace(f",0,{VIC1_DEMAND},", f",0,{demand},", 1))


def shifted_dispatch_is(minutes: int) -> tuple[str, bytes]:
    """The fixture moved ``minutes`` later: a (file name, zip bytes) pair for another interval."""
    original = datetime(2026, 10, 13, 4, 40, tzinfo=NEM_TZ)
    moved = original + timedelta(minutes=minutes)
    text = dispatch_is_text().replace(DISPATCH_IS_TIME, f"{moved:%Y/%m/%d %H:%M:%S}")
    name = f"PUBLIC_DISPATCHIS_{moved:%Y%m%d%H%M}_0000000542597921"
    return f"{name}.zip", zipped(text, f"{name}.CSV")


def monthly_price_and_demand(
    month: date, region: str = "VIC1", *, until: datetime | None = None
) -> str:
    """Synthetic monthly price-and-demand file in AEMO's layout, complete unless ``until``."""
    start, end = month_period(month)
    last = min(end, until) if until else end
    lines = ["REGION,SETTLEMENTDATE,TOTALDEMAND,RRP,PERIODTYPE"]
    ts = start + DISPATCH_INTERVAL
    i = 0
    while ts <= last:
        lines.append(f"{region},{ts:%Y/%m/%d %H:%M:%S},{4000 + i % 288:.2f},{50 + i % 7:.2f},TRADE")
        ts += DISPATCH_INTERVAL
        i += 1
    return "\n".join(lines) + "\n"


class FakeAemo:
    """In-memory stand-in for NEMWeb and aemo.com.au, served through httpx.MockTransport.

    ``files`` maps URL to body, or to an int to answer with that HTTP status instead. The
    DispatchIS Current listing is generated from whichever DispatchIS URLs are registered.
    """

    def __init__(self) -> None:
        self.files: dict[str, bytes | int] = {}
        self.requests: list[str] = []

    def add_dispatch_is(self, name: str, body: bytes | int) -> None:
        self.files[CURRENT_DISPATCH_IS + name] = body

    def client(self, cache_dir: Path | None = None) -> NemwebClient:
        return NemwebClient(
            cache_dir=cache_dir,
            min_interval_s=0,
            transport=httpx.MockTransport(self._handle),
            sleep=lambda _: None,
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url == CURRENT_DISPATCH_IS:
            links = "".join(
                f'<A HREF="{u.removeprefix("https://nemweb.com.au")}">{u}</A><br>'
                for u in self.files
                if u.startswith(CURRENT_DISPATCH_IS)
            )
            return httpx.Response(200, text=f"<html><pre>{links}</pre></html>")
        body = self.files.get(url, 404)
        if isinstance(body, int):
            return httpx.Response(body)
        return httpx.Response(200, content=body)
