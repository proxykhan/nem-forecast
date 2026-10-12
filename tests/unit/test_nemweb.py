from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import pytest

from nem_forecast.ingest.nemweb import (
    CURRENT_DISPATCH_IS,
    USER_AGENT,
    NemwebClient,
    NemwebError,
    price_and_demand_url,
)
from nem_forecast.schemas import Region

URL = "https://nemweb.com.au/Reports/Current/DispatchIS_Reports/PUBLIC_DISPATCHIS_x.zip"


class FakeClock:
    """Monotonic clock that only moves when the client sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


Handler = Callable[[httpx.Request], httpx.Response]


def sequence(*responses: int | Exception) -> tuple[Handler, list[httpx.Request]]:
    """A handler replaying status codes (or raising exceptions) in order, plus its request log."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, content=f"body {len(seen)}".encode())

    return handle, seen


def client(
    handler: Handler, clock: FakeClock, *, cache_dir: Path | None = None, min_interval_s: float = 0
) -> NemwebClient:
    return NemwebClient(
        cache_dir=cache_dir,
        min_interval_s=min_interval_s,
        transport=httpx.MockTransport(handler),
        clock=clock,
        sleep=clock.sleep,
    )


def test_sends_identifying_user_agent() -> None:
    handler, seen = sequence(200)
    with client(handler, FakeClock()) as nemweb:
        assert nemweb.get(URL) == b"body 1"

    assert seen[0].headers["User-Agent"] == USER_AGENT
    assert "nem-forecast" in USER_AGENT


def test_retries_server_errors_with_exponential_backoff() -> None:
    clock = FakeClock()
    handler, seen = sequence(503, 502, 200)

    with client(handler, clock) as nemweb:
        body = nemweb.get(URL)

    assert body == b"body 3"
    assert len(seen) == 3
    assert clock.sleeps == [1.0, 2.0]


def test_gives_up_after_three_retries() -> None:
    clock = FakeClock()
    handler, seen = sequence(500)

    with client(handler, clock) as nemweb, pytest.raises(NemwebError, match="after 4 attempts"):
        nemweb.get(URL)

    assert len(seen) == 4
    assert clock.sleeps == [1.0, 2.0, 4.0]


def test_retries_rate_limiting_and_network_errors() -> None:
    handler, seen = sequence(429, httpx.ConnectError("reset"), 200)

    with client(handler, FakeClock()) as nemweb:
        assert nemweb.get(URL) == b"body 3"

    assert len(seen) == 3


@pytest.mark.parametrize("status", [403, 404])
def test_client_errors_fail_immediately(status: int) -> None:
    handler, seen = sequence(status)

    with client(handler, FakeClock()) as nemweb, pytest.raises(NemwebError) as excinfo:
        nemweb.get(URL)

    assert excinfo.value.status_code == status
    assert len(seen) == 1


def test_requests_are_spaced_at_least_one_second_apart() -> None:
    clock = FakeClock()
    handler, _ = sequence(200)

    with client(handler, clock, min_interval_s=1.0) as nemweb:
        for _ in range(3):
            nemweb.get(URL, use_cache=False)
        clock.now += 5  # a long pause needs no extra wait
        nemweb.get(URL, use_cache=False)

    assert clock.sleeps == [1.0, 1.0]


def test_cache_serves_repeat_requests_across_clients(tmp_path: Path) -> None:
    handler, seen = sequence(200)

    with client(handler, FakeClock(), cache_dir=tmp_path) as first:
        assert first.get(URL) == b"body 1"
        assert first.get(URL) == b"body 1"
    with client(handler, FakeClock(), cache_dir=tmp_path) as second:
        assert second.get(URL) == b"body 1"

    assert len(seen) == 1
    assert [p.name.endswith("PUBLIC_DISPATCHIS_x.zip") for p in tmp_path.iterdir()] == [True]


def test_uncached_requests_always_hit_the_network(tmp_path: Path) -> None:
    handler, seen = sequence(200)

    with client(handler, FakeClock(), cache_dir=tmp_path) as nemweb:
        nemweb.get(URL, use_cache=False)
        nemweb.get(URL, use_cache=False)

    assert len(seen) == 2
    assert list(tmp_path.iterdir()) == []


def test_failed_downloads_are_not_cached(tmp_path: Path) -> None:
    handler, _ = sequence(404)

    with client(handler, FakeClock(), cache_dir=tmp_path) as nemweb, pytest.raises(NemwebError):
        nemweb.get(URL)

    assert list(tmp_path.iterdir()) == []


def test_list_links_returns_matching_absolute_urls() -> None:
    listing = (
        '<html><body><pre><A HREF="/Reports/Current/">[To Parent Directory]</A><br>'
        ' 10/13/2026  4:35 AM  20819 <A HREF="/Reports/Current/DispatchIS_Reports/'
        'PUBLIC_DISPATCHIS_202610130440_0000000542597921.zip">PUBLIC_DISPATCHIS_...zip</A><br>'
        ' 10/13/2026  4:30 AM  20755 <A HREF="/Reports/Current/DispatchIS_Reports/'
        'PUBLIC_DISPATCHIS_202610130435_0000000542597900.zip">PUBLIC_DISPATCHIS_...zip</A><br>'
        ' 10/13/2026  4:30 AM   1024 <A HREF="/Reports/Current/DispatchIS_Reports/notes.txt">'
        "notes.txt</A></pre></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CURRENT_DISPATCH_IS
        return httpx.Response(200, text=listing)

    with client(handler, FakeClock()) as nemweb:
        links = nemweb.list_links(CURRENT_DISPATCH_IS, re.compile(r"PUBLIC_DISPATCHIS_.*\.zip$"))

    assert links == [
        CURRENT_DISPATCH_IS + "PUBLIC_DISPATCHIS_202610130435_0000000542597900.zip",
        CURRENT_DISPATCH_IS + "PUBLIC_DISPATCHIS_202610130440_0000000542597921.zip",
    ]


def test_price_and_demand_url() -> None:
    assert price_and_demand_url(date(2023, 10, 1), Region.VIC1) == (
        "https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_202310_VIC1.csv"
    )
