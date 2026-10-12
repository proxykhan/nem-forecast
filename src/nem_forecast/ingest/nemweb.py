"""HTTP client for AEMO's public data: NEMWeb reports and the monthly price-and-demand files.

Polite by construction: one request per second, an identifying User-Agent, bounded exponential
backoff on server errors, and a local cache so development re-runs never re-download a file.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from types import TracebackType
from typing import Self
from urllib.parse import urljoin, urlsplit

import httpx

from nem_forecast import __version__
from nem_forecast.schemas import Region

log = logging.getLogger(__name__)

CURRENT_DISPATCH_IS = "https://nemweb.com.au/Reports/Current/DispatchIS_Reports/"
_PRICE_AND_DEMAND = (
    "https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{month}_{region}.csv"
)
USER_AGENT = (
    f"Mozilla/5.0 (compatible; nem-forecast/{__version__}; "
    "+https://github.com/proxykhan/nem-forecast)"
)
DEFAULT_CACHE_DIR = Path(".cache/nemweb")

_HREF = re.compile(r'href="([^"]+)"', re.IGNORECASE)


def price_and_demand_url(month: date, region: Region) -> str:
    return _PRICE_AND_DEMAND.format(month=f"{month:%Y%m}", region=region.value)


class NemwebError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class NemwebClient:
    """Rate-limited, retrying, caching GET client. Use as a context manager."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
        min_interval_s: float = 1.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout_s,
            follow_redirects=True,
            transport=transport,
        )
        self._cache_dir = cache_dir
        self._min_interval_s = min_interval_s
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._clock = clock
        self._sleep = sleep
        self._last_request_at: float | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def get(self, url: str, *, use_cache: bool = True) -> bytes:
        """Body of ``url``. Cached copies are served without touching the network.

        Pass ``use_cache=False`` for anything that can still change, such as the current
        month's price-and-demand file.
        """
        cache_path = self._cache_path(url) if use_cache else None
        if cache_path is not None and cache_path.exists():
            log.debug("cache hit %s", url)
            return cache_path.read_bytes()
        content = self._fetch(url)
        if cache_path is not None:
            _write_atomically(cache_path, content)
        return content

    def list_links(self, directory_url: str, filename: re.Pattern[str]) -> list[str]:
        """Absolute URLs of the files in a NEMWeb directory listing whose name matches."""
        html = self._fetch(directory_url).decode("utf-8", errors="replace")
        urls = {urljoin(directory_url, href) for href in _HREF.findall(html)}
        return sorted(url for url in urls if filename.search(_basename(url)))

    def _fetch(self, url: str) -> bytes:
        for attempt in range(self._max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = self._http.get(url)
            except httpx.TransportError as exc:
                reason = f"{type(exc).__name__}: {exc}"
            else:
                if response.is_success:
                    return response.content
                if not _is_retryable(response.status_code):
                    raise NemwebError(
                        f"GET {url} returned HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                reason = f"HTTP {response.status_code}"

            if attempt == self._max_retries:
                raise NemwebError(f"GET {url} failed after {attempt + 1} attempts: {reason}")
            delay = self._backoff_s * 2**attempt
            log.warning(
                "GET %s: %s; retry %d/%d in %.1fs",
                url,
                reason,
                attempt + 1,
                self._max_retries,
                delay,
            )
            self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_at is not None:
            wait = self._min_interval_s - (self._clock() - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()

    def _cache_path(self, url: str) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha256(url.encode()).hexdigest()
        return self._cache_dir / f"{digest[:16]}_{_basename(url)}"


def _is_retryable(status_code: int) -> bool:
    return status_code >= 500 or status_code == 429


def _basename(url: str) -> str:
    return urlsplit(url).path.rsplit("/", 1)[-1]


def _write_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    tmp.write_bytes(content)
    tmp.replace(path)
