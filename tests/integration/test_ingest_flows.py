"""Recent-ingest and backfill flows end to end: fake AEMO over HTTP, real PostgreSQL."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, text
from typer.testing import CliRunner

from nem_forecast import cli
from nem_forecast.db import apply_migrations
from nem_forecast.ingest.backfill import MonthStatus, backfill
from nem_forecast.ingest.loader import LoadResult
from nem_forecast.ingest.nemweb import price_and_demand_url
from nem_forecast.ingest.recent import ingest_dispatch_is
from nem_forecast.schemas import NEM_TZ, Region
from tests.aemo_files import (
    DISPATCH_IS_ZIP,
    FakeAemo,
    dispatch_is_text,
    monthly_price_and_demand,
    shifted_dispatch_is,
    zipped,
)

pytestmark = pytest.mark.integration

INTERVAL_END = datetime(2026, 10, 13, 4, 40, tzinfo=NEM_TZ)
SEP, OCT, NOV = date(2025, 9, 1), date(2025, 10, 1), date(2025, 11, 1)
INTERVALS = {SEP: 30 * 288, OCT: 31 * 288, NOV: 30 * 288}


@pytest.fixture
def engine(pg_engine: Engine) -> Engine:
    apply_migrations(pg_engine)
    return pg_engine


@pytest.fixture
def aemo() -> FakeAemo:
    return FakeAemo()


def scalar(engine: Engine, sql: str) -> Any:
    with engine.connect() as conn:
        return conn.execute(text(sql)).scalar_one()


def fixed_now(at: datetime) -> Callable[[], datetime]:
    return lambda: at


class TestRecentIngest:
    def test_loads_selected_intervals_and_survives_a_missing_file(
        self, engine: Engine, aemo: FakeAemo
    ) -> None:
        for minutes in (-15, -10):  # 04:25 is outside the window, 04:30 inside
            name, body = shifted_dispatch_is(minutes)
            aemo.add_dispatch_is(name, body)
        gone, _ = shifted_dispatch_is(-5)
        aemo.add_dispatch_is(gone, 404)  # rolled out of Current between listing and download
        aemo.add_dispatch_is(DISPATCH_IS_ZIP, zipped(dispatch_is_text()))

        with aemo.client() as client:
            report = ingest_dispatch_is(
                engine,
                client,
                start=INTERVAL_END - timedelta(minutes=15),
                end=INTERVAL_END,
                regions=[Region.VIC1],
            )

        assert (report.files, report.files_failed, report.records) == (3, 1, 2)
        assert report.missing_intervals == 1  # 04:35
        assert report.load == LoadResult(inserted=2, updated=0, unchanged=0)
        assert scalar(engine, "SELECT count(*) FROM raw_dispatch WHERE region = 'VIC1'") == 2

    def test_overlapping_runs_are_idempotent(self, engine: Engine, aemo: FakeAemo) -> None:
        aemo.add_dispatch_is(DISPATCH_IS_ZIP, zipped(dispatch_is_text()))
        window = {"start": INTERVAL_END - timedelta(hours=24), "end": INTERVAL_END}

        with aemo.client() as client:
            first = ingest_dispatch_is(engine, client, regions=list(Region), **window)
            second = ingest_dispatch_is(engine, client, regions=list(Region), **window)

        assert first.load.inserted == 5
        assert second.load == LoadResult(inserted=0, updated=0, unchanged=5)

    def test_rejects_an_empty_window(self, engine: Engine, aemo: FakeAemo) -> None:
        with aemo.client() as client, pytest.raises(ValueError, match="end must be after start"):
            ingest_dispatch_is(
                engine, client, start=INTERVAL_END, end=INTERVAL_END, regions=[Region.VIC1]
            )


class TestBackfill:
    def serve_months(self, aemo: FakeAemo, *months: date) -> None:
        for month in months:
            url = price_and_demand_url(month, Region.VIC1)
            aemo.files[url] = monthly_price_and_demand(month).encode()

    def run(
        self,
        engine: Engine,
        aemo: FakeAemo,
        *,
        now: datetime,
        cache_dir: Path | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        options: dict[str, Any] = {"first_month": SEP, "last_month": NOV} | kwargs
        with aemo.client(cache_dir) as client:
            return backfill(engine, client, region=Region.VIC1, now=fixed_now(now), **options)

    def test_loads_every_month_then_resumes_after_a_failure(
        self, engine: Engine, aemo: FakeAemo
    ) -> None:
        now = datetime(2026, 10, 13, tzinfo=NEM_TZ)
        self.serve_months(aemo, SEP, NOV)
        aemo.files[price_and_demand_url(OCT, Region.VIC1)] = 503  # AEMO hiccup mid-run

        first = self.run(engine, aemo, now=now)

        assert [o.status for o in first] == [
            MonthStatus.LOADED,
            MonthStatus.FAILED,
            MonthStatus.LOADED,
        ]
        assert first[0].records == INTERVALS[SEP]
        assert first[0].missing_intervals == 0

        self.serve_months(aemo, OCT)
        aemo.requests.clear()
        second = self.run(engine, aemo, now=now)

        assert [o.status for o in second] == [
            MonthStatus.SKIPPED,
            MonthStatus.LOADED,
            MonthStatus.SKIPPED,
        ]
        assert aemo.requests == [price_and_demand_url(OCT, Region.VIC1)]
        assert scalar(engine, "SELECT count(*) FROM raw_dispatch") == sum(INTERVALS.values())

    def test_force_reloads_complete_months_without_changing_them(
        self, engine: Engine, aemo: FakeAemo
    ) -> None:
        now = datetime(2026, 10, 13, tzinfo=NEM_TZ)
        self.serve_months(aemo, SEP)
        self.run(engine, aemo, now=now, last_month=SEP)

        again = self.run(engine, aemo, now=now, last_month=SEP, force=True)

        assert again[0].status is MonthStatus.LOADED
        assert again[0].load == LoadResult(inserted=0, updated=0, unchanged=INTERVALS[SEP])

    def test_current_month_is_reloaded_on_every_run(
        self, engine: Engine, aemo: FakeAemo, tmp_path: Path
    ) -> None:
        now = datetime(2025, 11, 13, 12, tzinfo=NEM_TZ)
        published_until = datetime(2025, 11, 13, tzinfo=NEM_TZ)
        url = price_and_demand_url(NOV, Region.VIC1)
        aemo.files[url] = monthly_price_and_demand(NOV, until=published_until).encode()

        for _ in range(2):
            (outcome,) = self.run(
                engine, aemo, now=now, cache_dir=tmp_path, first_month=NOV, last_month=NOV
            )
            assert outcome.status is MonthStatus.LOADED
            assert outcome.missing_intervals == 0

        assert aemo.requests == [url, url]  # never served from cache while it can still grow
        assert scalar(engine, "SELECT count(*) FROM raw_dispatch") == 12 * 288

    def test_finished_months_are_served_from_cache(
        self, engine: Engine, aemo: FakeAemo, tmp_path: Path
    ) -> None:
        now = datetime(2026, 10, 13, tzinfo=NEM_TZ)
        self.serve_months(aemo, SEP)

        for _ in range(2):
            self.run(engine, aemo, now=now, cache_dir=tmp_path, last_month=SEP, force=True)

        assert aemo.requests == [price_and_demand_url(SEP, Region.VIC1)]

    def test_month_with_a_gap_is_retried(self, engine: Engine, aemo: FakeAemo) -> None:
        now = datetime(2026, 10, 13, tzinfo=NEM_TZ)
        complete = monthly_price_and_demand(SEP)
        holed = "\n".join(line for line in complete.splitlines() if "2025/09/10 13:" not in line)
        aemo.files[price_and_demand_url(SEP, Region.VIC1)] = holed.encode()

        (first,) = self.run(engine, aemo, now=now, last_month=SEP)
        self.serve_months(aemo, SEP)
        (second,) = self.run(engine, aemo, now=now, last_month=SEP)

        assert first.missing_intervals == 12
        assert second.status is MonthStatus.LOADED
        assert second.load == LoadResult(inserted=12, updated=0, unchanged=INTERVALS[SEP] - 12)

    def test_unpublished_months_are_reported_not_failed(
        self, engine: Engine, aemo: FakeAemo
    ) -> None:
        now = datetime(2025, 11, 1, 0, 30, tzinfo=NEM_TZ)  # November's file does not exist yet
        self.serve_months(aemo, SEP, OCT)

        outcomes = self.run(engine, aemo, now=now, last_month=date(2025, 12, 1))

        assert [o.status for o in outcomes] == [
            MonthStatus.LOADED,
            MonthStatus.LOADED,
            MonthStatus.UNAVAILABLE,
            MonthStatus.UNAVAILABLE,
        ]

    def test_missing_past_month_is_a_failure(self, engine: Engine, aemo: FakeAemo) -> None:
        (outcome,) = self.run(
            engine, aemo, now=datetime(2026, 10, 13, tzinfo=NEM_TZ), last_month=SEP
        )

        assert outcome.status is MonthStatus.FAILED
        assert "404" in (outcome.error or "")

    def test_unparseable_month_is_a_failure(self, engine: Engine, aemo: FakeAemo) -> None:
        aemo.files[price_and_demand_url(SEP, Region.VIC1)] = b"<html>maintenance</html>"

        (outcome,) = self.run(
            engine, aemo, now=datetime(2026, 10, 13, tzinfo=NEM_TZ), last_month=SEP
        )

        assert outcome.status is MonthStatus.FAILED
        assert "unrecognised format" in (outcome.error or "")


class TestCommands:
    @pytest.fixture
    def run_cli(
        self, engine: Engine, aemo: FakeAemo, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> Callable[..., Any]:
        monkeypatch.setenv("DATABASE_URL", engine.url.render_as_string(hide_password=False))
        monkeypatch.setattr(cli, "NemwebClient", lambda cache_dir: aemo.client(tmp_path))
        runner = CliRunner()
        return lambda *args: runner.invoke(cli.app, list(args))

    def test_backfill_reports_months_and_coverage(
        self, run_cli: Callable[..., Any], aemo: FakeAemo
    ) -> None:
        for month in (SEP, OCT):
            aemo.files[price_and_demand_url(month, Region.VIC1)] = monthly_price_and_demand(
                month
            ).encode()

        result = run_cli("backfill", "--from", "2025-09", "--to", "2025-10")

        assert result.exit_code == 0, result.output
        assert "2025-09  loaded" in result.output
        assert (
            "raw_dispatch VIC1: 17,568 rows, 2025-09-01 00:05 to 2025-11-01 00:00" in result.output
        )

    def test_backfill_exits_non_zero_when_a_month_fails(self, run_cli: Callable[..., Any]) -> None:
        result = run_cli("backfill", "--from", "2025-09", "--to", "2025-09")

        assert result.exit_code == 1
        assert "2025-09  failed" in result.output
        assert "raw_dispatch VIC1: no rows" in result.output

    def test_ingest_reads_naive_times_as_market_time(
        self, run_cli: Callable[..., Any], aemo: FakeAemo
    ) -> None:
        aemo.add_dispatch_is(DISPATCH_IS_ZIP, zipped(dispatch_is_text()))

        result = run_cli(
            "ingest", "--start", "2026-10-13 04:35", "--end", "2026-10-13 04:40", "--region", "VIC1"
        )

        assert result.exit_code == 0, result.output
        assert "1 files (0 failed), 1 records: 1 inserted" in result.output
        assert "2026-10-13 04:40 to 2026-10-13 04:40" in result.output

    def test_ingest_defaults_to_the_last_day(
        self, run_cli: Callable[..., Any], aemo: FakeAemo
    ) -> None:
        name, body = shifted_dispatch_is(0)
        aemo.add_dispatch_is(name, body)

        result = run_cli("ingest", "--end", "2026-10-14 04:40")

        assert (
            result.exit_code == 1
        )  # the fixture interval is 24h before --end: outside (start, end]
        assert "0 files" in result.output

        result = run_cli("ingest", "--end", "2026-10-14 04:40", "--hours", "25")

        assert result.exit_code == 0, result.output
        assert "1 files (0 failed)" in result.output

    def test_ingest_exits_non_zero_when_nothing_was_found(
        self, run_cli: Callable[..., Any]
    ) -> None:
        result = run_cli("ingest", "--start", "2026-10-13 04:35", "--end", "2026-10-13 04:40")

        assert result.exit_code == 1
        assert "0 files" in result.output
