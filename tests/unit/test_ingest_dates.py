from __future__ import annotations

from datetime import date, datetime

import pytest

from nem_forecast.ingest.backfill import default_months, month_period, month_starts
from nem_forecast.ingest.recent import interval_end_from_filename
from nem_forecast.schemas import NEM_TZ


def test_default_months_cover_whole_years_up_to_this_month() -> None:
    assert default_months(date(2026, 10, 13), years=3) == (date(2023, 10, 1), date(2026, 10, 1))


def test_month_starts_cross_year_boundary() -> None:
    assert month_starts(date(2025, 11, 15), date(2026, 2, 1)) == [
        date(2025, 11, 1),
        date(2025, 12, 1),
        date(2026, 1, 1),
        date(2026, 2, 1),
    ]


def test_month_starts_empty_when_reversed() -> None:
    assert month_starts(date(2026, 2, 1), date(2025, 11, 1)) == []


def test_month_period_is_market_time_and_end_inclusive() -> None:
    start, end = month_period(date(2025, 12, 1))

    assert start == datetime(2025, 12, 1, tzinfo=NEM_TZ)
    assert end == datetime(2026, 1, 1, tzinfo=NEM_TZ)


def test_interval_end_from_dispatch_is_filename() -> None:
    assert interval_end_from_filename(
        "https://nemweb.com.au/x/PUBLIC_DISPATCHIS_202610130440_0000000542597921.zip"
    ) == datetime(2026, 10, 13, 4, 40, tzinfo=NEM_TZ)


def test_interval_end_rejects_other_files() -> None:
    with pytest.raises(ValueError, match="not a DispatchIS file"):
        interval_end_from_filename("PUBLIC_TRADINGIS_202610130440_0000000542597921.zip")
