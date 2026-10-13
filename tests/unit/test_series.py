from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nem_forecast.features.series import span, to_half_hourly

T0 = datetime(2026, 1, 1, 14, tzinfo=UTC)  # 2026-01-02 00:00 market time


def dispatch(minutes: list[int], values: list[float] | None = None) -> pl.DataFrame:
    """5-minute rows ending ``minutes`` after T0."""
    return pl.DataFrame(
        {
            "settlement_date": [T0 + timedelta(minutes=m) for m in minutes],
            "demand_mw": values or [float(m) for m in minutes],
        },
        schema={"settlement_date": pl.Datetime("us", "UTC"), "demand_mw": pl.Float64},
    )


def test_half_hour_is_the_mean_of_its_six_intervals_labelled_by_period_end() -> None:
    minutes = list(range(5, 65, 5))
    result = to_half_hourly(dispatch(minutes, [float(v) for v in range(1, 13)]))

    assert result.rows() == [
        (T0 + timedelta(minutes=30), 3.5),
        (T0 + timedelta(minutes=60), 9.5),
    ]


def test_period_missing_an_interval_is_null_not_averaged(caplog: pytest.LogCaptureFixture) -> None:
    minutes = [m for m in range(5, 95, 5) if m != 40]

    with caplog.at_level(logging.WARNING):
        result = to_half_hourly(dispatch(minutes))

    assert result["demand"].to_list() == [17.5, None, 77.5]
    assert "1 of 3 half-hour periods incomplete" in caplog.text


def test_missing_period_still_appears_on_the_grid() -> None:
    minutes = [*range(5, 35, 5), *range(65, 95, 5)]

    result = to_half_hourly(dispatch(minutes))

    assert result["period_end"].to_list() == [T0 + timedelta(minutes=m) for m in (30, 60, 90)]
    assert result["demand"].to_list()[1] is None


def test_partial_final_period_is_in_progress_and_dropped() -> None:
    result = to_half_hourly(dispatch(list(range(5, 75, 5))))

    assert result["period_end"].to_list() == [
        T0 + timedelta(minutes=30),
        T0 + timedelta(minutes=60),
    ]


def test_empty_input_gives_an_empty_series() -> None:
    result = to_half_hourly(dispatch([]))

    assert result.is_empty()
    assert result.columns == ["period_end", "demand"]


def test_span() -> None:
    result = to_half_hourly(dispatch(list(range(5, 65, 5))))

    assert span(result) == (T0 + timedelta(minutes=30), T0 + timedelta(minutes=60))
    with pytest.raises(ValueError, match="empty"):
        span(result.clear())
