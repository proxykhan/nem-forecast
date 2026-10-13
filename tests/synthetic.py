"""Synthetic series with known structure, so feature and backtest expectations are exact."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import polars as pl

from nem_forecast.features.series import PERIOD, SERIES_SCHEMA

# 2025-06-02 00:00 market time (UTC+10): period i ends i+1 half-hours later.
START = datetime(2025, 6, 1, 14, tzinfo=UTC)


def daily_profile(i: int) -> float:
    """A demand-like daily cycle: identical every day, so yesterday predicts today exactly."""
    return 4500 + 1200 * math.sin(2 * math.pi * (i % 48) / 48)


def series(
    days: int,
    value: Callable[[int], float] = daily_profile,
    *,
    noise_sd: float = 0.0,
    seed: int = 7,
    start: datetime = START,
) -> pl.DataFrame:
    rng = random.Random(seed)  # noqa: S311 - reproducible test noise, not security
    n = days * 48
    return pl.DataFrame(
        {
            "period_end": [start + (i + 1) * PERIOD for i in range(n)],
            "demand": [value(i) + (rng.gauss(0, noise_sd) if noise_sd else 0.0) for i in range(n)],
        },
        schema=SERIES_SCHEMA,
    )


def index_series(days: int) -> pl.DataFrame:
    """Demand equal to the period's position, so every lag and window has a closed form."""
    return series(days, value=float)


def dispatch(half_hourly: pl.DataFrame) -> pl.DataFrame:
    """5-minute rows that average back to ``half_hourly`` (each value repeated six times)."""
    rows = [
        (end - timedelta(minutes=25 - 5 * k), demand)
        for end, demand in half_hourly.iter_rows()
        for k in range(6)
    ]
    return pl.DataFrame(
        rows,
        schema={"settlement_date": pl.Datetime("us", "UTC"), "demand_mw": pl.Float64},
        orient="row",
    )
