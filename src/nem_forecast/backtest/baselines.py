"""Baseline forecasters: the bar every later model has to clear.

Every forecaster, baseline or model, implements :class:`Forecaster`:

* ``fit(history)`` learns from a half-hourly series that ends at the training cutoff. The
  harness never passes anything later.
* ``predict(features)`` takes the per-(origin, horizon) feature frame from
  :func:`~nem_forecast.features.build.build_features`, already leakage-checked and with the
  label removed, and returns the same rows with ``p10``, ``p50`` and ``p90``.

The doc's three baselines (seasonal naive, weekly seasonal naive, 4-week same-weekday mean)
each read one fixed lag of the *target*, so they ignore everything observed after that lag and
their error is flat across horizons. Persistence, the last observed value, is added as the
standard short-horizon bar; without it, any model that sees recent demand would look
spectacular at short horizons against baselines that never could.

The baselines' point forecasts are fixed rules with nothing to learn. ``fit`` estimates only
their prediction intervals: the empirical 10th and 90th percentiles of each rule's own errors,
per horizon, over the 90 days before the cutoff. 90 days is a fixed choice and is not tuned.
"""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar, Protocol

import polars as pl

from nem_forecast.features.build import (
    DEFAULT_HORIZONS,
    KEYS,
    LABEL,
    assert_no_leakage,
    build_features,
)
from nem_forecast.features.series import span

QUANTILES = (0.1, 0.5, 0.9)
PREDICTIONS = ("p10", "p50", "p90")
INTERVAL_WINDOW = timedelta(days=90)


class Forecaster(Protocol):
    @property
    def name(self) -> str: ...

    def fit(self, history: pl.DataFrame) -> None: ...

    def predict(self, features: pl.DataFrame) -> pl.DataFrame: ...


class RuleBaseline:
    """A point forecast read straight from one feature, with empirical error quantiles."""

    name: ClassVar[str]
    point: ClassVar[str]
    description: ClassVar[str]

    def __init__(self, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> None:
        self._horizons = horizons
        self._offsets: pl.DataFrame | None = None

    def fit(self, history: pl.DataFrame) -> None:
        _, cutoff = span(history)
        origins = history.filter(
            pl.col("period_end") >= cutoff - INTERVAL_WINDOW,
            pl.col("period_end") < cutoff,
        )["period_end"]
        frame = build_features(history, origins, self._horizons)
        assert_no_leakage(frame)
        errors = frame.drop_nulls([LABEL, self.point]).select(
            "horizon", (pl.col(LABEL) - pl.col(self.point)).alias("error")
        )
        if errors.is_empty():
            raise ValueError(
                f"{self.name}: not enough history before {cutoff} to estimate intervals"
            )
        self._offsets = errors.group_by("horizon").agg(
            pl.col("error").quantile(QUANTILES[0]).alias("_lower"),
            pl.col("error").quantile(QUANTILES[2]).alias("_upper"),
        )

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        if self._offsets is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")
        point = pl.col(self.point)
        # A biased rule can have both error quantiles on one side of zero; clip so the
        # interval always contains the point forecast (p10 <= p50 <= p90).
        return (
            features.join(self._offsets, on="horizon", how="left")
            .with_columns(
                p50=point,
                p10=pl.min_horizontal(point + pl.col("_lower"), point),
                p90=pl.max_horizontal(point + pl.col("_upper"), point),
            )
            .select(*KEYS, *PREDICTIONS)
        )


class SeasonalNaive(RuleBaseline):
    name = "seasonal_naive"
    point = "same_slot_prev_day"
    description = "Same half-hour yesterday (t-48)"


class WeeklySeasonalNaive(RuleBaseline):
    name = "weekly_seasonal_naive"
    point = "same_slot_prev_week"
    description = "Same half-hour last week (t-336)"


class SameWeekdayMean(RuleBaseline):
    name = "same_weekday_mean_4w"
    point = "same_slot_mean_4w"
    description = "Mean of the same half-hour on the last 4 same weekdays"


class Persistence(RuleBaseline):
    name = "persistence"
    point = "demand_at_origin"
    description = "Last observed half-hour, held flat (demand at the origin)"


BASELINES: tuple[type[RuleBaseline], ...] = (
    SeasonalNaive,
    WeeklySeasonalNaive,
    SameWeekdayMean,
    Persistence,
)
