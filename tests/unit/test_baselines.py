from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from nem_forecast.backtest.baselines import (
    BASELINES,
    Persistence,
    RuleBaseline,
    SameWeekdayMean,
    SeasonalNaive,
)
from nem_forecast.features.build import LABEL, build_features
from nem_forecast.features.series import PERIOD
from tests.synthetic import START, series

CUTOFF = START + 50 * 48 * PERIOD
ORIGINS = [CUTOFF + k * PERIOD for k in range(0, 96, 7)]


def forecast(
    baseline: RuleBaseline, data: pl.DataFrame, origins: list[datetime] = ORIGINS
) -> pl.DataFrame:
    baseline.fit(data.filter(pl.col("period_end") <= CUTOFF))
    frame = build_features(data, origins)
    return frame.select("origin", "horizon", LABEL, "demand_at_origin").join(
        baseline.predict(frame.drop(LABEL)), on=["origin", "horizon"]
    )


@pytest.mark.parametrize("baseline", [b for b in BASELINES if b is not Persistence])
def test_seasonal_rules_are_exact_on_a_perfectly_repeating_day(
    baseline: type[RuleBaseline],
) -> None:
    out = forecast(baseline(), series(days=60))

    assert (out["p50"] - out[LABEL]).abs().max() == pytest.approx(0, abs=1e-9)
    # No error in the 90 days before the cutoff, so no interval width either.
    assert out.select((pl.col("p90") - pl.col("p10")).max()).item() == pytest.approx(0, abs=1e-9)


def test_persistence_holds_the_last_value_flat() -> None:
    out = forecast(Persistence(), series(days=60))

    assert out["p50"].to_list() == out["demand_at_origin"].to_list()


def test_intervals_always_contain_the_median_and_are_roughly_calibrated() -> None:
    noisy = series(days=120, noise_sd=150.0)
    late_origins = [START + 100 * 48 * PERIOD + k * PERIOD for k in range(0, 48 * 10, 5)]

    baseline = SeasonalNaive()
    baseline.fit(noisy.filter(pl.col("period_end") <= late_origins[0]))
    frame = build_features(noisy, late_origins)
    out = frame.select("origin", "horizon", LABEL).join(
        baseline.predict(frame.drop(LABEL)), on=["origin", "horizon"]
    )

    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()
    coverage = out.select(pl.col(LABEL).is_between(pl.col("p10"), pl.col("p90")).mean()).item()
    assert 0.7 < coverage < 0.9


def test_predict_before_fit_is_an_error() -> None:
    frame = build_features(series(days=60), ORIGINS, with_label=False)

    with pytest.raises(RuntimeError, match="call fit"):
        SeasonalNaive().predict(frame)


def test_fit_needs_enough_history_for_its_rule() -> None:
    with pytest.raises(ValueError, match="not enough history"):
        SameWeekdayMean().fit(series(days=20))


def test_each_baseline_has_a_distinct_name_and_rule() -> None:
    assert len({b.name for b in BASELINES}) == len(BASELINES)
    assert len({b.point for b in BASELINES}) == len(BASELINES)
    assert all(b.description for b in BASELINES)
