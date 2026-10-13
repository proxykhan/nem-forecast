from __future__ import annotations

import math

import polars as pl
import pytest

from nem_forecast.backtest.metrics import (
    ALL_BUCKETS,
    ALL_HORIZONS,
    METRICS,
    pinball,
    score,
    tod_bucket,
)


def test_pinball_penalises_under_and_over_forecasts_asymmetrically() -> None:
    frame = pl.DataFrame({"actual": [100.0, 100.0], "predicted": [90.0, 110.0]})

    loss = frame.select(pinball(pl.col("actual"), pl.col("predicted"), 0.9).alias("l"))["l"]

    assert loss.to_list() == pytest.approx([9.0, 1.0])  # under-forecasting a p90 costs 9x more


@pytest.mark.parametrize(
    ("half_hour", "bucket"),
    [
        (0, "00-06 overnight"),
        (11, "00-06 overnight"),
        (12, "06-10 morning ramp"),
        (31, "10-16 solar trough"),
        (32, "16-21 evening peak"),
        (47, "21-24 late evening"),
    ],
)
def test_time_of_day_buckets(half_hour: int, bucket: str) -> None:
    frame = pl.DataFrame({"half_hour": [half_hour]})

    assert frame.select(tod_bucket(pl.col("half_hour")))[0, 0] == bucket


PREDICTIONS = pl.DataFrame(
    {
        "model": ["m", "m", "m", "m"],
        "horizon": [1, 1, 2, 2],
        "half_hour": [0, 40, 0, 40],
        "demand": [1000.0, 2000.0, 1000.0, 2000.0],
        "p10": [900.0, 1900.0, 800.0, 2100.0],
        "p50": [1000.0, 2100.0, 900.0, 2200.0],
        "p90": [1100.0, 2200.0, 950.0, 2300.0],
    }
)


def metric(tidy: pl.DataFrame, name: str, horizon: int, bucket: str = ALL_BUCKETS) -> float:
    return float(
        tidy.filter(
            pl.col("metric") == name, pl.col("horizon") == horizon, pl.col("tod_bucket") == bucket
        )["value"][0]
    )


def test_point_metrics_by_horizon_and_overall() -> None:
    tidy = score(PREDICTIONS)

    assert metric(tidy, "mae", 1) == pytest.approx(50.0)  # errors 0 and 100
    assert metric(tidy, "mae", 2) == pytest.approx(150.0)  # errors 100 and 200
    assert metric(tidy, "mae", ALL_HORIZONS) == pytest.approx(100.0)
    assert metric(tidy, "rmse", 2) == pytest.approx(math.sqrt((100**2 + 200**2) / 2))
    assert metric(tidy, "mape", 1) == pytest.approx((0 + 5) / 2)


def test_interval_coverage_and_pinball() -> None:
    tidy = score(PREDICTIONS)

    assert metric(tidy, "coverage_80", 1) == pytest.approx(100.0)
    assert metric(tidy, "coverage_80", 2) == pytest.approx(0.0)  # both actuals outside
    # Horizon 1, row 2: actual 2000 vs p10 1900 -> 0.1 * 100 = 10; row 1 hits p10 at 900 -> 10.
    assert metric(tidy, "pinball_q10", 1) == pytest.approx(10.0)
    expected_avg = sum(metric(tidy, f"pinball_q{q}", 1) for q in (10, 50, 90)) / 3
    assert metric(tidy, "pinball_avg", 1) == pytest.approx(expected_avg)


def test_metrics_by_time_of_day() -> None:
    tidy = score(PREDICTIONS)

    assert metric(tidy, "mae", ALL_HORIZONS, "00-06 overnight") == pytest.approx(50.0)
    assert metric(tidy, "mae", ALL_HORIZONS, "16-21 evening peak") == pytest.approx(150.0)
    assert metric(tidy, "mae", 2, "16-21 evening peak") == pytest.approx(200.0)


def test_output_is_tidy() -> None:
    tidy = score(PREDICTIONS)

    assert tidy.columns == ["model", "metric", "horizon", "tod_bucket", "value", "n_obs"]
    assert set(tidy["metric"]) == set(METRICS)
    assert not tidy.select(["model", "metric", "horizon", "tod_bucket"]).is_duplicated().any()
    # 2 horizons x 2 buckets, 2 horizons, 2 buckets, 1 overall = 9 cells per metric
    assert tidy.height == 9 * len(METRICS)
    overall = tidy.filter(pl.col("horizon") == ALL_HORIZONS, pl.col("tod_bucket") == ALL_BUCKETS)
    assert set(overall["n_obs"]) == {4}


def test_rows_without_an_actual_or_a_forecast_are_not_scored() -> None:
    with_gaps = pl.concat(
        [
            PREDICTIONS,
            PREDICTIONS.head(2).with_columns(demand=None),
            PREDICTIONS.head(1).with_columns(p50=None),
        ],
        how="vertical_relaxed",
    )

    assert metric(score(with_gaps), "mae", ALL_HORIZONS) == pytest.approx(100.0)
