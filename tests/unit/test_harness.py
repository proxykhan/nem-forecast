from __future__ import annotations

import itertools
from datetime import datetime, timedelta

import polars as pl
import pytest

from nem_forecast.backtest.baselines import BASELINES, PREDICTIONS, SeasonalNaive
from nem_forecast.backtest.harness import (
    BacktestConfig,
    HarnessError,
    plan_folds,
    run_backtest,
    walk_forward,
)
from nem_forecast.backtest.metrics import ALL_BUCKETS
from nem_forecast.features.build import ASOF, KEYS, LABEL
from nem_forecast.features.series import span
from nem_forecast.schemas import NEM_TZ
from tests.synthetic import series

CONFIG = BacktestConfig(test_days=14, refit_every_days=5)
DATA = series(days=60)


def test_folds_start_at_market_midnight_and_tile_the_test_period() -> None:
    folds = plan_folds(DATA, CONFIG)
    last = DATA["period_end"].max()

    first_cutoff = folds[0].cutoff.astimezone(NEM_TZ)
    assert (first_cutoff.hour, first_cutoff.minute) == (0, 0)
    assert len(folds) == 3
    assert all(a.end == b.cutoff for a, b in itertools.pairwise(folds))
    assert folds[-1].end == last
    assert all(f.end - f.cutoff <= timedelta(days=5) for f in folds)


def test_a_test_period_longer_than_the_data_is_refused() -> None:
    with pytest.raises(HarnessError, match="no training data"):
        plan_folds(DATA, BacktestConfig(test_days=90))


def test_seasonal_naive_scores_zero_on_a_perfectly_repeating_day() -> None:
    result = run_backtest(DATA, [SeasonalNaive()], CONFIG)
    mae = result.metrics.filter(pl.col("metric") == "mae", pl.col("tod_bucket") == ALL_BUCKETS)

    assert set(mae["horizon"]) == {0, *range(1, 49)}
    assert mae["value"].max() == pytest.approx(0, abs=1e-9)


class Spy:
    """Records exactly what the harness lets a forecaster see."""

    name = "spy"

    def __init__(self) -> None:
        self.calls: list[tuple[datetime, pl.DataFrame]] = []
        self._latest_seen: datetime | None = None

    def fit(self, history: pl.DataFrame) -> None:
        self._latest_seen = span(history)[1]

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        assert self._latest_seen is not None
        self.calls.append((self._latest_seen, features))
        return features.select(*KEYS).with_columns(
            [pl.lit(1.0).alias(column) for column in PREDICTIONS]
        )


def test_training_data_never_extends_past_a_forecast_origin() -> None:
    spy = Spy()

    walk_forward(DATA, spy, CONFIG)

    assert len(spy.calls) == len(plan_folds(DATA, CONFIG))
    for trained_until, features in spy.calls:
        assert (features["origin"] >= trained_until).all()
        assert LABEL not in features.columns
        for column in [c for c in features.columns if c.endswith(ASOF)]:
            assert (features[column] <= features["origin"]).all()


class Broken(Spy):
    def __init__(self, mutate: str) -> None:
        super().__init__()
        self.mutate = mutate

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        out = super().predict(features)
        return out.drop("p90") if self.mutate == "drop" else out.head(-1)


@pytest.mark.parametrize(("mutate", "message"), [("drop", "did not return"), ("short", "rows")])
def test_malformed_predictions_are_rejected(mutate: str, message: str) -> None:
    with pytest.raises(HarnessError, match=message):
        walk_forward(DATA, Broken(mutate), CONFIG)


def test_run_backtest_scores_every_baseline_over_the_test_period() -> None:
    noisy = series(days=60, noise_sd=100.0)
    result = run_backtest(noisy, [b() for b in BASELINES], CONFIG)

    assert set(result.predictions["model"]) == {b.name for b in BASELINES}
    assert result.eval_start == result.folds[0].cutoff
    assert result.predictions["origin"].min() == result.eval_start
    assert result.predictions["target"].max() == result.eval_end == noisy["period_end"].max()
    assert result.train_start == noisy["period_end"].min()
    assert list(result.descriptions) == [b.name for b in BASELINES]
