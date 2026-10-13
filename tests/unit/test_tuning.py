from __future__ import annotations

from datetime import UTC, timedelta
from pathlib import Path
from typing import Any

import pytest

from nem_forecast.backtest.harness import BacktestConfig, plan_folds
from nem_forecast.features.build import LABEL
from nem_forecast.models.tuning import (
    MAX_TRIALS,
    load_params,
    params_path,
    save_params,
    tune_lightgbm,
    validation_frames,
    validation_split,
)
from nem_forecast.schemas import Region
from tests.synthetic import series

DATA = series(days=75, noise_sd=80.0)
CONFIG = BacktestConfig(test_days=10, refit_every_days=5)


def test_validation_ends_exactly_where_the_backtest_begins() -> None:
    split = validation_split(DATA, CONFIG, days=14)

    assert split.valid_end == plan_folds(DATA, CONFIG)[0].cutoff
    assert split.valid_end - split.train_end == timedelta(days=14)


def test_validation_frames_never_touch_the_backtest_period() -> None:
    split = validation_split(DATA, CONFIG, days=14)

    train, valid = validation_frames(DATA, split)

    assert (train["target"] <= split.train_end.astimezone(UTC)).all()
    assert valid["origin"].min() == split.train_end
    assert (valid["target"] <= split.valid_end.astimezone(UTC)).all()
    assert valid[LABEL].null_count() == 0


def test_search_returns_the_best_trial_and_reports_every_trial() -> None:
    seen: list[tuple[int, dict[str, Any], float]] = []

    result = tune_lightgbm(
        DATA,
        validation_split(DATA, CONFIG, days=14),
        trials=3,
        on_trial=lambda n, p, mae: seen.append((n, p, mae)),
    )

    assert [n for n, _, _ in seen] == [0, 1, 2]
    assert result.best_mae == min(mae for _, _, mae in seen)
    assert result.best_params in [p for _, p, _ in seen]
    assert {"learning_rate", "num_leaves", "num_boost_round"} <= set(result.best_params)


@pytest.mark.parametrize("trials", [0, MAX_TRIALS + 1])
def test_trial_budget_is_capped(trials: int) -> None:
    with pytest.raises(ValueError, match="trials must be between"):
        tune_lightgbm(DATA, validation_split(DATA, CONFIG, days=14), trials=trials)


def test_params_round_trip(tmp_path: Path) -> None:
    result = tune_lightgbm(DATA, validation_split(DATA, CONFIG, days=14), trials=1)
    path = params_path(Region.VIC1, tmp_path)

    save_params(path, result, git_sha="abc1234")

    assert path.name == "lightgbm_vic1.json"
    assert load_params(path) == result.best_params
    assert load_params(tmp_path / "missing.json") is None
