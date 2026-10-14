"""Optuna search for LightGBM hyperparameters, on a validation split strictly before the backtest.

Tuning on the backtest period would let the test set choose the model. The validation split is
the 60 days immediately before the backtest's first origin: training uses data up to the split,
validation forecasts are scored only on targets that end at or before the backtest starts, and
the series is cut there so not even a validation label comes from the test period.

The search tunes the squared-error point model on MAE of its median forecast; the quantile model
reuses the same parameters with a quantile objective.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import optuna
import polars as pl

from nem_forecast.backtest.harness import BacktestConfig, plan_folds
from nem_forecast.features.build import LABEL, assert_no_leakage, build_features
from nem_forecast.features.series import PERIOD, TIME_ZONE
from nem_forecast.models.base import TRAIN_ORIGIN_EVERY, training_frame
from nem_forecast.models.lgbm import LightGBMDirect
from nem_forecast.models.params import MAX_TRIALS, VALIDATION_DAYS, load_params, params_path

log = logging.getLogger(__name__)

__all__ = ["MAX_TRIALS", "VALIDATION_DAYS", "load_params", "params_path"]  # re-exported

SEED = 42

TrialCallback = Callable[[int, dict[str, Any], float], None]


@dataclass(frozen=True)
class ValidationSplit:
    train_end: datetime  # training data and validation origins start here
    valid_end: datetime  # == the backtest's first origin; no validation target goes past it


@dataclass(frozen=True)
class TuningResult:
    split: ValidationSplit
    best_params: dict[str, Any]
    best_mae: float
    trials: list[tuple[int, dict[str, Any], float]]


def validation_split(
    series: pl.DataFrame, config: BacktestConfig, days: int = VALIDATION_DAYS
) -> ValidationSplit:
    test_start = plan_folds(series, config)[0].cutoff
    return ValidationSplit(train_end=test_start - timedelta(days=days), valid_end=test_start)


def search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "num_boost_round": trial.suggest_int("num_boost_round", 100, 1000, step=50),
    }


def validation_frames(
    series: pl.DataFrame, split: ValidationSplit
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(training rows, validation rows), neither touching anything after ``split.valid_end``."""
    train = training_frame(series.filter(pl.col("period_end") <= split.train_end))
    known = series.filter(pl.col("period_end") <= split.valid_end)
    origins = pl.datetime_range(
        split.train_end,
        split.valid_end - PERIOD,
        interval=TRAIN_ORIGIN_EVERY,
        time_zone=TIME_ZONE,
        eager=True,
    )
    valid = build_features(known, origins)
    assert_no_leakage(valid)
    return train, valid.filter(pl.col(LABEL).is_not_null())


def tune_lightgbm(
    series: pl.DataFrame,
    split: ValidationSplit,
    *,
    trials: int = MAX_TRIALS,
    on_trial: TrialCallback | None = None,
) -> TuningResult:
    if not 1 <= trials <= MAX_TRIALS:
        raise ValueError(f"trials must be between 1 and {MAX_TRIALS}")
    train, valid = validation_frames(series, split)
    log.info("tuning on %d training rows, %d validation rows", train.height, valid.height)
    history: list[tuple[int, dict[str, Any], float]] = []

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        model = LightGBMDirect(params)
        model.fit_frame(train)
        scored = valid.join(model.raw_predictions(valid), on=["origin", "target", "horizon"])
        mae = float(scored.select((pl.col(LABEL) - pl.col("_point")).abs().mean()).item())
        history.append((trial.number, params, mae))
        if on_trial is not None:
            on_trial(trial.number, params, mae)
        log.info("trial %d: MAE %.1f", trial.number, mae)
        return mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    best = next(params for number, params, _ in history if number == study.best_trial.number)
    return TuningResult(split, best, float(study.best_value), history)


def save_params(path: Path, result: TuningResult, *, git_sha: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "params": result.best_params,
        "validation_mae": result.best_mae,
        "trials": len(result.trials),
        "split": {key: value.isoformat() for key, value in asdict(result.split).items()},
        "git_sha": git_sha,
        "tuned_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
