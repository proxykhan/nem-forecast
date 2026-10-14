"""What gets logged to MLflow, and how runs nest.

* Backtest: one parent run per model, one nested child run per walk-forward fold. Every run gets
  params (backtest config and model hyperparameters), metrics overall, by horizon and by time of
  day, the feature list, the data date range, the git SHA, and the backtest report as artifacts.
* Tuning: one parent run per search, one nested child run per Optuna trial.

Metric names: ``mae`` (all horizons), ``mae_h07`` (horizon 7), ``mae_tod_16-21_evening_peak``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import mlflow
import polars as pl

from nem_forecast.backtest.baselines import Forecaster
from nem_forecast.backtest.harness import BacktestConfig, Fold, forecaster_params
from nem_forecast.backtest.metrics import ALL_BUCKETS, ALL_HORIZONS, score
from nem_forecast.features.build import FEATURE_VERSION, FEATURES
from nem_forecast.models.tuning import TuningResult, ValidationSplit
from nem_forecast.schemas import Region
from nem_forecast.tracking.client import experiment_id

log = logging.getLogger(__name__)

REPORT_ARTIFACTS = "backtest_report"


def flatten_metrics(predictions: pl.DataFrame) -> dict[str, float]:
    """Overall, per-horizon and per-time-of-day metrics for one model's predictions."""
    flat: dict[str, float] = {}
    for row in score(predictions).iter_rows(named=True):
        horizon, bucket = row["horizon"], row["tod_bucket"]
        if horizon == ALL_HORIZONS and bucket == ALL_BUCKETS:
            name = row["metric"]
        elif bucket == ALL_BUCKETS:
            name = f"{row['metric']}_h{horizon:02d}"
        elif horizon == ALL_HORIZONS:
            name = f"{row['metric']}_tod_{bucket.replace(' ', '_')}"
        else:
            continue  # horizon x time-of-day cells live in model_run_metrics
        flat[name] = float(row["value"])
    return flat


model_params = forecaster_params


class MlflowBacktestObserver:
    """Implements the harness's BacktestObserver."""

    def __init__(
        self,
        *,
        region: Region,
        config: BacktestConfig,
        git_sha: str | None,
        data_span: tuple[datetime, datetime],
        eval_start: datetime,
        experiment: str | None = None,
    ) -> None:
        self._experiment_id = experiment_id(experiment or f"{region.value} backtest")
        self._tags = {
            "git_sha": git_sha or "unknown",
            "mlflow.source.git.commit": git_sha or "unknown",
            "region": region.value,
        }
        self._common = {
            **{
                f"backtest.{key}": value
                for key, value in asdict(config).items()
                if key != "horizons"
            },
            "horizons": f"1-{max(config.horizons)}",
            "region": region.value,
            "feature_version": FEATURE_VERSION,
            "features": ",".join(FEATURES),
            "data_start": data_span[0].isoformat(),
            "data_end": data_span[1].isoformat(),
            "eval_start": eval_start.isoformat(),
        }
        self.parent_runs: dict[str, str] = {}
        self.child_runs: list[str] = []

    def start_model(self, forecaster: Forecaster) -> None:
        run = mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=forecaster.name,
            tags=self._tags | {"model_kind": getattr(forecaster, "kind", "model")},
        )
        mlflow.log_params(self._common | model_params(forecaster))
        mlflow.log_dict(
            {"feature_version": FEATURE_VERSION, "features": list(FEATURES)}, "features.json"
        )
        self.parent_runs[forecaster.name] = run.info.run_id

    def end_fold(self, forecaster: Forecaster, fold: Fold, predictions: pl.DataFrame) -> None:
        with mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=f"{forecaster.name} fold {fold.number:02d}",
            nested=True,
            tags=self._tags | {"fold": str(fold.number)},
        ) as child:
            mlflow.log_params(
                self._common
                | model_params(forecaster)
                | {
                    "fold": fold.number,
                    "train_end": fold.cutoff.isoformat(),
                    "fold_start": fold.cutoff.isoformat(),
                    "fold_end": fold.end.isoformat(),
                }
            )
            mlflow.log_metrics(flatten_metrics(predictions))
            self.child_runs.append(child.info.run_id)

    def end_model(self, forecaster: Forecaster, predictions: pl.DataFrame) -> None:
        mlflow.log_metrics(flatten_metrics(predictions))
        mlflow.end_run()

    def abort_model(self, forecaster: Forecaster) -> None:
        mlflow.end_run(status="FAILED")

    def log_report(self, paths: Iterable[Path], *, include_folds: bool = True) -> None:
        """Attach the backtest report to every model run, and to every fold run unless told not
        to (the nightly run skips folds to stay inside free-tier artifact storage)."""
        client = mlflow.MlflowClient()
        files = list(paths)
        runs = [*self.parent_runs.values(), *(self.child_runs if include_folds else [])]
        for run_id in runs:
            for path in files:
                client.log_artifact(run_id, str(path), artifact_path=REPORT_ARTIFACTS)


class MlflowTuningLogger:
    """Parent run for an Optuna search; call it once per trial to log a nested child run."""

    def __init__(
        self,
        *,
        region: Region,
        split: ValidationSplit,
        git_sha: str | None,
        trials: int,
        experiment: str | None = None,
    ) -> None:
        self._experiment_id = experiment_id(experiment or f"{region.value} tuning")
        self._tags = {"git_sha": git_sha or "unknown", "region": region.value}
        self._common = {
            "model": "lightgbm_direct",
            "objective": "validation MAE of the median forecast",
            "train_end": split.train_end.isoformat(),
            "valid_end": split.valid_end.isoformat(),
            "max_trials": trials,
            "features": ",".join(FEATURES),
        }
        self.run_id: str | None = None

    def __enter__(self) -> Self:
        run = mlflow.start_run(
            experiment_id=self._experiment_id, run_name="lightgbm_direct optuna", tags=self._tags
        )
        mlflow.log_params(self._common)
        self.run_id = run.info.run_id
        return self

    def __call__(self, number: int, params: dict[str, Any], mae: float) -> None:
        with mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=f"trial {number:02d}",
            nested=True,
            tags=self._tags,
        ):
            mlflow.log_params(self._common | params | {"trial": number})
            mlflow.log_metric("validation_mae", mae)

    def finish(self, result: TuningResult) -> None:
        mlflow.log_params({f"best.{key}": value for key, value in result.best_params.items()})
        mlflow.log_metric("best_validation_mae", result.best_mae)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
