"""MLflow logging against a throwaway local store (each test runs in its own directory)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
from pydantic import SecretStr

from nem_forecast.backtest.baselines import SeasonalNaive
from nem_forecast.backtest.harness import BacktestConfig, plan_folds, run_backtest
from nem_forecast.config import Settings
from nem_forecast.features.series import span
from nem_forecast.models.ridge import RidgeForecaster
from nem_forecast.models.tuning import tune_lightgbm, validation_split
from nem_forecast.schemas import Region
from nem_forecast.tracking.client import configure_mlflow
from nem_forecast.tracking.runs import (
    REPORT_ARTIFACTS,
    MlflowBacktestObserver,
    MlflowTuningLogger,
    flatten_metrics,
)
from tests.synthetic import series

DATA = series(days=75, noise_sd=80.0)
CONFIG = BacktestConfig(test_days=6, refit_every_days=3)


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path: Path) -> Iterator[None]:
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    yield
    while mlflow.active_run():
        mlflow.end_run()


def settings(**values: str) -> Settings:
    return Settings(_env_file=None, database_url=SecretStr("sqlite://"), **values)  # type: ignore[arg-type]


def test_without_a_tracking_uri_runs_stay_local(tmp_path: Path) -> None:
    assert configure_mlflow(settings()) == f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_dagshub_credentials_reach_mlflow_through_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    uri = "https://dagshub.com/someone/nem-forecast.mlflow"

    configured = configure_mlflow(
        settings(
            mlflow_tracking_uri=uri,
            mlflow_tracking_username="someone",
            mlflow_tracking_password="token-123",
        )
    )

    assert configured == uri
    assert mlflow.get_tracking_uri() == uri
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "someone"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "token-123"


def runs(experiment: str) -> list[mlflow.entities.Run]:
    found = mlflow.search_runs(experiment_names=[experiment], output_format="list")
    assert isinstance(found, list)
    return found


def test_backtest_logs_a_parent_per_model_and_a_child_per_fold(tmp_path: Path) -> None:
    observer = MlflowBacktestObserver(
        region=Region.VIC1,
        config=CONFIG,
        git_sha="abc1234",
        data_span=span(DATA),
        eval_start=plan_folds(DATA, CONFIG)[0].cutoff,
    )
    run_backtest(DATA, [SeasonalNaive(), RidgeForecaster()], CONFIG, observer=observer)
    report = tmp_path / "README.md"
    report.write_text("# report", encoding="utf-8")
    observer.log_report([report])

    logged = runs("VIC1 backtest")
    parents = [r for r in logged if "mlflow.parentRunId" not in r.data.tags]
    children = [r for r in logged if "mlflow.parentRunId" in r.data.tags]
    folds = len(plan_folds(DATA, CONFIG))

    assert sorted(r.info.run_name for r in parents) == ["ridge", "seasonal_naive"]
    assert len(children) == 2 * folds
    assert {c.data.tags["mlflow.parentRunId"] for c in children} == {p.info.run_id for p in parents}
    for run in logged:
        assert run.data.tags["git_sha"] == "abc1234"
        assert {"mae", "mae_h01", "mae_h48", "pinball_avg_h24"} <= set(run.data.metrics)
        assert run.data.params["features"].startswith("horizon,half_hour")
        assert "data_start" in run.data.params
        assert "eval_start" in run.data.params
        artifacts = mlflow.MlflowClient().list_artifacts(run.info.run_id, REPORT_ARTIFACTS)
        assert [a.path for a in artifacts] == [f"{REPORT_ARTIFACTS}/README.md"]
    assert all(r.info.status == "FINISHED" for r in logged)


def test_a_failing_model_marks_its_run_failed() -> None:
    class Exploding(SeasonalNaive):
        def predict(self, features):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    observer = MlflowBacktestObserver(
        region=Region.VIC1,
        config=CONFIG,
        git_sha=None,
        data_span=span(DATA),
        eval_start=plan_folds(DATA, CONFIG)[0].cutoff,
    )
    with pytest.raises(RuntimeError, match="boom"):
        run_backtest(DATA, [Exploding()], CONFIG, observer=observer)

    assert [r.info.status for r in runs("VIC1 backtest")] == ["FAILED"]


def test_tuning_logs_every_trial_under_one_search() -> None:
    split = validation_split(DATA, BacktestConfig(test_days=6), days=7)

    with MlflowTuningLogger(region=Region.VIC1, split=split, git_sha="abc", trials=2) as logger:
        result = tune_lightgbm(DATA, split, trials=2, on_trial=logger)
        logger.finish(result)

    logged = runs("VIC1 tuning")
    parent = next(r for r in logged if "mlflow.parentRunId" not in r.data.tags)
    trials = [r for r in logged if "mlflow.parentRunId" in r.data.tags]
    assert sorted(r.info.run_name for r in trials) == ["trial 00", "trial 01"]
    assert all("validation_mae" in r.data.metrics for r in trials)
    assert parent.data.metrics["best_validation_mae"] == pytest.approx(result.best_mae)
    assert parent.data.params["valid_end"] == split.valid_end.isoformat()


def test_flattened_metric_names() -> None:
    from nem_forecast.backtest.harness import walk_forward

    flat = flatten_metrics(walk_forward(DATA, SeasonalNaive(), CONFIG))

    assert {"mae", "mae_h01", "mae_h48", "coverage_80", "mae_tod_16-21_evening_peak"} <= set(flat)
    assert not any("h01_tod" in name for name in flat)


def test_report_can_skip_fold_runs(tmp_path: Path) -> None:
    observer = MlflowBacktestObserver(
        region=Region.VIC1,
        config=CONFIG,
        git_sha="abc1234",
        data_span=span(DATA),
        eval_start=plan_folds(DATA, CONFIG)[0].cutoff,
    )
    run_backtest(DATA, [SeasonalNaive()], CONFIG, observer=observer)
    report = tmp_path / "README.md"
    report.write_text("# report", encoding="utf-8")

    observer.log_report([report], include_folds=False)

    client = mlflow.MlflowClient()
    assert [
        a.path
        for a in client.list_artifacts(observer.parent_runs["seasonal_naive"], REPORT_ARTIFACTS)
    ] == [f"{REPORT_ARTIFACTS}/README.md"]
    assert all(
        client.list_artifacts(run_id, REPORT_ARTIFACTS) == [] for run_id in observer.child_runs
    )
