"""Features table, model_runs persistence and the Phase 2 commands, against real PostgreSQL."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from sqlalchemy import Engine, text
from typer.testing import CliRunner

from nem_forecast import cli
from nem_forecast.backtest.baselines import BASELINES
from nem_forecast.backtest.harness import BacktestConfig, run_backtest, save_results
from nem_forecast.backtest.metrics import ALL_BUCKETS, ALL_HORIZONS
from nem_forecast.db import apply_migrations
from nem_forecast.features.build import FEATURE_VERSION, FEATURES, one_step_features, write_features
from nem_forecast.features.series import load_dispatch, to_half_hourly
from nem_forecast.ingest.loader import upsert_intervals
from nem_forecast.schemas import DispatchInterval, Region
from tests.synthetic import dispatch, series

pytestmark = pytest.mark.integration

SERIES = series(days=45, noise_sd=80.0)


@pytest.fixture
def engine(pg_engine: Engine) -> Engine:
    apply_migrations(pg_engine)
    return pg_engine


@pytest.fixture
def with_dispatch(engine: Engine) -> Engine:
    """raw_dispatch holding SERIES as 5-minute rows, loaded the way ingestion loads them."""
    records = [
        DispatchInterval(
            settlement_date=ts, region=Region.VIC1, demand_mw=mw, rrp=50.0, source_file="synthetic"
        )
        for ts, mw in dispatch(SERIES).iter_rows()
    ]
    with engine.begin() as conn:
        upsert_intervals(conn, records)
    return engine


def plain(output: str) -> str:
    """CLI error text without colour codes, box drawing or line wrapping (CI forces colour)."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", output)
    return " ".join(text.replace("\u2502", " ").split())


def scalar(engine: Engine, sql: str, **params: Any) -> Any:
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar_one()


def test_dispatch_round_trips_to_the_same_half_hourly_series(with_dispatch: Engine) -> None:
    loaded = to_half_hourly(load_dispatch(with_dispatch, Region.VIC1))

    assert loaded["period_end"].equals(SERIES["period_end"])
    assert (loaded["demand"] - SERIES["demand"]).abs().max() == pytest.approx(0, abs=1e-6)


def test_features_table_holds_one_traceable_row_per_period(engine: Engine) -> None:
    frame = one_step_features(SERIES)

    assert write_features(engine, Region.VIC1, frame) == SERIES.height - 1
    assert write_features(engine, Region.VIC1, frame) == SERIES.height - 1  # replaces, no dupes

    assert scalar(engine, "SELECT count(*) FROM features") == SERIES.height - 1
    assert scalar(engine, "SELECT bool_and(data_cutoff < settlement_date) FROM features")
    assert scalar(engine, "SELECT DISTINCT feature_version FROM features") == FEATURE_VERSION
    stored = scalar(
        engine,
        "SELECT feature_values FROM features ORDER BY settlement_date DESC LIMIT 1",
    )
    values = stored if isinstance(stored, dict) else json.loads(stored)
    assert set(values) == set(FEATURES)
    assert values["horizon"] == 1


def test_backtest_results_are_recorded_per_model(engine: Engine) -> None:
    result = run_backtest(
        SERIES, [b() for b in BASELINES], BacktestConfig(test_days=7, refit_every_days=3)
    )

    run_ids = save_results(
        engine, result, region=Region.VIC1, git_sha="abc1234", started_at=datetime.now(UTC)
    )

    assert set(run_ids) == {b.name for b in BASELINES}
    assert scalar(engine, "SELECT count(*) FROM model_runs WHERE status = 'succeeded'") == len(
        BASELINES
    )
    assert scalar(engine, "SELECT count(*) FROM model_run_metrics") == result.metrics.height
    stored_mae = scalar(
        engine,
        "SELECT value FROM model_run_metrics WHERE run_id = :run_id AND metric = 'mae' "
        "AND horizon = 0 AND tod_bucket = 'all'",
        run_id=run_ids["seasonal_naive"],
    )
    expected = result.metrics.filter(
        pl.col("model") == "seasonal_naive",
        pl.col("metric") == "mae",
        pl.col("horizon") == ALL_HORIZONS,
        pl.col("tod_bucket") == ALL_BUCKETS,
    )["value"][0]
    assert stored_mae == pytest.approx(expected)
    params = scalar(engine, "SELECT params FROM model_runs WHERE model_name = 'persistence'")
    assert params["refit_every_days"] == 3


class TestCommands:
    @pytest.fixture
    def run_cli(self, with_dispatch: Engine, monkeypatch: pytest.MonkeyPatch) -> Any:
        monkeypatch.setenv("DATABASE_URL", with_dispatch.url.render_as_string(hide_password=False))
        runner = CliRunner()
        return lambda *args: runner.invoke(cli.app, list(args))

    def test_build_features(self, run_cli: Any, with_dispatch: Engine) -> None:
        result = run_cli("build-features")

        assert result.exit_code == 0, result.output
        assert f"Wrote {SERIES.height - 1:,} v1 feature rows for VIC1" in result.output

    def test_backtest_records_runs_and_writes_the_report(
        self, run_cli: Any, with_dispatch: Engine, tmp_path: Path
    ) -> None:
        result = run_cli(
            "backtest",
            "--no-models",
            "--test-days",
            "7",
            "--refit-every-days",
            "3",
            "--report-dir",
            str(tmp_path),
        )

        assert result.exit_code == 0, result.output
        for baseline in BASELINES:
            assert baseline.name in result.output
        assert scalar(with_dispatch, "SELECT count(*) FROM model_runs") == len(BASELINES)
        assert (tmp_path / "README.md").exists()
        assert (tmp_path / "mae_by_horizon.png").exists()

    def test_backtest_without_saving(
        self, run_cli: Any, with_dispatch: Engine, tmp_path: Path
    ) -> None:
        result = run_cli(
            "backtest",
            "--test-days",
            "7",
            "--refit-every-days",
            "3",
            "--report-dir",
            str(tmp_path),
            "--no-save",
            "--no-models",
        )

        assert result.exit_code == 0, result.output
        assert scalar(with_dispatch, "SELECT count(*) FROM model_runs") == 0


class TestModelCommands:
    """tune, backtest with models, and train, end to end on a small synthetic database."""

    @pytest.fixture
    def with_dispatch(self, engine: Engine) -> Engine:
        """Models need warm-up plus a calibration window before the first refit: 75 days."""
        records = [
            DispatchInterval(
                settlement_date=ts, region=Region.VIC1, demand_mw=mw, source_file="synthetic"
            )
            for ts, mw in dispatch(series(days=75, noise_sd=80.0)).iter_rows()
        ]
        with engine.begin() as conn:
            upsert_intervals(conn, records)
        return engine

    @pytest.fixture
    def run_cli(
        self, with_dispatch: Engine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> Any:
        monkeypatch.setenv("DATABASE_URL", with_dispatch.url.render_as_string(hide_password=False))
        config = tmp_path / "config"
        config.mkdir()
        (config / "lightgbm_vic1.json").write_text(
            json.dumps({"params": {"num_boost_round": 40, "num_leaves": 15}}), encoding="utf-8"
        )
        runner = CliRunner()
        return lambda *args: runner.invoke(cli.app, list(args))

    def test_tune_writes_parameters(self, run_cli: Any, tmp_path: Path) -> None:
        result = run_cli("tune", "--trials", "2", "--validation-days", "7", "--test-days", "7")

        assert result.exit_code == 0, result.output
        assert "after 2 trials" in result.output
        saved = json.loads((tmp_path / "config" / "lightgbm_vic1.json").read_text("utf-8"))
        assert saved["trials"] == 2

    def test_backtest_with_models_records_kinds_and_mlflow_runs(
        self, run_cli: Any, with_dispatch: Engine, tmp_path: Path
    ) -> None:
        result = run_cli(
            "backtest", "--test-days", "4", "--refit-every-days", "2", "--report-dir", str(tmp_path)
        )

        assert result.exit_code == 0, result.output
        for name in ("ridge", "lightgbm_direct", "lightgbm_quantile"):
            assert f"{name}: " in result.output
        kinds = dict(
            tuple(row)
            for row in with_dispatch.connect().execute(
                text("SELECT model_name, model_kind FROM model_runs")
            )
        )
        assert kinds["ridge"] == "model"
        assert kinds["seasonal_naive"] == "baseline"
        assert scalar(with_dispatch, "SELECT count(mlflow_run_id) FROM model_runs") == 7
        assert "## Verdict" in (tmp_path / "README.md").read_text("utf-8")

    def test_train_saves_a_loadable_model_with_its_card(
        self, run_cli: Any, with_dispatch: Engine, tmp_path: Path
    ) -> None:
        from nem_forecast.models.catalog import load_model

        run_cli(
            "backtest", "--test-days", "4", "--refit-every-days", "2", "--report-dir", str(tmp_path)
        )
        result = run_cli(
            "train", "--model", "lightgbm_quantile", "--output-dir", str(tmp_path / "out")
        )

        assert result.exit_code == 0, result.output
        saved = tmp_path / "out" / "vic1-lightgbm_quantile"
        assert load_model(saved).name == "lightgbm_quantile"
        card = (saved / "README.md").read_text("utf-8")
        assert "## Known failure modes" in card
        assert (saved / "model_card.md").read_text("utf-8") == card
        assert (
            scalar(with_dispatch, "SELECT count(*) FROM model_runs WHERE run_type = 'train'") == 1
        )

    def test_promotion_refuses_code_that_is_not_committed(self, run_cli: Any) -> None:
        result = run_cli("train", "--model", "ridge", "--promote")

        assert result.exit_code == 2
        assert "commit your changes first" in plain(result.output)

    def test_unknown_model_is_refused(self, run_cli: Any) -> None:
        result = run_cli("train", "--model", "prophet")

        assert result.exit_code == 2
        assert "choose one of" in plain(result.output)
