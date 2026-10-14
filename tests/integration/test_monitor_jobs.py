"""The monitoring and retraining jobs end to end against PostgreSQL, with fake Hub and GitHub."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import Engine, text

from nem_forecast.backtest.baselines import BASELINES
from nem_forecast.backtest.harness import BacktestConfig, run_backtest, save_results
from nem_forecast.db import apply_migrations
from nem_forecast.features.series import PERIOD
from nem_forecast.ingest.loader import upsert_intervals
from nem_forecast.models.base import DirectModel
from nem_forecast.models.catalog import load_model
from nem_forecast.models.registry import PromotionError, release_tag
from nem_forecast.models.ridge import RidgeForecaster
from nem_forecast.monitor.drift import INPUT_COLUMNS, Thresholds
from nem_forecast.monitor.jobs import run_monitor, run_retrain
from nem_forecast.schemas import DispatchInterval, Region
from nem_forecast.serve.model_store import LoadedModel
from tests.synthetic import START, dispatch, series

pytestmark = pytest.mark.integration

DAYS = 75
DATA = series(days=DAYS, noise_sd=60.0)
DATA_END = START + DAYS * 48 * PERIOD
NOW = DATA_END + timedelta(minutes=10)
SHA = "0123456789abcdef0123456789abcdef01234567"
LOOSE = Thresholds(dict.fromkeys(INPUT_COLUMNS, 1.0), "test: 1.0 everywhere")


class FakeIssues:
    def __init__(self) -> None:
        self.reports: list[dict[str, str]] = []

    def report(self, *, title: str, label: str, body: str) -> str:
        self.reports.append({"title": title, "label": label, "body": body})
        return f"https://github.com/o/r/issues/{len(self.reports)}"


class FakeHub:
    """Publishes into a local folder, keyed by release, and loads back from it."""

    repo_id = "someone/nem-forecast-vic1"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.published: list[str] = []

    def publish(self, directory: Path, *, version: str, release: str) -> str:
        shutil.copytree(directory, self.root / release)
        self.published.append(release)
        return f"https://huggingface.co/{self.repo_id}/tree/{release}"

    def fetch(self, release: str) -> DirectModel:
        return load_model(self.root / release)


@pytest.fixture
def engine(pg_engine: Engine) -> Engine:
    apply_migrations(pg_engine)
    records = [
        DispatchInterval(settlement_date=ts, region=Region.VIC1, demand_mw=mw, source_file="t")
        for ts, mw in dispatch(DATA).iter_rows()
    ]
    with pg_engine.begin() as conn:
        upsert_intervals(conn, records)
    seasonal_naive = next(b() for b in BASELINES if b().name == "seasonal_naive")
    result = run_backtest(
        DATA, [seasonal_naive, RidgeForecaster()], BacktestConfig(test_days=7, refit_every_days=7)
    )
    save_results(pg_engine, result, region=Region.VIC1, git_sha=SHA, started_at=NOW)
    return pg_engine


def loaded(model: DirectModel, version: str = "v1") -> LoadedModel:
    assert model.trained_on is not None
    return LoadedModel(
        model=model,
        name=model.name,
        version=version,
        region=Region.VIC1,
        max_horizon=max(model.horizons),
        trained_to=model.trained_on[1],
        source="test",
    )


def ridge(history: pl.DataFrame) -> RidgeForecaster:
    model = RidgeForecaster()
    model.fit(history)
    return model


def log_forecasts(engine: Engine, *, days: int, offset: float, version: str) -> None:
    """A forecast from every half-hour of the last ``days``, each ``offset`` MW off the actual."""
    actual = dict(DATA.iter_rows())
    rows = []
    for k in range(days * 48):
        origin = DATA_END - k * PERIOD
        request = uuid.uuid4()
        for h in range(1, 49):
            point = actual.get(origin + h * PERIOD, 5000.0) + offset
            rows.append(
                {
                    "request": request,
                    "origin": origin,
                    "target": origin + h * PERIOD,
                    "h": h,
                    "mw": point,
                    "lo": point - 300,
                    "hi": point + 300,
                    "version": version,
                    "served": origin + timedelta(minutes=4),
                }
            )
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO predictions (request_id, region, forecast_origin, target_period, "
                "horizon, predicted_mw, lower_mw, upper_mw, model_version, served_at) VALUES "
                "(:request, 'VIC1', :origin, :target, :h, :mw, :lo, :hi, :version, :served)"
            ),
            rows,
        )


# -- monitoring ---------------------------------------------------------------------------------


def test_monitor_scores_checks_publishes_and_raises_an_issue(
    engine: Engine, tmp_path: Path
) -> None:
    served = loaded(ridge(DATA.filter(pl.col("period_end") <= START + timedelta(days=60))))
    log_forecasts(engine, days=5, offset=900.0, version=served.version)  # badly wrong forecasts
    issues = FakeIssues()

    outcome = run_monitor(
        engine,
        served,
        region=Region.VIC1,
        now=NOW,
        site_dir=tmp_path / "site",
        report_url="https://o.github.io/r/drift_report.html",
        run_url="https://github.com/o/r/actions/runs/1",
        thresholds=LOOSE,
        issues=issues,
        git_sha=SHA,
    )

    checks = {c.name: c for c in outcome.checks}
    assert set(checks) == {"realised_error", "feature_drift", "prediction_volume"}
    assert checks["realised_error"].breached
    assert checks["realised_error"].value == pytest.approx(900.0)
    assert checks["prediction_volume"].status == "ok"
    assert outcome.retrain
    assert [r["label"] for r in issues.reports] == ["alert:realised_error"]
    assert outcome.issues == ["https://github.com/o/r/issues/1"]

    status = outcome.status
    assert status["scoring"]["scored"] > 9000
    groups = [row["group"] for row in status["realised_vs_backtest"]]
    assert groups == ["h1", "h2-3", "h4-6", "h7-12", "h13-24", "h25-48", "all"]
    assert len(status["realised_by_horizon"]) == 48
    site = tmp_path / "site"
    for name in ("index.html", "drift_report.html", "verdict.json", "status.json"):
        assert (site / name).exists()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT model_version, alerts, realised FROM monitor_runs")).one()
    assert row.model_version == "v1"
    assert {a["name"] for a in row.alerts} == set(checks)
    assert row.realised["window_days"] == 30


def test_monitor_without_traffic_reports_insufficient_data_and_an_outage(
    engine: Engine, tmp_path: Path
) -> None:
    served = loaded(ridge(DATA.filter(pl.col("period_end") <= START + timedelta(days=60))))

    outcome = run_monitor(
        engine,
        served,
        region=Region.VIC1,
        now=NOW,
        site_dir=tmp_path / "site",
        thresholds=LOOSE,
    )

    checks = {c.name: c for c in outcome.checks}
    assert checks["realised_error"].status == "insufficient data"
    assert checks["prediction_volume"].breached
    assert not outcome.retrain  # an outage is not something retraining fixes
    assert outcome.status["drift"]["prediction_drift"]["drifted"] is None


# -- retraining ---------------------------------------------------------------------------------


def test_a_better_candidate_is_promoted_and_round_trips(engine: Engine, tmp_path: Path) -> None:
    stale = DATA.filter(pl.col("period_end") <= START + timedelta(days=58))
    incumbent = loaded(ridge(stale.with_columns(pl.col("demand") + 400)))  # learned a wrong level
    hub = FakeHub(tmp_path / "hub")

    outcome = run_retrain(
        engine,
        incumbent,
        RidgeForecaster,
        region=Region.VIC1,
        started=NOW,
        trigger="alert: realised_error",
        output_dir=tmp_path / "artifacts",
        publisher=hub,
        git_sha=SHA,
    )

    assert outcome.decision == "promoted"
    assert outcome.comparison is not None
    assert outcome.comparison.wins
    assert outcome.release == release_tag(SHA, DATA_END) == "20250816-0000-0123456"
    assert hub.published == [outcome.release]
    saved = json.loads((hub.root / str(outcome.release) / "model.json").read_text())
    assert saved["release"] == outcome.release
    assert saved["trained_on"][1] == DATA_END.isoformat()  # refit on everything after the gate
    card = (hub.root / str(outcome.release) / "README.md").read_text(encoding="utf-8")
    assert "## Promotion gate" in card
    assert f"**Release:** `{outcome.release}`" in card
    with engine.connect() as conn:
        run = conn.execute(text("SELECT decision, release, trigger FROM retrain_runs")).one()
        model_run = conn.execute(
            text("SELECT hf_model_version FROM model_runs WHERE run_type = 'retrain'")
        ).one()
    assert (run.decision, run.release, run.trigger) == (
        "promoted",
        outcome.release,
        "alert: realised_error",
    )
    assert model_run.hf_model_version == outcome.release


def test_a_candidate_that_does_not_win_everywhere_is_rejected(
    engine: Engine, tmp_path: Path
) -> None:
    cutoff = DATA_END - timedelta(days=14)
    incumbent = loaded(ridge(DATA.filter(pl.col("period_end") <= cutoff)))
    hub = FakeHub(tmp_path / "hub")
    issues = FakeIssues()

    outcome = run_retrain(
        engine,
        incumbent,
        RidgeForecaster,  # the same recipe on the same data: a tie, which is not a win
        region=Region.VIC1,
        started=NOW,
        trigger="manual",
        output_dir=tmp_path / "artifacts",
        publisher=hub,
        issues=issues,
        git_sha=SHA,
    )

    assert outcome.decision == "rejected"
    assert hub.published == []
    assert issues.reports[0]["label"] == "retrain:rejected"
    assert "Regressed: **h1, h2-3" in issues.reports[0]["body"]
    with engine.connect() as conn:
        assert conn.execute(text("SELECT decision FROM retrain_runs")).scalar() == "rejected"


def test_retraining_is_skipped_when_the_incumbent_is_too_fresh(
    engine: Engine, tmp_path: Path
) -> None:
    incumbent = loaded(ridge(DATA.filter(pl.col("period_end") <= DATA_END - timedelta(days=3))))

    outcome = run_retrain(
        engine,
        incumbent,
        RidgeForecaster,
        region=Region.VIC1,
        started=NOW,
        trigger="manual",
        output_dir=tmp_path,
        publisher=FakeHub(tmp_path / "hub"),
        git_sha=SHA,
    )

    assert outcome.decision == "skipped"
    assert "at least 7" in outcome.reason
    with engine.connect() as conn:
        assert conn.execute(text("SELECT decision FROM retrain_runs")).scalar() == "skipped"


def test_a_dry_run_compares_but_records_nothing(engine: Engine, tmp_path: Path) -> None:
    stale = DATA.filter(pl.col("period_end") <= START + timedelta(days=58))
    incumbent = loaded(ridge(stale.with_columns(pl.col("demand") + 400)))

    outcome = run_retrain(
        engine,
        incumbent,
        RidgeForecaster,
        region=Region.VIC1,
        started=NOW,
        trigger="manual",
        output_dir=tmp_path,
        publisher=None,
    )

    assert outcome.decision == "dry-run"
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM retrain_runs")).scalar() == 0


def test_uncommitted_code_is_never_promoted(engine: Engine, tmp_path: Path) -> None:
    stale = DATA.filter(pl.col("period_end") <= START + timedelta(days=58))
    incumbent = loaded(ridge(stale.with_columns(pl.col("demand") + 400)))

    with pytest.raises(PromotionError, match="uncommitted"):
        run_retrain(
            engine,
            incumbent,
            RidgeForecaster,
            region=Region.VIC1,
            started=datetime.now(UTC),
            trigger="manual",
            output_dir=tmp_path,
            publisher=FakeHub(tmp_path / "hub"),
            git_sha=SHA + "-dirty",
        )
