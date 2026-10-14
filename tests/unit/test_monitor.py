"""Monitoring pieces that need no database: alerts, issues, the retrain gate, drift, the site."""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import polars as pl
import pytest
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift

from nem_forecast.features.series import PERIOD
from nem_forecast.models.ridge import RidgeForecaster
from nem_forecast.monitor import alerts, retrain
from nem_forecast.monitor.alerts import BREACHED, INSUFFICIENT, OK, GitHubIssues
from nem_forecast.monitor.drift import (
    DRIFT_FEATURES,
    INPUT_COLUMNS,
    MIN_ROWS,
    DriftResult,
    Thresholds,
    calibrate,
    check_drift,
    column_scores,
    drift_frames,
    normed_wasserstein,
    weekly_scores,
)
from nem_forecast.monitor.scoring import SCORES_SCHEMA, compare_with_backtest
from nem_forecast.monitor.site import render_index, write_site
from nem_forecast.serve.model_store import model_version
from tests.synthetic import START, series

NOW = datetime(2026, 10, 20, 2, 0, tzinfo=UTC)


def backtest_rows(model: str, cells: dict[tuple[int, str], float]) -> pl.DataFrame:
    rows = [
        (model, "model", START, START, "mae", horizon, bucket, value, 100)
        for (horizon, bucket), value in cells.items()
    ]
    return pl.DataFrame(
        rows,
        schema=[
            "model",
            "kind",
            "eval_start",
            "eval_end",
            "metric",
            "horizon",
            "tod_bucket",
            "value",
            "n_obs",
        ],
        orient="row",
    )


def scores_frame(rows: list[tuple[int, datetime, float, float]]) -> pl.DataFrame:
    """(horizon, target_period, predicted, actual) rows as load_scores returns them."""
    return pl.DataFrame(
        [
            (
                "v1",
                target - horizon * PERIOD,
                target,
                horizon,
                predicted,
                predicted - 100,
                predicted + 100,
                actual,
            )
            for horizon, target, predicted, actual in rows
        ],
        schema=SCORES_SCHEMA,
        orient="row",
    )


# -- realised error against the backtest --------------------------------------------------------


def test_backtest_estimate_is_weighted_by_the_live_mix_of_horizon_and_time_of_day() -> None:
    overnight = datetime(2025, 6, 1, 17, 30, tzinfo=UTC)  # 03:30 Melbourne: starts 03:00
    evening = datetime(2025, 6, 2, 8, 30, tzinfo=UTC)  # 18:30 Melbourne: starts 18:00
    backtest = backtest_rows(
        "lightgbm_quantile",
        {
            (1, "00-06 overnight"): 100.0,
            (1, "16-21 evening peak"): 400.0,
            (2, "00-06 overnight"): 150.0,
            (2, "16-21 evening peak"): 500.0,
        },
    )
    # Three h1 forecasts overnight and one in the evening peak; one h2 in the evening peak.
    scores = scores_frame(
        [
            (1, overnight, 5000.0, 5100.0),
            (1, overnight + timedelta(days=1), 5000.0, 5100.0),
            (1, overnight + timedelta(days=2), 5000.0, 5100.0),
            (1, evening, 6000.0, 6500.0),
            (2, evening, 6000.0, 6600.0),
        ]
    )

    table = compare_with_backtest(scores, backtest, model="lightgbm_quantile")

    rows = {row["group"]: row for row in table.iter_rows(named=True)}
    assert rows["h1"]["n"] == 4
    assert rows["h1"]["realised_mae"] == pytest.approx((100 * 3 + 500) / 4)
    assert rows["h1"]["backtest_mae"] == pytest.approx((100 * 3 + 400) / 4)
    assert rows["h2-3"]["backtest_mae"] == pytest.approx(500.0)
    assert rows["all"]["n"] == 5
    assert rows["all"]["ratio"] == pytest.approx((300 + 500 + 600) / (300 + 400 + 500))
    assert list(rows) == ["h1", "h2-3", "all"]


def test_comparison_needs_a_recorded_backtest() -> None:
    scores = scores_frame([(1, datetime(2025, 6, 1, 17, 30, tzinfo=UTC), 1.0, 2.0)])
    with pytest.raises(ValueError, match="no recorded backtest"):
        compare_with_backtest(scores, backtest_rows("ridge", {(1, "all"): 1.0}), model="ridge")


# -- alert checks -------------------------------------------------------------------------------


def test_baseline_mae_reads_the_overall_backtest_figure() -> None:
    backtest = backtest_rows("seasonal_naive", {(0, "all"): 560.0, (1, "all"): 500.0})
    assert alerts.baseline_mae(backtest) == 560.0
    assert alerts.baseline_mae(backtest_rows("ridge", {(0, "all"): 1.0})) is None


def test_realised_error_needs_three_days_of_scores() -> None:
    few = scores_frame([(1, NOW, 5000.0, 6000.0)])

    check = alerts.check_realised_error(few, baseline=560.0)

    assert check.status == INSUFFICIENT
    assert not check.breached


@pytest.mark.parametrize(("error", "status"), [(400.0, OK), (700.0, BREACHED)])
def test_realised_error_against_seasonal_naive(error: float, status: str) -> None:
    targets = [NOW - k * PERIOD for k in range(alerts.MIN_SCORED)]
    scores = scores_frame([(1, t, 5000.0 + error, 5000.0) for t in targets])

    check = alerts.check_realised_error(scores, baseline=560.0)

    assert check.status == status
    assert check.value == pytest.approx(error)
    assert check.retrain


def test_realised_error_without_a_baseline_is_insufficient() -> None:
    assert alerts.check_realised_error(scores_frame([]), baseline=None).status == INSUFFICIENT


def verdict(scores: dict[str, float]) -> dict[str, Any]:
    columns = {
        name: {"score": s, "threshold": 0.1, "drifted": s >= 0.1} for name, s in scores.items()
    }
    return {
        "generated_at": NOW.isoformat(),
        "threshold": 0.1,
        "data_drift": {
            "drifted": any(c["drifted"] for c in columns.values()),
            "drifted_columns": [n for n, c in columns.items() if c["drifted"]],
            "columns": columns,
        },
    }


def test_feature_drift_fires_on_any_single_feature() -> None:
    quiet = alerts.check_feature_drift(verdict({"a": 0.02, "b": 0.05}))
    loud = alerts.check_feature_drift(verdict({"a": 0.02, "b": 0.31}))

    assert quiet.status == OK
    assert loud.status == BREACHED
    assert loud.value == 0.31
    assert "b" in loud.detail


def test_volume_is_fine_with_a_forecast_every_half_hour() -> None:
    served = [NOW - k * PERIOD for k in range(61)]
    check = alerts.check_volume(served, now=NOW)
    assert check.status == OK
    assert check.value == pytest.approx(0.5)
    assert not check.retrain


def test_volume_fires_on_six_silent_hours() -> None:
    served = [NOW - k * PERIOD for k in range(61) if not 4 <= k < 18]  # 7 hours missing

    check = alerts.check_volume(served, now=NOW)

    assert check.status == BREACHED
    assert check.value == pytest.approx(7.5)


def test_volume_fires_when_nothing_was_served_at_all() -> None:
    check = alerts.check_volume([], now=NOW)
    assert check.status == BREACHED
    assert check.value == pytest.approx(30.0)


# -- GitHub issues ------------------------------------------------------------------------------


class FakeGitHub:
    def __init__(self, open_issues: list[dict[str, Any]]) -> None:
        self.open_issues = open_issues
        self.requests: list[tuple[str, str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        assert request.headers["Authorization"] == "Bearer token-for-tests"
        if request.url.path.endswith("/labels"):
            return httpx.Response(422, json={"message": "already_exists"})
        if request.method == "GET":
            return httpx.Response(200, json=self.open_issues)
        if request.url.path.endswith("/comments"):
            return httpx.Response(201, json={"id": 1})
        return httpx.Response(
            201, json={"number": 7, "html_url": "https://github.com/o/r/issues/7"}
        )

    def client(self) -> GitHubIssues:
        transport = httpx.MockTransport(self.handler)
        return GitHubIssues("o/r", "token-for-tests", client=httpx.Client(transport=transport))


def test_a_breach_opens_a_labelled_issue() -> None:
    github = FakeGitHub(open_issues=[])

    url = github.client().report(title="Monitoring alert: x", label="alert:x", body="details")

    assert url == "https://github.com/o/r/issues/7"
    created = [r for r in github.requests if r[0] == "POST" and r[1] == "/repos/o/r/issues"]
    assert created[0][2] == {
        "title": "Monitoring alert: x",
        "body": "details",
        "labels": ["monitoring", "alert:x"],
    }


def test_a_repeat_breach_comments_on_the_open_issue() -> None:
    github = FakeGitHub(open_issues=[{"number": 3, "html_url": "https://github.com/o/r/issues/3"}])

    url = github.client().report(title="Monitoring alert: x", label="alert:x", body="again")

    assert url == "https://github.com/o/r/issues/3"
    posts = [r for r in github.requests if r[0] == "POST" and not r[1].endswith("/labels")]
    assert posts == [("POST", "/repos/o/r/issues/3/comments", {"body": "again"})]


def test_alert_body_links_the_report_and_embeds_the_verdict() -> None:
    check = alerts.check_feature_drift(verdict({"a": 0.4}))

    body = alerts.alert_body(
        check,
        verdict=verdict({"a": 0.4}),
        report_url="https://o.github.io/r/drift_report.html",
        run_url="https://github.com/o/r/actions/runs/1",
    )

    assert "https://o.github.io/r/drift_report.html" in body
    assert "actions/runs/1" in body
    assert '"drifted_columns": [\n      "a"\n    ]' in body
    assert "Retraining was triggered" in body


# -- the retraining gate ------------------------------------------------------------------------

DATA = series(days=75, noise_sd=60.0)
DATA_END = START + 75 * 48 * PERIOD


def test_holdout_is_the_last_two_weeks_when_the_incumbent_is_old() -> None:
    start, end = retrain.plan_holdout(DATA_END, DATA_END - timedelta(days=40))
    assert (start, end) == (DATA_END - timedelta(days=14), DATA_END)


def test_holdout_never_overlaps_the_incumbents_training_data() -> None:
    trained_to = DATA_END - timedelta(days=10)
    start, _ = retrain.plan_holdout(DATA_END, trained_to)
    assert start == trained_to


def test_retraining_is_skipped_when_the_incumbent_is_too_fresh() -> None:
    with pytest.raises(retrain.HoldoutTooShort, match="at least 7"):
        retrain.plan_holdout(DATA_END, DATA_END - timedelta(days=3))


def test_holdout_without_a_known_training_end() -> None:
    assert retrain.plan_holdout(DATA_END, None)[0] == DATA_END - timedelta(days=14)


def shifted(frame: pl.DataFrame, mw: float) -> pl.DataFrame:
    return frame.with_columns(pl.col("demand") + mw)


def test_candidate_beating_every_horizon_group_wins() -> None:
    holdout = retrain.plan_holdout(DATA_END, None)
    incumbent = RidgeForecaster()
    incumbent.fit(shifted(DATA, 400.0).filter(pl.col("period_end") <= holdout[0]))

    comparison = retrain.compare(DATA, incumbent, RidgeForecaster(), holdout=holdout)

    assert comparison.wins
    assert comparison.regressed == []
    assert [g.group for g in comparison.groups] == [
        "h1",
        "h2-3",
        "h4-6",
        "h7-12",
        "h13-24",
        "h25-48",
    ]
    assert sum(g.n for g in comparison.groups) > 0
    assert "| h1 |" in comparison.markdown()


def test_a_tie_is_not_a_win() -> None:
    holdout = retrain.plan_holdout(DATA_END, None)
    incumbent = RidgeForecaster()
    incumbent.fit(DATA.filter(pl.col("period_end") <= holdout[0]))

    comparison = retrain.compare(DATA, incumbent, RidgeForecaster(), holdout=holdout)

    assert not comparison.wins
    assert comparison.regressed == [g.group for g in comparison.groups]
    assert comparison.to_dict()["wins"] is False


# -- drift --------------------------------------------------------------------------------------

LOOSE = Thresholds({column: 1.0 for column in INPUT_COLUMNS}, "test: 1.0 everywhere")


@pytest.fixture(scope="module")
def drift_model() -> RidgeForecaster:
    model = RidgeForecaster()
    model.fit(DATA.filter(pl.col("period_end") <= START + timedelta(days=60)))
    return model


def served_for(frame: pl.DataFrame) -> pl.DataFrame:
    """A logged p50 for every origin in the last week, as load_served returns it."""
    origins = [DATA_END - k * PERIOD for k in range(7 * 48)]
    lookup = dict(frame.iter_rows())
    rows = [
        (origin, h, lookup.get(origin + h * PERIOD, 5000.0))
        for origin in origins
        for h in range(1, 49)
    ]
    schema = pl.Schema(
        {
            "forecast_origin": pl.Datetime("us", "UTC"),
            "horizon": pl.Int16(),
            "predicted_mw": pl.Float64(),
        }
    )
    return pl.DataFrame(rows, schema=schema, orient="row")


def run(
    frame: pl.DataFrame, model: RidgeForecaster, served: pl.DataFrame, thresholds: Thresholds
) -> DriftResult:
    assert model.trained_on is not None
    return check_drift(
        drift_frames(frame, model, served),
        thresholds=thresholds,
        model_name="ridge",
        model_version="v1",
        trained_on=model.trained_on,
        generated_at=NOW,
    )


def test_drift_frames_take_one_row_per_half_hour(drift_model: RidgeForecaster) -> None:
    frames = drift_frames(DATA, drift_model, served_for(DATA))

    assert frames.seasonal
    assert frames.current_inputs.height == 7 * 48 - 1  # origins whose next half-hour exists
    assert frames.current_forecasts.height == 7 * 48 * 48
    assert frames.reference_forecasts.height == min(1000, frames.reference_inputs.height) * 48
    assert frames.reference_window.end <= frames.current_window.start
    assert drift_model.trained_on is not None
    assert frames.reference_window.end <= drift_model.trained_on[1]


def test_no_drift_when_the_data_behaves_like_training(drift_model: RidgeForecaster) -> None:
    result = run(DATA, drift_model, served_for(DATA), LOOSE)

    assert not result.data_drift
    assert result.verdict["prediction_drift"]["drifted"] is False
    assert result.verdict["target_drift"]["drifted"] is False
    assert result.verdict["thresholds"] == "test: 1.0 everywhere"


def test_a_level_shift_is_drift_in_features_and_target(drift_model: RidgeForecaster) -> None:
    moved = DATA.with_columns(
        pl.when(pl.col("period_end") > DATA_END - timedelta(days=10))
        .then(pl.col("demand") + 1500)
        .otherwise(pl.col("demand"))
    )

    result = run(moved, drift_model, served_for(moved), LOOSE)

    assert result.data_drift
    assert "demand_at_origin" in result.verdict["data_drift"]["drifted_columns"]
    assert result.verdict["target_drift"]["drifted"] is True
    assert result.verdict["prediction_drift"]["drifted"] is True


def test_each_column_is_judged_against_its_own_threshold(drift_model: RidgeForecaster) -> None:
    strict = Thresholds(LOOSE.by_column | {"same_slot_prev_week": 0.0001}, "test")

    result = run(DATA, drift_model, served_for(DATA), strict)

    assert result.verdict["data_drift"]["drifted_columns"] == ["same_slot_prev_week"]
    assert result.verdict["data_drift"]["columns"]["same_slot_prev_week"]["threshold"] == 0.0001


def test_prediction_drift_needs_a_day_of_served_forecasts(
    drift_model: RidgeForecaster, tmp_path: Path
) -> None:
    result = run(DATA, drift_model, served_for(DATA).head(MIN_ROWS - 1), LOOSE)
    html, verdict_path = result.save(tmp_path)

    assert result.verdict["prediction_drift"]["drifted"] is None
    assert result.verdict["prediction_drift"]["status"].startswith("insufficient data")
    assert html.stat().st_size > 10_000
    assert json.loads(verdict_path.read_text())["model"]["version"] == "v1"


def test_column_scores_refuse_a_p_value_method() -> None:
    snapshot = {
        "metrics": [
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "x",
                    "method": "K-S p_value",
                    "threshold": 0.05,
                },
                "value": 0.01,
            }
        ]
    }
    with pytest.raises(ValueError, match="unexpected drift method"):
        column_scores(snapshot)


def test_calibration_scores_match_evidently() -> None:
    rng = np.random.default_rng(1)
    reference, current = rng.normal(0, 2, 3000), rng.normal(0.5, 2, 400)
    definition = DataDefinition(numerical_columns=["x"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snapshot = Report([ValueDrift(column="x", method="wasserstein", threshold=0.1)]).run(
            Dataset.from_pandas(pd.DataFrame({"x": current}), data_definition=definition),
            Dataset.from_pandas(pd.DataFrame({"x": reference}), data_definition=definition),
        )

    evidently_score = column_scores(snapshot.dict())["x"]["score"]

    assert normed_wasserstein(reference, current) == pytest.approx(evidently_score, abs=1e-4)


def test_weekly_scores_walk_back_from_the_latest_week() -> None:
    scores = weekly_scores(DATA, weeks=8)

    # 75 days, 28 of them warm-up: the sixth week back has under a week of history before it.
    assert scores.height == 5
    assert scores["week_end"].to_list() == [DATA_END - timedelta(days=7 * k) for k in range(5)]
    assert set(INPUT_COLUMNS) <= set(scores.columns)
    assert (
        scores["reference_rows"].diff().drop_nulls() <= 0
    ).all()  # never more history further back


def test_calibration_hits_the_alarm_rate_where_it_can() -> None:
    rng = np.random.default_rng(3)
    weeks = 200
    frame = pl.DataFrame(
        {"week_end": [DATA_END - timedelta(days=7 * k) for k in range(weeks)]}
        | {column: rng.gamma(2.0, 0.2, weeks) for column in INPUT_COLUMNS}
    )

    result = calibrate(frame, alarm_rate=0.05)

    # Rates come in steps (here 3.5% or 6.5% around the target); the nearest one is taken.
    assert result["achieved_alarm_rate"] == pytest.approx(0.05, abs=0.02)
    assert result["default_alarm_rate"] > 0.9  # the default 0.1 fires nearly every week
    fired = pl.any_horizontal(pl.col(c) > result["thresholds"][c] for c in DRIFT_FEATURES)
    assert frame.select(fired.mean()).item() == pytest.approx(result["achieved_alarm_rate"])


def test_calibration_needs_enough_history() -> None:
    with pytest.raises(ValueError, match="at least 20 weeks"):
        calibrate(pl.DataFrame({"week_end": [NOW]} | {c: [0.1] for c in INPUT_COLUMNS}))


def test_thresholds_load_from_the_calibration_file(tmp_path: Path) -> None:
    path = tmp_path / "drift_vic1.json"
    path.write_text(
        json.dumps(
            {
                "computed_at": "2026-10-14T14:00:00+00:00",
                "weeks": 104,
                "achieved_alarm_rate": 0.058,
                "quantile": 0.991,
                "thresholds": dict.fromkeys(INPUT_COLUMNS, 0.75),
            }
        )
    )

    loaded = Thresholds.load(path)
    missing = Thresholds.load(tmp_path / "absent.json")

    assert loaded.by_column["demand"] == 0.75
    assert loaded.for_forecast() == 0.75
    assert "calibrated 2026-10-14" in loaded.source
    assert set(missing.by_column.values()) == {0.1}
    assert "Evidently default" in missing.source


# -- the published site and versions ------------------------------------------------------------


def test_site_files_and_index(tmp_path: Path) -> None:
    report = tmp_path / "evidently.html"
    report.write_text("<html>report</html>", encoding="utf-8")
    drift = verdict({"demand_at_origin": 0.04}) | {
        "thresholds": "calibrated",
        "current": {"rows": 100},
        "reference": {"rows": 1000, "description": "training window"},
        "prediction_drift": {"drifted": None, "status": "insufficient data: 3 served forecasts"},
        "target_drift": {"drifted": False, "score": 0.02, "threshold": 0.7, "status": "ok"},
    }
    checks = [
        alerts.check_feature_drift(drift),
        alerts.check_volume([], now=NOW),
    ]
    status = {
        "generated_at": NOW.isoformat(),
        "region": "VIC1",
        "model": {"name": "lightgbm_quantile", "version": "abc", "trained_to": NOW.isoformat()},
        "window_days": 30,
        "realised_vs_backtest": [
            {"group": "h1", "n": 10, "realised_mae": 120.0, "backtest_mae": 100.0, "ratio": 1.2}
        ],
        "drift": drift,
    }

    index = write_site(tmp_path / "site", status=status, checks=checks, report_html=report)

    site = index.parent
    assert (site / "drift_report.html").read_text(encoding="utf-8") == "<html>report</html>"
    assert json.loads((site / "verdict.json").read_text())["thresholds"] == "calibrated"
    assert json.loads((site / "status.json").read_text())["region"] == "VIC1"
    html = index.read_text(encoding="utf-8")
    assert "1 alert" in html
    assert "1.20" in html
    assert "insufficient data: 3 served forecasts" in html
    assert render_index(status, [checks[0]]).count("All checks passing") == 1


def test_served_version_prefers_the_release_tag() -> None:
    assert model_version({"version": "0123456789abcdef", "release": None}) == "0123456789ab"
    assert model_version({"version": "0123456789abcdef"}) == "0123456789ab"
    assert model_version({"version": "0123", "release": "20261020-1430-0123456"}) == (
        "20261020-1430-0123456"
    )
