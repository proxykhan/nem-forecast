"""Scoring served forecasts against actuals in PostgreSQL, including revised actuals."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import polars as pl
import pytest
from sqlalchemy import Engine, text

from nem_forecast.db import apply_migrations
from nem_forecast.features.series import PERIOD
from nem_forecast.ingest.loader import upsert_intervals
from nem_forecast.monitor.scoring import (
    load_scores,
    load_served,
    realised_metrics,
    score_predictions,
)
from nem_forecast.schemas import DispatchInterval, Region
from tests.synthetic import START, dispatch, series

pytestmark = pytest.mark.integration

SERIES = series(days=3, noise_sd=40.0)
DATA_END = START + 3 * 48 * PERIOD
NOW = DATA_END + timedelta(minutes=10)
ORIGIN = START + timedelta(days=1)
ACTUAL = dict(SERIES.iter_rows())


@pytest.fixture
def engine(pg_engine: Engine) -> Engine:
    apply_migrations(pg_engine)
    records = [
        DispatchInterval(settlement_date=ts, region=Region.VIC1, demand_mw=mw, source_file="t")
        for ts, mw in dispatch(SERIES).iter_rows()
    ]
    with pg_engine.begin() as conn:
        upsert_intervals(conn, records)
    return pg_engine


def serve(
    engine: Engine,
    origin: datetime,
    horizons: range,
    *,
    offset: float,
    served_at: datetime,
    version: str = "v1",
) -> uuid.UUID:
    """Log a forecast as the API would: actual + offset at every horizon."""
    request_id = uuid.uuid4()
    rows = []
    for h in horizons:
        target = origin + h * PERIOD
        point = ACTUAL.get(target, 5000.0) + offset
        rows.append(
            {
                "request_id": request_id,
                "region": "VIC1",
                "origin": origin,
                "target": target,
                "h": h,
                "mw": point,
                "lo": point - 200,
                "hi": point + 200,
                "version": version,
                "served_at": served_at,
            }
        )
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO predictions (request_id, region, forecast_origin, target_period, "
                "horizon, predicted_mw, lower_mw, upper_mw, model_version, served_at) VALUES "
                "(:request_id, :region, :origin, :target, :h, :mw, :lo, :hi, :version, :served_at)"
            ),
            rows,
        )
    return request_id


def all_scores(engine: Engine) -> pl.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT s.prediction_id, p.request_id, s.target_period, s.horizon, s.actual_mw, "
                "s.error_mw, s.scored_at, s.superseded_at FROM prediction_scores s "
                "JOIN predictions p ON p.id = s.prediction_id ORDER BY s.target_period, s.id"
            )
        ).all()
    return pl.DataFrame(
        [tuple(r) for r in rows],
        schema=[
            "prediction_id",
            "request_id",
            "target_period",
            "horizon",
            "actual_mw",
            "error_mw",
            "scored_at",
            "superseded_at",
        ],
        orient="row",
    ).with_columns(pl.col("target_period").dt.convert_time_zone("UTC"))


def test_each_forecast_is_scored_once_from_its_first_serving(engine: Engine) -> None:
    first = serve(engine, ORIGIN, range(1, 5), offset=100.0, served_at=ORIGIN + PERIOD)
    serve(engine, ORIGIN, range(1, 5), offset=999.0, served_at=ORIGIN + 2 * PERIOD)

    report = score_predictions(engine, Region.VIC1, now=NOW)

    assert (report.scored, report.rescored, report.unchanged, report.pending) == (4, 0, 0, 0)
    scores = all_scores(engine)
    assert scores.height == 4
    assert set(scores["request_id"]) == {first}
    assert (scores["error_mw"] - 100.0).abs().max() < 1e-6  # type: ignore[operator]


def test_forecasts_wait_for_a_complete_half_hour(engine: Engine) -> None:
    target = ORIGIN + 2 * PERIOD
    with engine.begin() as conn:  # one 5-minute interval of the target half-hour is missing
        conn.execute(
            text("DELETE FROM raw_dispatch WHERE settlement_date = :ts"),
            {"ts": target - timedelta(minutes=10)},
        )
    serve(engine, ORIGIN, range(1, 4), offset=50.0, served_at=ORIGIN)
    serve(engine, DATA_END - PERIOD, range(1, 4), offset=50.0, served_at=DATA_END)  # future

    report = score_predictions(engine, Region.VIC1, now=NOW)

    assert report.scored == 3  # h1 and h3 of the first forecast, h1 of the second
    assert report.pending == 1  # h2: its half-hour has five of six intervals
    assert target not in all_scores(engine)["target_period"].to_list()


def test_a_revised_actual_is_rescored_and_both_scores_are_kept(engine: Engine) -> None:
    serve(engine, ORIGIN, range(1, 5), offset=100.0, served_at=ORIGIN)
    score_predictions(engine, Region.VIC1, now=NOW)
    revised = ORIGIN + 3 * PERIOD
    with engine.begin() as conn:  # AEMO restates one 5-minute interval by +60 MW
        conn.execute(
            text(
                "UPDATE raw_dispatch SET demand_mw = demand_mw + 60, revision = revision + 1 "
                "WHERE settlement_date = :ts"
            ),
            {"ts": revised - timedelta(minutes=5)},
        )
    later = NOW + timedelta(hours=1)

    report = score_predictions(engine, Region.VIC1, now=later)

    assert (report.scored, report.rescored, report.unchanged) == (0, 1, 3)
    history = all_scores(engine).filter(pl.col("target_period") == revised)
    old, new = history.row(0, named=True), history.row(1, named=True)
    assert old["superseded_at"] == later
    assert new["superseded_at"] is None
    assert new["actual_mw"] == pytest.approx(old["actual_mw"] + 10.0)  # 60 MW over six intervals
    assert new["error_mw"] == pytest.approx(old["error_mw"] - 10.0)
    assert new["prediction_id"] == old["prediction_id"]
    current = load_scores(engine, Region.VIC1, start=START, end=later)
    assert current.height == 4  # one current score per forecast


def test_rerunning_without_revisions_changes_nothing(engine: Engine) -> None:
    serve(engine, ORIGIN, range(1, 5), offset=100.0, served_at=ORIGIN)
    score_predictions(engine, Region.VIC1, now=NOW)

    again = score_predictions(engine, Region.VIC1, now=NOW + timedelta(days=1))

    assert (again.scored, again.rescored, again.unchanged) == (0, 0, 4)
    assert all_scores(engine).height == 4


def test_old_unscored_forecasts_are_still_scored(engine: Engine) -> None:
    serve(engine, ORIGIN, range(1, 3), offset=100.0, served_at=ORIGIN)

    # Run a long time later, with a rescore window that no longer reaches the forecast.
    report = score_predictions(engine, Region.VIC1, now=NOW + timedelta(days=60), rescore_days=1)

    assert report.scored == 2


def test_realised_metrics_use_the_backtest_definitions(engine: Engine) -> None:
    serve(engine, ORIGIN, range(1, 5), offset=-80.0, served_at=ORIGIN)
    score_predictions(engine, Region.VIC1, now=NOW)

    metrics = realised_metrics(load_scores(engine, Region.VIC1, start=START, end=NOW))

    overall = metrics.filter(pl.col("horizon") == 0, pl.col("tod_bucket") == "all")
    values = dict(zip(overall["metric"], overall["value"], strict=True))
    assert values["mae"] == pytest.approx(80.0)
    assert values["coverage_80"] == pytest.approx(100.0)  # actual within +-200 of the point
    assert set(metrics["model"]) == {"live"}
    assert metrics.filter(pl.col("tod_bucket") != "all").height > 0


def test_load_served_returns_each_forecast_once(engine: Engine) -> None:
    serve(engine, ORIGIN, range(1, 4), offset=1.0, served_at=ORIGIN)
    serve(engine, ORIGIN, range(1, 4), offset=2.0, served_at=ORIGIN + PERIOD)
    serve(engine, ORIGIN, range(1, 4), offset=3.0, served_at=ORIGIN, version="v2")

    served = load_served(engine, Region.VIC1, model_version="v1", start=START, end=DATA_END).sort(
        "horizon"
    )

    assert served["horizon"].to_list() == [1, 2, 3]
    expected = [ACTUAL[ORIGIN + h * PERIOD] + 1.0 for h in (1, 2, 3)]
    assert served["predicted_mw"].to_list() == pytest.approx(expected)
