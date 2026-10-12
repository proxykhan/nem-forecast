"""The repository's real migrations, applied to a real PostgreSQL database."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError

from nem_forecast.db import apply_migrations, discover_migrations
from nem_forecast.schemas import NEM_TZ

pytestmark = pytest.mark.integration

SETTLEMENT = datetime(2026, 10, 12, 14, 5, tzinfo=NEM_TZ)


def insert_dispatch(engine: Engine, demand_mw: float) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO raw_dispatch (settlement_date, region, demand_mw, rrp, source_file) "
                "VALUES (:ts, 'VIC1', :demand, 85.2, 'fixture.zip')"
            ),
            {"ts": SETTLEMENT, "demand": demand_mw},
        )


def test_all_migrations_apply_cleanly_and_only_once(pg_engine: Engine) -> None:
    applied = apply_migrations(pg_engine)

    assert [m.version for m in applied] == [m.version for m in discover_migrations()]
    assert apply_migrations(pg_engine) == []
    assert {
        "raw_dispatch",
        "features",
        "predictions",
        "model_runs",
        "model_run_metrics",
        "schema_migrations",
    } <= set(inspect(pg_engine).get_table_names())


def test_raw_dispatch_is_unique_per_interval_and_region(pg_engine: Engine) -> None:
    apply_migrations(pg_engine)
    insert_dispatch(pg_engine, 4875.3)

    with pytest.raises(IntegrityError, match="raw_dispatch_settlement_region_uq"):
        insert_dispatch(pg_engine, 4900.0)


def test_raw_dispatch_starts_at_revision_one_and_keeps_market_time(pg_engine: Engine) -> None:
    apply_migrations(pg_engine)
    insert_dispatch(pg_engine, 4875.3)

    with pg_engine.connect() as conn:
        revision, settlement = conn.execute(
            text("SELECT revision, settlement_date FROM raw_dispatch")
        ).one()

    assert revision == 1
    assert settlement == SETTLEMENT  # same instant, whatever the session time zone


def test_raw_dispatch_rejects_unknown_region(pg_engine: Engine) -> None:
    apply_migrations(pg_engine)

    with pytest.raises(IntegrityError, match="check"), pg_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO raw_dispatch (settlement_date, region, demand_mw, source_file) "
                "VALUES (now(), 'WA1', 1.0, 'x')"
            )
        )


def test_features_cannot_use_data_after_their_own_timestamp(pg_engine: Engine) -> None:
    apply_migrations(pg_engine)

    with pytest.raises(IntegrityError, match="check"), pg_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO features (settlement_date, region, feature_version, data_cutoff, "
                "feature_values) VALUES (:ts, 'VIC1', 'v1', :ts + interval '30 minutes', '{}')"
            ),
            {"ts": SETTLEMENT},
        )


def test_predictions_reject_crossed_intervals(pg_engine: Engine) -> None:
    apply_migrations(pg_engine)

    with pytest.raises(IntegrityError, match="check"), pg_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO predictions (request_id, region, forecast_origin, target_period, "
                "horizon, predicted_mw, lower_mw, upper_mw, model_version) VALUES "
                "(gen_random_uuid(), 'VIC1', :ts, :ts + interval '30 minutes', 1, "
                "5000, 5200, 4800, 'abc1234')"
            ),
            {"ts": SETTLEMENT},
        )


def test_model_run_metrics_cascade_with_their_run(pg_engine: Engine) -> None:
    apply_migrations(pg_engine)

    with pg_engine.begin() as conn:
        run_id = conn.execute(
            text(
                "INSERT INTO model_runs (model_name, model_kind, run_type, region) "
                "VALUES ('seasonal_naive', 'baseline', 'backtest', 'VIC1') RETURNING run_id"
            )
        ).scalar_one()
        conn.execute(
            text(
                "INSERT INTO model_run_metrics (run_id, metric, horizon, value, n_obs) "
                "VALUES (:run_id, 'mae', 1, 95.4, 1000), (:run_id, 'mae', 48, 210.7, 1000)"
            ),
            {"run_id": run_id},
        )
        conn.execute(text("DELETE FROM model_runs WHERE run_id = :run_id"), {"run_id": run_id})
        remaining = conn.execute(text("SELECT count(*) FROM model_run_metrics")).scalar_one()

    assert remaining == 0
