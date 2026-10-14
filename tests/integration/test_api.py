"""The forecast API end to end: httpx AsyncClient -> FastAPI -> real PostgreSQL.

A small LightGBM quantile model is trained once on synthetic data; a controllable clock stands in
for "now", so origin rules (not in the future, at most 7 days old) are tested deterministically.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
from asgi_lifespan import LifespanManager
from pydantic import SecretStr
from sqlalchemy import Engine, text

from nem_forecast.config import Settings
from nem_forecast.db import apply_migrations
from nem_forecast.ingest.loader import upsert_intervals
from nem_forecast.models.lgbm import LightGBMQuantile
from nem_forecast.schemas import NEM_TZ, DispatchInterval, Region
from nem_forecast.serve.app import create_app
from nem_forecast.serve.model_store import ModelStore
from nem_forecast.serve.problems import DOCS, PROBLEM_JSON
from tests.synthetic import START, dispatch, series

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

DAYS = 45
SERIES = series(days=DAYS, noise_sd=60.0)
DATA_END = START + DAYS * 48 * timedelta(minutes=30)  # last period_end in the data
NOW = DATA_END + timedelta(minutes=17)  # the half-hour after DATA_END is still in progress
VERSION = "0123456789abcdef0123456789abcdef01234567"


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model = LightGBMQuantile({"num_boost_round": 30, "num_leaves": 15, "min_data_in_leaf": 20})
    model.fit(SERIES)
    directory = tmp_path_factory.mktemp("model")
    model.save(directory, version=VERSION)
    return directory


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def database(pg_engine: Engine) -> Engine:
    apply_migrations(pg_engine)
    records = [
        DispatchInterval(settlement_date=ts, region=Region.VIC1, demand_mw=mw, source_file="t")
        for ts, mw in dispatch(SERIES).iter_rows()
    ]
    with pg_engine.begin() as conn:
        upsert_intervals(conn, records)
    return pg_engine


@pytest.fixture
def clock() -> Clock:
    return Clock(NOW)


def make_settings(engine: Engine, **overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        database_url=SecretStr(engine.url.render_as_string(hide_password=False)),
        git_sha="feedface",
        **overrides,
    )


@pytest.fixture
async def client(
    database: Engine, model_dir: Path, clock: Clock
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(make_settings(database, model_dir=model_dir), engine=database, clock=clock)
    async with LifespanManager(app):
        await app.state.model_loading
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as http:
            yield http


def rows(engine: Engine, sql: str, **params: Any) -> list[Any]:
    with engine.connect() as conn:
        return list(conn.execute(text(sql), params).all())


def market(ts: datetime) -> str:
    return ts.astimezone(NEM_TZ).isoformat()


def assert_problem(response: httpx.Response, status: int, slug: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM_JSON
    body: dict[str, Any] = response.json()
    assert body["type"] == f"{DOCS}#{slug}"
    assert body["status"] == status
    assert body["title"]
    assert body["detail"]
    return body


# -- liveness and readiness --------------------------------------------------------------------


async def test_health_reports_model_and_code_versions(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service_git_sha"] == "feedface"
    assert body["model"]["version"] == VERSION[:12]
    assert body["model"]["name"] == "lightgbm_quantile"
    assert body["model"]["max_horizon"] == 48


async def test_ready_when_database_and_model_answer(client: httpx.AsyncClient) -> None:
    response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {"database": "ok", "model": "ok"}}


async def test_alive_but_not_ready_without_a_model(database: Engine, clock: Clock) -> None:
    app = create_app(make_settings(database), engine=database, clock=clock)
    async with LifespanManager(app):
        await app.state.model_loading
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://api"
        ) as http:
            health, ready, forecast = (
                await http.get("/health"),
                await http.get("/ready"),
                await http.get("/forecast/latest"),
            )

    assert health.status_code == 200
    assert health.json()["model"] is None
    assert ready.status_code == 503
    assert ready.json()["checks"] == {
        "database": "ok",
        "model": "error: set MODEL_REPO_ID or MODEL_DIR",
    }
    assert_problem(forecast, 503, "model-unavailable")


async def test_forecasts_wait_for_a_model_that_is_still_loading(
    database: Engine, model_dir: Path, clock: Clock
) -> None:
    store = ModelStore(make_settings(database, model_dir=model_dir))
    store.load = lambda: None  # type: ignore[method-assign]  # never finishes loading
    app = create_app(make_settings(database), engine=database, models=store, clock=clock)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as http,
    ):
        response = await http.post("/forecast", json={"forecast_origin": market(DATA_END)})
        ready = await http.get("/ready")

    body = assert_problem(response, 503, "model-loading")
    assert response.headers["retry-after"] == "10"
    assert "loading" in body["detail"]
    assert ready.json()["checks"]["model"] == "loading"


# -- forecasts -----------------------------------------------------------------------------------


async def test_forecast_returns_quantiles_for_every_period(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/forecast", json={"region": "VIC1", "forecast_origin": market(DATA_END), "horizon": 6}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["region"] == "VIC1"
    assert body["model_version"] == VERSION[:12]
    assert datetime.fromisoformat(body["forecast_origin"]) == DATA_END
    assert [p["horizon"] for p in body["points"]] == [1, 2, 3, 4, 5, 6]
    for k, point in enumerate(body["points"], start=1):
        assert datetime.fromisoformat(point["target_period"]) == DATA_END + k * timedelta(
            minutes=30
        )
        assert point["lower_mw"] <= point["predicted_mw"] <= point["upper_mw"]
        assert 2000 < point["predicted_mw"] < 7000


async def test_every_served_forecast_is_in_the_predictions_table(
    client: httpx.AsyncClient, database: Engine
) -> None:
    response = await client.post(
        "/forecast", json={"forecast_origin": market(DATA_END), "horizon": 12}
    )
    body = response.json()

    stored = rows(
        database,
        "SELECT horizon, target_period, predicted_mw, lower_mw, upper_mw, model_version, "
        "forecast_origin, region FROM predictions WHERE request_id = :id ORDER BY horizon",
        id=body["request_id"],
    )
    assert len(stored) == 12
    for row, point in zip(stored, body["points"], strict=True):
        assert row.horizon == point["horizon"]
        assert row.target_period == datetime.fromisoformat(point["target_period"])
        assert (row.predicted_mw, row.lower_mw, row.upper_mw) == (
            point["predicted_mw"],
            point["lower_mw"],
            point["upper_mw"],
        )
        assert row.model_version == VERSION[:12]
        assert row.forecast_origin == DATA_END
        assert row.region == "VIC1"


async def test_horizon_defaults_to_a_day(client: httpx.AsyncClient) -> None:
    response = await client.post("/forecast", json={"forecast_origin": market(DATA_END)})

    assert len(response.json()["points"]) == 48


async def test_a_forecast_that_cannot_be_recorded_is_not_served(
    client: httpx.AsyncClient, database: Engine
) -> None:
    with database.begin() as conn:
        conn.execute(text("ALTER TABLE predictions RENAME TO predictions_offline"))

    response = await client.post("/forecast", json={"forecast_origin": market(DATA_END)})

    body = assert_problem(response, 503, "prediction-not-recorded")
    assert "points" not in body
    assert response.headers["retry-after"] == "5"


async def test_latest_forecast_is_cached_for_five_minutes(
    client: httpx.AsyncClient, database: Engine, clock: Clock
) -> None:
    first = await client.get("/forecast/latest")
    clock.now += timedelta(minutes=2)
    cached = await client.get("/forecast/latest")
    clock.now += timedelta(minutes=4)
    fresh = await client.get("/forecast/latest")

    assert first.status_code == 200
    assert datetime.fromisoformat(first.json()["forecast_origin"]) == DATA_END
    assert first.headers["cache-control"] == "public, max-age=300"
    assert cached.json()["request_id"] == first.json()["request_id"]
    assert cached.headers["age"] == "120"
    assert cached.headers["cache-control"] == "public, max-age=180"
    assert fresh.json()["request_id"] != first.json()["request_id"]
    # Two computed forecasts were logged; the cache hit added nothing.
    assert rows(database, "SELECT count(DISTINCT request_id), count(*) FROM predictions") == [
        (2, 96)
    ]


# -- validation, as RFC 9457 problems --------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "slug"),
    [
        ({"forecast_origin": market(DATA_END + timedelta(hours=1))}, "origin-in-future"),
        ({"forecast_origin": market(DATA_END - timedelta(days=8))}, "origin-too-old"),
        ({"forecast_origin": market(DATA_END), "horizon": 60}, "horizon-not-supported"),
        ({"forecast_origin": market(DATA_END), "horizon": 97}, "invalid-request"),
        ({"forecast_origin": market(DATA_END), "horizon": 0}, "invalid-request"),
        ({"forecast_origin": market(DATA_END), "region": "WA1"}, "invalid-request"),
        ({"forecast_origin": market(DATA_END), "region": "NSW1"}, "region-not-supported"),
        ({"forecast_origin": market(DATA_END + timedelta(minutes=10))}, "invalid-request"),
        ({"forecast_origin": "2026-10-13T12:00:00"}, "invalid-request"),  # no time zone
        ({"horizon": 4}, "invalid-request"),
        ({"forecast_origin": market(DATA_END), "unexpected": 1}, "invalid-request"),
    ],
)
async def test_invalid_requests_are_problems_and_log_nothing(
    client: httpx.AsyncClient, database: Engine, payload: dict[str, Any], slug: str
) -> None:
    response = await client.post("/forecast", json=payload)

    assert_problem(response, 422, slug)
    assert rows(database, "SELECT count(*) FROM predictions") == [(0,)]


async def test_validation_problems_list_each_error(client: httpx.AsyncClient) -> None:
    response = await client.post("/forecast", json={"horizon": 500, "region": "WA1"})

    body = assert_problem(response, 422, "invalid-request")
    locations = sorted(tuple(e["location"]) for e in body["errors"])
    assert locations == [("body", "forecast_origin"), ("body", "horizon"), ("body", "region")]
    assert body["instance"] == "/forecast"


async def test_horizon_limit_comes_from_the_model(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/forecast", json={"forecast_origin": market(DATA_END), "horizon": 49}
    )

    body = assert_problem(response, 422, "horizon-not-supported")
    assert body["max_horizon"] == 48


async def test_origin_without_data_yet_points_to_the_latest_available(
    client: httpx.AsyncClient, clock: Clock
) -> None:
    clock.now = DATA_END + timedelta(hours=3)

    response = await client.post(
        "/forecast", json={"forecast_origin": market(DATA_END + timedelta(hours=2))}
    )

    body = assert_problem(response, 422, "origin-not-ingested")
    assert body["latest_available"] == f"{DATA_END.astimezone(NEM_TZ):%Y-%m-%d %H:%M} AEST"


async def test_unknown_routes_are_problems_too(client: httpx.AsyncClient) -> None:
    assert_problem(await client.get("/nope"), 404, "not-found")
    assert_problem(await client.get("/forecast"), 405, "method-not-allowed")


# -- metrics -------------------------------------------------------------------------------------


async def test_metrics_count_requests_latency_and_model_version(client: httpx.AsyncClient) -> None:
    await client.post("/forecast", json={"forecast_origin": market(DATA_END), "horizon": 3})
    await client.post("/forecast", json={"forecast_origin": market(DATA_END), "horizon": 99})
    await client.get("/forecast/latest")
    await client.get("/forecast/latest")

    response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    lines = response.text.splitlines()
    expected = [
        'nem_forecast_http_requests_total{method="POST",route="/forecast",status="200"} 1.0',
        'nem_forecast_http_requests_total{method="POST",route="/forecast",status="422"} 1.0',
        'nem_forecast_http_requests_total{method="GET",route="/forecast/latest",status="200"} 2.0',
        'nem_forecast_forecasts_served_total{cache="hit",endpoint="/forecast/latest"} 1.0',
        "nem_forecast_model_info"
        f'{{name="lightgbm_quantile",region="VIC1",version="{VERSION[:12]}"}} 1.0',
        "nem_forecast_predictions_logged_total 51.0",  # 3 + 48; the cache hit logs nothing
    ]
    for line in expected:
        assert line in lines
    assert any(
        line.startswith('nem_forecast_http_request_duration_seconds_bucket{le="0.5",method="POST"')
        for line in lines
    )


async def test_openapi_documents_the_forecast_endpoints(client: httpx.AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()

    assert {"/health", "/ready", "/forecast", "/forecast/latest"} <= set(schema["paths"])
    assert PROBLEM_JSON in schema["paths"]["/forecast"]["post"]["responses"]["422"]["content"]


def test_series_fixture_ends_where_the_tests_expect() -> None:
    assert SERIES["period_end"][-1] == DATA_END
    assert DATA_END.tzinfo == UTC
    assert not SERIES["demand"].is_null().any()
    assert isinstance(SERIES, pl.DataFrame)


@pytest.fixture(autouse=True)
def _no_real_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing in these tests may reach Hugging Face."""

    def refuse(*args: object, **kwargs: object) -> str:
        raise AssertionError("tests must not download models")

    monkeypatch.setattr("nem_forecast.serve.model_store.snapshot_download", refuse)


async def test_history_windows_start_on_a_half_hour_boundary(
    client: httpx.AsyncClient, clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    """/forecast/latest measures 29 days back from "now", which is rarely on a boundary."""
    clock.now = NOW + timedelta(minutes=8)

    with caplog.at_level("WARNING"):
        response = await client.get("/forecast/latest")

    assert response.status_code == 200
    assert "incomplete" not in caplog.text
