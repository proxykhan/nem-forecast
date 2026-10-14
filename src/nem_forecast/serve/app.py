"""The forecast API.

* ``GET /health``: liveness. Always 200 while the process runs; reports the model and code versions.
* ``GET /ready``: readiness. 200 only when the database answers and the model is loaded.
* ``POST /forecast``: median and 10th/90th percentile forecasts for each half-hour after an origin.
* ``GET /forecast/latest``: the forecast from the latest ingested half-hour, cached 5 minutes.
* ``GET /metrics``: Prometheus metrics.

Every error is an RFC 9457 problem (``application/problem+json``). Every forecast returned has
already been written to the ``predictions`` table.

Run with ``python -m nem_forecast.serve`` (or ``nem-forecast serve``); ``create_app`` is the
uvicorn factory.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy import Engine, text

from nem_forecast import __version__
from nem_forecast.config import Settings, get_settings
from nem_forecast.db import create_db_engine
from nem_forecast.gitinfo import git_sha
from nem_forecast.schemas import ForecastRequest, ForecastResponse
from nem_forecast.serve.forecasting import LATEST_TTL, Clock, ForecastService, utc_now
from nem_forecast.serve.metrics import Metrics
from nem_forecast.serve.model_store import ModelState, ModelStore
from nem_forecast.serve.problems import PROBLEM_JSON, install_problem_handlers

log = logging.getLogger(__name__)

PROBLEM_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"content": {PROBLEM_JSON: {}}, "description": description}
    for code, description in (
        (422, "Invalid or unsupported request (problem details)"),
        (503, "Model loading, unavailable, or the forecast could not be recorded"),
    )
}


def service_sha(settings: Settings) -> str | None:
    """Commit of the running code: baked in at image build, set by Render, or read from git."""
    return settings.git_sha or os.getenv("RENDER_GIT_COMMIT") or git_sha()


def create_app(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    models: ModelStore | None = None,
    clock: Clock = utc_now,
) -> FastAPI:
    settings = settings or get_settings()
    engine = engine or create_db_engine(settings.database_url.get_secret_value())
    metrics = Metrics()
    models = models or ModelStore(settings)
    models.on_loaded(metrics.model_loaded)
    service = ForecastService(
        engine, models, clock=clock, model_wait_seconds=settings.model_wait_seconds
    )
    code_sha = service_sha(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load in the background: liveness answers at once, readiness waits for the model.
        app.state.model_loading = asyncio.create_task(asyncio.to_thread(models.load))
        yield
        engine.dispose()

    app = FastAPI(
        title="nem-forecast",
        version=__version__,
        description="Half-hourly VIC1 electricity demand forecasts with 10th-90th percentile "
        "intervals. Errors are RFC 9457 problem details.",
        lifespan=lifespan,
    )
    install_problem_handlers(app)
    metrics.install(app)

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, Any]:
        loaded = models.loaded
        return {
            "status": "ok",
            "service_git_sha": code_sha,
            "model_state": models.state.value,
            "model": None
            if loaded is None
            else {
                "name": loaded.name,
                "version": loaded.version,
                "region": loaded.region.value,
                "trained_to": loaded.trained_to,
                "max_horizon": loaded.max_horizon,
            },
        }

    @app.get("/ready", tags=["operations"])
    def ready() -> JSONResponse:
        checks = {"database": _check_database(engine), "model": _model_check(models)}
        is_ready = all(value == "ok" for value in checks.values())
        return JSONResponse(
            {"status": "ready" if is_ready else "not ready", "checks": checks},
            status_code=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.post(
        "/forecast",
        response_model=ForecastResponse,
        responses=PROBLEM_RESPONSES,
        tags=["forecasts"],
    )
    def forecast(request: ForecastRequest) -> ForecastResponse:
        response = service.forecast(request)
        metrics.forecasts.labels("/forecast", "miss").inc()
        metrics.predictions_logged.inc(len(response.points))
        return response

    @app.get(
        "/forecast/latest",
        response_model=ForecastResponse,
        responses=PROBLEM_RESPONSES,
        tags=["forecasts"],
    )
    def latest(response: Response) -> ForecastResponse:
        latest = service.latest()
        metrics.forecasts.labels("/forecast/latest", "hit" if latest.from_cache else "miss").inc()
        if not latest.from_cache:  # a cache hit returns the forecast already logged
            metrics.predictions_logged.inc(len(latest.response.points))
        remaining = max(0, int(LATEST_TTL.total_seconds()) - latest.age_seconds)
        response.headers["Cache-Control"] = f"public, max-age={remaining}"
        response.headers["Age"] = str(latest.age_seconds)
        return latest.response

    @app.get("/metrics", tags=["operations"], include_in_schema=False)
    def prometheus() -> Response:
        return metrics.render()

    return app


def _check_database(engine: Engine) -> str:
    for attempt in range(2):  # Neon may need a moment to wake from auto-suspend
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return "ok"
        except Exception as exc:
            if attempt == 1:
                return f"error: {type(exc).__name__}"
    return "error"  # pragma: no cover - loop always returns


def _model_check(models: ModelStore) -> str:
    if models.state is ModelState.READY:
        return "ok"
    if models.state is ModelState.FAILED:
        return f"error: {models.error}"
    if models.state is ModelState.NOT_CONFIGURED:
        return "error: set MODEL_REPO_ID or MODEL_DIR"
    return "loading"
