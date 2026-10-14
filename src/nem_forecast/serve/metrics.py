"""Prometheus metrics for the API, on a registry owned by each app instance."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from nem_forecast.serve.model_store import LoadedModel

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "nem_forecast_http_requests",
            "HTTP requests by method, route template and status code.",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "nem_forecast_http_request_duration_seconds",
            "HTTP request latency by method and route template.",
            ["method", "route"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.model_info = Gauge(
            "nem_forecast_model_info",
            "The loaded forecasting model; always 1, identified by its labels.",
            ["name", "version", "region"],
            registry=self.registry,
        )
        self.forecasts = Counter(
            "nem_forecast_forecasts_served",
            "Forecasts returned, by endpoint and whether they came from the cache.",
            ["endpoint", "cache"],
            registry=self.registry,
        )
        self.predictions_logged = Counter(
            "nem_forecast_predictions_logged",
            "Prediction rows written to the predictions table.",
            registry=self.registry,
        )

    def model_loaded(self, loaded: LoadedModel) -> None:
        self.model_info.labels(loaded.name, loaded.version, loaded.region.value).set(1)

    def render(self) -> Response:
        return Response(generate_latest(self.registry), media_type=CONTENT_TYPE_LATEST)

    def install(self, app: FastAPI) -> None:
        @app.middleware("http")
        async def record(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            started = time.perf_counter()
            status = 500
            try:
                response = await call_next(request)
                status = response.status_code
                return response
            finally:
                route = request.scope.get("route")
                template = getattr(route, "path", "unmatched")
                self.requests.labels(request.method, template, str(status)).inc()
                self.latency.labels(request.method, template).observe(time.perf_counter() - started)
