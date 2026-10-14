"""Turn a validated request into a forecast that is logged before it is returned.

Every served forecast is written to ``predictions`` (one row per target period, keyed by a
``request_id``) inside the request. If the write fails the client gets a 503 problem instead of
the numbers: a forecast that monitoring can never score is worse than no forecast.

Features come from the same builder and leakage check as training, over the 29 days of history
the longest lag needs. Neon suspends idle databases, so reads are retried once to ride out a
cold start.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import polars as pl
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError

from nem_forecast.features.build import assert_no_leakage, build_features
from nem_forecast.features.series import load_half_hourly
from nem_forecast.schemas import (
    NEM_TZ,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    Region,
)
from nem_forecast.serve.model_store import LoadedModel, ModelStore
from nem_forecast.serve.problems import Problem

log = logging.getLogger(__name__)

HISTORY = timedelta(days=29)  # four weeks of same-weekday lags, plus a day of margin
MAX_ORIGIN_AGE = timedelta(days=7)
LATEST_TTL = timedelta(minutes=5)

Clock = Callable[[], datetime]

_INSERT = """
INSERT INTO predictions (
    request_id, region, forecast_origin, target_period, horizon,
    predicted_mw, lower_mw, upper_mw, model_version
) VALUES (
    :request_id, :region, :forecast_origin, :target_period, :horizon,
    :predicted_mw, :lower_mw, :upper_mw, :model_version
)
"""


@dataclass(frozen=True)
class CachedForecast:
    response: ForecastResponse
    computed_at: datetime


@dataclass(frozen=True)
class LatestForecast:
    response: ForecastResponse
    age_seconds: int
    from_cache: bool


def utc_now() -> datetime:
    return datetime.now(UTC)


class ForecastService:
    def __init__(
        self,
        engine: Engine,
        models: ModelStore,
        *,
        clock: Clock = utc_now,
        model_wait_seconds: float = 0.0,
    ) -> None:
        self._engine = engine
        self._models = models
        self._clock = clock
        self._model_wait = model_wait_seconds
        self._latest: dict[Region, CachedForecast] = {}
        self._latest_lock = threading.Lock()

    # -- public ----------------------------------------------------------------------------------

    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        loaded = self._models.require(wait=self._model_wait)
        origin = request.forecast_origin.astimezone(UTC)
        self._validate(request.region, origin, request.horizon, loaded)
        series = self._history(request.region, end=origin)
        if series.is_empty() or series["period_end"][-1] != origin or series["demand"][-1] is None:
            latest = series["period_end"][-1] if not series.is_empty() else None
            raise Problem(
                422,
                "origin-not-ingested",
                "No data at the forecast origin yet",
                f"Demand for the half-hour ending {_market(origin)} has not been ingested."
                + (f" The latest complete half-hour ends {_market(latest)}." if latest else ""),
                latest_available=_market(latest) if latest else None,
            )
        return self._predict_and_log(series, origin, request.horizon, request.region, loaded)

    def latest(self, region: Region = Region.VIC1) -> LatestForecast:
        """The forecast from the latest ingested half-hour, recomputed (and logged) every 5 min."""
        loaded = self._models.require(wait=self._model_wait)
        self._validate_region(region, loaded)
        now = self._clock()
        with self._latest_lock:
            cached = self._latest.get(region)
            if (
                cached is not None
                and cached.response.model_version == loaded.version
                and now - cached.computed_at < LATEST_TTL
            ):
                age = int((now - cached.computed_at).total_seconds())
                return LatestForecast(cached.response, age, from_cache=True)
            series = self._history(region, end=now)
            if series.is_empty():
                raise Problem(
                    503,
                    "no-recent-data",
                    "No recent data",
                    f"No {region} demand in the last 29 days.",
                )
            origin = series["period_end"][-1]
            response = self._predict_and_log(series, origin, loaded.max_horizon, region, loaded)
            self._latest[region] = CachedForecast(response, now)
            return LatestForecast(response, 0, from_cache=False)

    # -- validation --------------------------------------------------------------------------------

    def _validate(
        self, region: Region, origin: datetime, horizon: int, loaded: LoadedModel
    ) -> None:
        self._validate_region(region, loaded)
        now = self._clock()
        if origin > now:
            raise Problem(
                422,
                "origin-in-future",
                "Forecast origin is in the future",
                f"{_market(origin)} is after the current time {_market(now)}; "
                "a forecast can only start from data that exists.",
            )
        if origin < now - MAX_ORIGIN_AGE:
            raise Problem(
                422,
                "origin-too-old",
                "Forecast origin is too old",
                f"Origins may be at most {MAX_ORIGIN_AGE.days} days in the past; "
                f"{_market(origin)} is older.",
            )
        if horizon > loaded.max_horizon:
            raise Problem(
                422,
                "horizon-not-supported",
                "Horizon beyond what the model was evaluated for",
                f"The served model ({loaded.name} {loaded.version}) was trained and backtested for "
                f"up to {loaded.max_horizon} half-hours ahead; {horizon} was requested.",
                max_horizon=loaded.max_horizon,
            )

    @staticmethod
    def _validate_region(region: Region, loaded: LoadedModel) -> None:
        if region is not loaded.region:
            raise Problem(
                422,
                "region-not-supported",
                "Region not supported",
                f"The served model forecasts {loaded.region} only; {region} was requested.",
                supported_regions=[loaded.region.value],
            )

    # -- work --------------------------------------------------------------------------------------

    def _history(self, region: Region, *, end: datetime) -> pl.DataFrame:
        # Start on a half-hour boundary, or the window would cut its first period in two and
        # report it as incomplete.
        start = _floor_to_period(end - HISTORY)
        return _retry_once(lambda: load_half_hourly(self._engine, region, start=start, end=end))

    def _predict_and_log(
        self,
        series: pl.DataFrame,
        origin: datetime,
        horizon: int,
        region: Region,
        loaded: LoadedModel,
    ) -> ForecastResponse:
        features = build_features(
            series, [origin], horizons=tuple(range(1, horizon + 1)), with_label=False
        )
        assert_no_leakage(features)
        predicted = self._models.predict(features).sort("horizon")
        response = ForecastResponse(
            region=region,
            forecast_origin=origin.astimezone(NEM_TZ),
            model_version=loaded.version,
            request_id=uuid.uuid4(),
            generated_at=self._clock(),
            points=[
                ForecastPoint(
                    target_period=row["target"].astimezone(NEM_TZ),
                    horizon=row["horizon"],
                    predicted_mw=round(row["p50"], 2),
                    lower_mw=round(row["p10"], 2),
                    upper_mw=round(row["p90"], 2),
                )
                for row in predicted.iter_rows(named=True)
            ],
        )
        self._record(response)
        return response

    def _record(self, response: ForecastResponse) -> None:
        rows = [
            {
                "request_id": response.request_id,
                "region": response.region.value,
                "forecast_origin": response.forecast_origin,
                "target_period": point.target_period,
                "horizon": point.horizon,
                "predicted_mw": point.predicted_mw,
                "lower_mw": point.lower_mw,
                "upper_mw": point.upper_mw,
                "model_version": response.model_version,
            }
            for point in response.points
        ]
        try:
            with self._engine.begin() as conn:
                conn.execute(text(_INSERT), rows)
        except Exception as exc:
            log.exception("could not record forecast %s", response.request_id)
            raise Problem(
                503,
                "prediction-not-recorded",
                "Forecast could not be recorded",
                "The forecast was computed but could not be saved for monitoring, so it is not "
                "being returned. Retry shortly.",
                headers={"Retry-After": "5"},
            ) from exc


def _floor_to_period(ts: datetime) -> datetime:
    utc = ts.astimezone(UTC)
    return utc.replace(minute=utc.minute - utc.minute % 30, second=0, microsecond=0)


def _retry_once[R](read: Callable[[], R]) -> R:
    try:
        return read()
    except OperationalError:
        log.warning("database read failed, retrying once (Neon may be waking up)")
        return read()


def _market(ts: datetime) -> str:
    return f"{ts.astimezone(NEM_TZ):%Y-%m-%d %H:%M} AEST"
