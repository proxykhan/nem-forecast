"""Core domain objects shared by ingestion, modelling, serving and monitoring.

All timestamps are timezone-aware. AEMO publishes in market time, which is fixed AEST
(UTC+10) all year round, so ``NEM_TZ`` is a fixed offset and never follows Victorian DST.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

NEM_TZ = timezone(timedelta(hours=10), name="AEST")
DISPATCH_INTERVAL = timedelta(minutes=5)
FORECAST_PERIOD = timedelta(minutes=30)
MAX_HORIZON = 96  # 48 hours of half-hour periods


class Region(StrEnum):
    NSW1 = "NSW1"
    QLD1 = "QLD1"
    SA1 = "SA1"
    TAS1 = "TAS1"
    VIC1 = "VIC1"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_aligned(ts: datetime, step: timedelta) -> bool:
    since_midnight = ts - ts.replace(hour=0, minute=0, second=0, microsecond=0)
    return since_midnight % step == timedelta(0)


class _Domain(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class DispatchInterval(_Domain):
    """One regional demand/price observation, as parsed from a NEMWeb file."""

    settlement_date: AwareDatetime
    region: Region
    demand_mw: float = Field(allow_inf_nan=False)
    rrp: float | None = Field(default=None, allow_inf_nan=False)
    source_file: str = Field(min_length=1)
    ingested_at: AwareDatetime = Field(default_factory=_utc_now)

    @field_validator("settlement_date")
    @classmethod
    def _on_dispatch_boundary(cls, value: datetime) -> datetime:
        if not _is_aligned(value, DISPATCH_INTERVAL):
            raise ValueError(f"settlement_date {value.isoformat()} is not on a 5-minute boundary")
        return value


class ForecastRequest(_Domain):
    region: Region = Region.VIC1
    forecast_origin: AwareDatetime
    horizon: int = Field(default=48, ge=1, le=MAX_HORIZON)

    @field_validator("forecast_origin")
    @classmethod
    def _on_period_boundary(cls, value: datetime) -> datetime:
        if not _is_aligned(value, FORECAST_PERIOD):
            raise ValueError(f"forecast_origin {value.isoformat()} is not on a 30-minute boundary")
        return value


class ForecastPoint(_Domain):
    """Forecast for one half-hour period: the median plus a 10th-90th percentile interval."""

    target_period: AwareDatetime
    horizon: int = Field(ge=1, le=MAX_HORIZON)
    predicted_mw: float = Field(allow_inf_nan=False)
    lower_mw: float = Field(allow_inf_nan=False)
    upper_mw: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _quantiles_ordered(self) -> Self:
        if not self.lower_mw <= self.predicted_mw <= self.upper_mw:
            raise ValueError(
                f"quantiles must satisfy lower <= predicted <= upper, got "
                f"{self.lower_mw} / {self.predicted_mw} / {self.upper_mw}"
            )
        return self


class ForecastResponse(_Domain):
    region: Region
    forecast_origin: AwareDatetime
    model_version: str = Field(min_length=1)
    lower_quantile: float = Field(default=0.1, gt=0, lt=0.5)
    upper_quantile: float = Field(default=0.9, gt=0.5, lt=1)
    generated_at: AwareDatetime = Field(default_factory=_utc_now)
    points: list[ForecastPoint] = Field(min_length=1, max_length=MAX_HORIZON)

    @model_validator(mode="after")
    def _points_follow_origin(self) -> Self:
        for expected_horizon, point in enumerate(self.points, start=1):
            expected_target = self.forecast_origin + expected_horizon * FORECAST_PERIOD
            if point.horizon != expected_horizon or point.target_period != expected_target:
                raise ValueError(
                    f"point {expected_horizon} should be horizon {expected_horizon} at "
                    f"{expected_target.isoformat()}, got horizon {point.horizon} at "
                    f"{point.target_period.isoformat()}"
                )
        return self


class ModelMetadata(_Domain):
    """What a promoted model was trained on and how it is versioned (by git SHA)."""

    model_name: str = Field(min_length=1)
    version: str = Field(pattern=r"^[0-9a-f]{7,40}$", description="git SHA of the training code")
    region: Region = Region.VIC1
    trained_at: AwareDatetime
    train_start: AwareDatetime
    train_end: AwareDatetime
    feature_version: str = Field(min_length=1)
    features: list[str] = Field(min_length=1)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    metrics: dict[str, float] = Field(default_factory=dict)
    hf_repo_id: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.train_start >= self.train_end:
            raise ValueError("train_start must be before train_end")
        if len(set(self.features)) != len(self.features):
            raise ValueError("features contains duplicates")
        if any(not 0 < q < 1 for q in self.quantiles) or list(self.quantiles) != sorted(
            set(self.quantiles)
        ):
            raise ValueError("quantiles must be strictly increasing and inside (0, 1)")
        return self
