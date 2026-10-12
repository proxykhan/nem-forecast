from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from nem_forecast.schemas import (
    FORECAST_PERIOD,
    NEM_TZ,
    DispatchInterval,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    ModelMetadata,
    Region,
)

ORIGIN = datetime(2026, 10, 12, 14, 0, tzinfo=NEM_TZ)


def interval(**overrides: Any) -> DispatchInterval:
    fields: dict[str, Any] = {
        "settlement_date": datetime(2026, 10, 12, 14, 5, tzinfo=NEM_TZ),
        "region": "VIC1",
        "demand_mw": 4875.31,
        "rrp": -12.5,
        "source_file": "PUBLIC_DISPATCHIS_202610121405_0000000400000001.zip",
    }
    return DispatchInterval(**(fields | overrides))


def point(horizon: int, **overrides: Any) -> ForecastPoint:
    fields: dict[str, Any] = {
        "target_period": ORIGIN + horizon * FORECAST_PERIOD,
        "horizon": horizon,
        "predicted_mw": 5000.0,
        "lower_mw": 4700.0,
        "upper_mw": 5300.0,
    }
    return ForecastPoint(**(fields | overrides))


class TestNemTimezone:
    def test_is_fixed_utc_plus_10_all_year(self) -> None:
        # Victoria moves to UTC+11 in summer; AEMO market time never does.
        for month in (1, 4, 7, 10):
            assert datetime(2026, month, 15, tzinfo=NEM_TZ).utcoffset() == timedelta(hours=10)


class TestDispatchInterval:
    def test_valid_record(self) -> None:
        record = interval()

        assert record.region is Region.VIC1
        assert record.rrp == -12.5
        assert record.ingested_at.tzinfo is not None

    def test_rrp_is_optional(self) -> None:
        assert interval(rrp=None).rrp is None

    def test_naive_settlement_date_rejected(self) -> None:
        with pytest.raises(ValidationError, match="timezone"):
            interval(settlement_date=datetime(2026, 10, 12, 14, 5))  # noqa: DTZ001

    def test_off_boundary_settlement_date_rejected(self) -> None:
        with pytest.raises(ValidationError, match="5-minute boundary"):
            interval(settlement_date=datetime(2026, 10, 12, 14, 7, tzinfo=NEM_TZ))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_demand_rejected(self, bad: float) -> None:
        with pytest.raises(ValidationError, match="demand_mw"):
            interval(demand_mw=bad)

    def test_unknown_region_rejected(self) -> None:
        with pytest.raises(ValidationError, match="region"):
            interval(region="WA1")  # WA is not in the NEM

    def test_empty_source_file_rejected(self) -> None:
        with pytest.raises(ValidationError, match="source_file"):
            interval(source_file="")

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            interval(totaldemand=1.0)

    def test_is_immutable(self) -> None:
        record = interval()
        with pytest.raises(ValidationError, match="frozen"):
            record.demand_mw = 1.0  # type: ignore[misc]

    def test_same_instant_in_utc_is_equal(self) -> None:
        in_utc = datetime(2026, 10, 12, 4, 5, tzinfo=UTC)
        assert interval(settlement_date=in_utc).settlement_date == interval().settlement_date


class TestForecastRequest:
    def test_defaults_to_vic1_and_48_periods(self) -> None:
        request = ForecastRequest(forecast_origin=ORIGIN)

        assert request.region is Region.VIC1
        assert request.horizon == 48

    @pytest.mark.parametrize("horizon", [0, 97, -1])
    def test_horizon_out_of_range_rejected(self, horizon: int) -> None:
        with pytest.raises(ValidationError, match="horizon"):
            ForecastRequest(forecast_origin=ORIGIN, horizon=horizon)

    def test_origin_off_half_hour_rejected(self) -> None:
        with pytest.raises(ValidationError, match="30-minute boundary"):
            ForecastRequest(forecast_origin=ORIGIN + timedelta(minutes=5))

    def test_naive_origin_rejected(self) -> None:
        with pytest.raises(ValidationError, match="timezone"):
            ForecastRequest(forecast_origin=datetime(2026, 10, 12, 14, 0))  # noqa: DTZ001


class TestForecastPoint:
    def test_valid_point(self) -> None:
        assert point(1).predicted_mw == 5000.0

    @pytest.mark.parametrize(
        ("lower", "predicted", "upper"),
        [(5100.0, 5000.0, 5300.0), (4700.0, 5000.0, 4900.0), (5000.0, 4000.0, 3000.0)],
    )
    def test_crossed_quantiles_rejected(self, lower: float, predicted: float, upper: float) -> None:
        with pytest.raises(ValidationError, match="lower <= predicted <= upper"):
            point(1, lower_mw=lower, predicted_mw=predicted, upper_mw=upper)

    def test_degenerate_interval_allowed(self) -> None:
        assert point(1, lower_mw=5000.0, upper_mw=5000.0).lower_mw == 5000.0


class TestForecastResponse:
    def test_valid_48_period_response(self) -> None:
        response = ForecastResponse(
            region=Region.VIC1,
            forecast_origin=ORIGIN,
            model_version="a1b2c3d",
            points=[point(h) for h in range(1, 49)],
        )

        assert len(response.points) == 48
        assert response.points[-1].target_period == ORIGIN + timedelta(hours=24)
        assert (response.lower_quantile, response.upper_quantile) == (0.1, 0.9)

    def test_empty_points_rejected(self) -> None:
        with pytest.raises(ValidationError, match="points"):
            ForecastResponse(
                region=Region.VIC1, forecast_origin=ORIGIN, model_version="a1b2c3d", points=[]
            )

    def test_gap_in_horizons_rejected(self) -> None:
        with pytest.raises(ValidationError, match="point 2 should be horizon 2"):
            ForecastResponse(
                region=Region.VIC1,
                forecast_origin=ORIGIN,
                model_version="a1b2c3d",
                points=[point(1), point(3)],
            )

    def test_target_period_must_match_origin(self) -> None:
        shifted = point(1, target_period=ORIGIN + timedelta(hours=1))
        with pytest.raises(ValidationError, match="point 1 should be horizon 1"):
            ForecastResponse(
                region=Region.VIC1,
                forecast_origin=ORIGIN,
                model_version="a1b2c3d",
                points=[shifted],
            )

    def test_quantile_levels_must_straddle_median(self) -> None:
        with pytest.raises(ValidationError, match="lower_quantile"):
            ForecastResponse(
                region=Region.VIC1,
                forecast_origin=ORIGIN,
                model_version="a1b2c3d",
                lower_quantile=0.6,
                points=[point(1)],
            )


class TestModelMetadata:
    @staticmethod
    def metadata(**overrides: Any) -> ModelMetadata:
        fields: dict[str, Any] = {
            "model_name": "lightgbm_quantile",
            "version": "0123abcd",
            "trained_at": datetime(2026, 10, 12, tzinfo=UTC),
            "train_start": datetime(2023, 10, 1, tzinfo=NEM_TZ),
            "train_end": datetime(2026, 9, 30, tzinfo=NEM_TZ),
            "feature_version": "v1",
            "features": ["half_hour", "lag_48", "lag_336"],
            "metrics": {"mae": 101.2},
        }
        return ModelMetadata(**(fields | overrides))

    def test_valid_metadata(self) -> None:
        meta = self.metadata()

        assert meta.region is Region.VIC1
        assert meta.quantiles == (0.1, 0.5, 0.9)

    @pytest.mark.parametrize("version", ["latest", "ABCDEF1", "abc12", "g123456"])
    def test_version_must_be_a_git_sha(self, version: str) -> None:
        with pytest.raises(ValidationError, match="version"):
            self.metadata(version=version)

    def test_train_range_must_be_ordered(self) -> None:
        with pytest.raises(ValidationError, match="train_start must be before train_end"):
            self.metadata(train_start=datetime(2026, 10, 1, tzinfo=NEM_TZ))

    def test_duplicate_features_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicates"):
            self.metadata(features=["lag_48", "lag_48"])

    @pytest.mark.parametrize(
        "quantiles", [(0.5, 0.1, 0.9), (0.1, 0.1, 0.9), (0.0, 0.5), (0.5, 1.0)]
    )
    def test_bad_quantiles_rejected(self, quantiles: tuple[float, ...]) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            self.metadata(quantiles=quantiles)
