from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nem_forecast.features.build import (
    ASOF,
    DATA_FEATURES,
    LABEL,
    LeakageError,
    assert_no_leakage,
    build_features,
    one_step_features,
)
from nem_forecast.features.series import PERIOD
from tests.synthetic import START, index_series

SERIES = index_series(days=40)
ORIGIN_INDEX = 35 * 48  # an origin with more than four weeks of history behind it
ORIGIN = START + (ORIGIN_INDEX + 1) * PERIOD


def row(frame: pl.DataFrame, horizon: int) -> dict[str, object]:
    return frame.filter(pl.col("horizon") == horizon).row(0, named=True)


class TestValues:
    """On a series whose demand equals the period index, every feature has a closed form."""

    frame = build_features(SERIES, [ORIGIN], horizons=(1, 2, 48))

    def test_demand_at_origin_is_the_last_known_value(self) -> None:
        for h in (1, 2, 48):
            assert row(self.frame, h)["demand_at_origin"] == ORIGIN_INDEX

    def test_same_slot_lags_are_relative_to_the_target(self) -> None:
        for h in (1, 2, 48):
            target = ORIGIN_INDEX + h
            values = row(self.frame, h)
            assert values[LABEL] == target
            assert values["same_slot_prev_day"] == target - 48
            assert values["same_slot_prev_week"] == target - 336
            assert values["same_slot_mean_4w"] == target - (336 + 672 + 1008 + 1344) / 4

    def test_rolling_windows_end_at_the_origin(self) -> None:
        values = row(self.frame, 2)
        assert values["rolling_mean_24h"] == ORIGIN_INDEX - 23.5
        assert values["rolling_mean_7d"] == ORIGIN_INDEX - 167.5
        # Sample std of 48 consecutive integers: sqrt(48 * 49 / 12) = 14.
        assert values["rolling_std_24h"] == pytest.approx(14.0)

    def test_beyond_a_day_ahead_the_same_slot_comes_from_two_days_back(self) -> None:
        values = row(build_features(SERIES, [ORIGIN], horizons=(49,)), 49)

        assert values["same_slot_prev_day"] == ORIGIN_INDEX + 49 - 96

    def test_four_week_mean_needs_all_four_weeks(self) -> None:
        early = START + 25 * 48 * PERIOD  # only three weeks of history

        assert row(build_features(SERIES, [early], horizons=(1,)), 1)["same_slot_mean_4w"] is None

    def test_rolling_windows_need_full_history(self) -> None:
        values = row(build_features(SERIES, [START + 10 * PERIOD], horizons=(1,)), 1)

        assert values["rolling_mean_24h"] is None
        assert values["rolling_mean_7d"] is None

    @pytest.mark.parametrize("horizons", [(0,), (1, 337), ()])
    def test_horizons_are_validated(self, horizons: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="horizons"):
            build_features(SERIES, [ORIGIN], horizons=horizons)

    def test_without_label(self) -> None:
        assert LABEL not in build_features(SERIES, [ORIGIN], with_label=False).columns


class TestCalendar:
    @staticmethod
    def calendar(period_start_utc: datetime) -> dict[str, object]:
        origin = period_start_utc  # the target is the period that starts at the origin
        return row(build_features(SERIES, [origin], horizons=(1,)), 1)

    def test_half_hour_follows_melbourne_wall_clock_across_daylight_saving(self) -> None:
        # 5 Oct 2025: Melbourne clocks jump from 02:00 AEST to 03:00 AEDT (16:00 UTC).
        assert self.calendar(datetime(2025, 10, 4, 15, 30, tzinfo=UTC))["half_hour"] == 3  # 01:30
        assert self.calendar(datetime(2025, 10, 4, 16, 0, tzinfo=UTC))["half_hour"] == 6  # 03:00

    def test_day_of_week_and_month(self) -> None:
        values = self.calendar(datetime(2026, 10, 12, 0, tzinfo=UTC))  # Mon 12 Oct 11:00 AEDT

        assert (values["day_of_week"], values["month"], values["half_hour"]) == (0, 10, 22)

    @pytest.mark.parametrize(
        ("local_noon_utc", "holiday"),
        [
            (datetime(2025, 11, 4, 1, tzinfo=UTC), True),  # Melbourne Cup Day
            (datetime(2025, 11, 5, 1, tzinfo=UTC), False),
            (datetime(2025, 9, 26, 2, tzinfo=UTC), True),  # Friday before the AFL Grand Final
            (datetime(2025, 12, 25, 1, tzinfo=UTC), True),
        ],
    )
    def test_victorian_public_holidays(self, local_noon_utc: datetime, holiday: bool) -> None:
        assert self.calendar(local_noon_utc)["is_public_holiday"] is holiday


LEAK_ORIGINS = [ORIGIN + k * PERIOD for k in range(6)]


class TestLeakage:
    def test_builder_output_passes(self) -> None:
        assert_no_leakage(build_features(SERIES, LEAK_ORIGINS))

    def test_a_future_value_wrongly_included_is_caught(self) -> None:
        """The classic multi-horizon bug: lag-1 taken relative to the target, not the origin."""
        frame = build_features(SERIES, LEAK_ORIGINS)
        leaky = frame.with_columns(
            demand_at_origin_asof=pl.col("target") - PERIOD,
            demand_at_origin=pl.col(LABEL) - 1,
        )

        with pytest.raises(LeakageError, match=r"demand_at_origin: \d+ rows use data after"):
            assert_no_leakage(leaky)
        # At horizon 1 the period before the target *is* the origin, so it is legitimately fine.
        assert_no_leakage(leaky.filter(pl.col("horizon") == 1))

    def test_a_feature_without_provenance_is_refused(self) -> None:
        frame = build_features(SERIES, LEAK_ORIGINS).with_columns(
            next_period=pl.col(LABEL).shift(-1)
        )

        with pytest.raises(LeakageError, match=r"no provenance for columns \['next_period'\]"):
            assert_no_leakage(frame)

    def test_one_cutoff_for_every_row(self) -> None:
        frame = build_features(SERIES, LEAK_ORIGINS)

        assert_no_leakage(frame, cutoff=LEAK_ORIGINS[-1])
        with pytest.raises(LeakageError):
            assert_no_leakage(frame, cutoff=LEAK_ORIGINS[0])

    def test_features_do_not_change_when_the_future_does(self) -> None:
        """The strongest check: rewrite everything after the origin and nothing may move."""
        features = [c for c in build_features(SERIES, [ORIGIN]).columns if c != LABEL]
        rewritten = SERIES.with_columns(
            demand=pl.when(pl.col("period_end") > ORIGIN)
            .then(pl.col("demand") * 1000 + 7)
            .otherwise(pl.col("demand"))
        )

        before = build_features(SERIES, [ORIGIN]).select(features)
        after = build_features(rewritten, [ORIGIN]).select(features)

        assert before.equals(after)
        assert not build_features(SERIES, [ORIGIN])[LABEL].equals(
            build_features(rewritten, [ORIGIN])[LABEL]
        )

    def test_every_data_feature_carries_provenance(self) -> None:
        columns = build_features(SERIES, [ORIGIN]).columns

        assert all(f"{name}{ASOF}" in columns for name in DATA_FEATURES)


def test_one_step_features_cover_every_period_at_horizon_one() -> None:
    frame = one_step_features(SERIES)

    assert frame.height == SERIES.height - 1
    assert set(frame["horizon"].unique()) == {1}
    assert (frame["target"] - frame["origin"]).unique().to_list() == [timedelta(minutes=30)]
    assert_no_leakage(frame)
    assert not math.isnan(frame["rolling_mean_7d"].drop_nulls()[0])
