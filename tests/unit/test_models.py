from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from nem_forecast.backtest.baselines import SeasonalNaive
from nem_forecast.features.build import LABEL, build_features
from nem_forecast.features.series import PERIOD, span
from nem_forecast.models.base import (
    METADATA_FILE,
    TRAIN_ORIGIN_EVERY,
    WARMUP,
    DirectModel,
    training_frame,
)
from nem_forecast.models.catalog import MODELS, load_model
from nem_forecast.models.lgbm import LightGBMDirect, LightGBMQuantile
from nem_forecast.models.ridge import RidgeForecaster
from tests.synthetic import START, series

FAST = {"num_boost_round": 60, "num_leaves": 15, "min_data_in_leaf": 20}
NOISY = series(days=90, noise_sd=100.0)
CUTOFF = START + 75 * 48 * PERIOD
ORIGINS = [CUTOFF + k * PERIOD for k in range(0, 48 * 10, 3)]


def make(name: str) -> DirectModel:
    cls = MODELS[name]
    return cls(FAST) if issubclass(cls, LightGBMDirect | LightGBMQuantile) else cls()


def scored(model: DirectModel | SeasonalNaive) -> pl.DataFrame:
    model.fit(NOISY.filter(pl.col("period_end") <= CUTOFF))
    frame = build_features(NOISY, ORIGINS)
    return frame.select("origin", "target", "horizon", LABEL).join(
        model.predict(frame.drop(LABEL)), on=["origin", "target", "horizon"]
    )


def mae(frame: pl.DataFrame) -> float:
    return float(frame.select((pl.col(LABEL) - pl.col("p50")).abs().mean()).item())


def test_training_frame_starts_after_warmup_and_stays_inside_history() -> None:
    history = NOISY.filter(pl.col("period_end") <= CUTOFF)

    frame = training_frame(history)

    assert frame.select(pl.col("origin").min()).item() == span(history)[0] + WARMUP
    assert (frame["target"] <= CUTOFF).all()
    assert frame[LABEL].null_count() == 0
    gaps = frame["origin"].unique().sort().diff().drop_nulls().unique().to_list()
    assert gaps == [TRAIN_ORIGIN_EVERY]


def test_training_frame_needs_more_than_the_warmup() -> None:
    with pytest.raises(ValueError, match="need more than 28 days"):
        training_frame(series(days=20))


@pytest.mark.parametrize("name", list(MODELS))
def test_models_learn_the_daily_shape_better_than_seasonal_naive(name: str) -> None:
    """With noise sd 100, yesterday's value carries the noise twice (MAE ~113); a model that
    learns the shape should get near the noise floor (MAE ~80)."""
    model_rows = scored(make(name))
    naive_rows = scored(SeasonalNaive())

    assert model_rows.height == len(ORIGINS) * 48
    assert mae(model_rows) < mae(naive_rows)
    assert (model_rows["p10"] <= model_rows["p50"]).all()
    assert (model_rows["p50"] <= model_rows["p90"]).all()
    coverage = model_rows.select(pl.col(LABEL).is_between(pl.col("p10"), pl.col("p90")).mean())
    assert 0.55 < coverage.item() < 0.95


@pytest.mark.parametrize("name", list(MODELS))
def test_save_and_load_reproduce_predictions(name: str, tmp_path: Path) -> None:
    model = make(name)
    model.fit(NOISY.filter(pl.col("period_end") <= CUTOFF))
    features = build_features(NOISY, ORIGINS[:20], with_label=False)

    model.save(tmp_path / name, version="0123abcd")
    restored = load_model(tmp_path / name)

    assert type(restored) is type(model)
    assert restored.predict(features).equals(model.predict(features))
    assert restored.trained_on == model.trained_on
    metadata = json.loads((tmp_path / name / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["version"] == "0123abcd"
    assert not list((tmp_path / name).glob("*.pkl"))


def test_loading_checks_the_model_type_and_features(tmp_path: Path) -> None:
    model = RidgeForecaster()
    model.fit(NOISY.filter(pl.col("period_end") <= CUTOFF))
    model.save(tmp_path, version="0123abcd")

    with pytest.raises(ValueError, match="holds ridge"):
        LightGBMDirect.load(tmp_path)

    metadata = json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))
    metadata["features"] = ["something_else"]
    (tmp_path / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="other features"):
        RidgeForecaster.load(tmp_path)

    metadata["name"] = "mystery"
    (tmp_path / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown model 'mystery'"):
        load_model(tmp_path)


def test_horizons_without_a_model_are_refused() -> None:
    model = RidgeForecaster(horizon_groups=((1,), (2, 3)))
    model.fit(NOISY.filter(pl.col("period_end") <= CUTOFF))

    with pytest.raises(ValueError, match=r"no model for horizons \[4\]"):
        model.predict(build_features(NOISY, ORIGINS[:2], horizons=(1, 4), with_label=False))


def test_quantile_model_returns_ordered_quantiles_even_when_boosters_cross() -> None:
    model = LightGBMQuantile(FAST)
    model.fit(NOISY.filter(pl.col("period_end") <= CUTOFF))
    # Swap the p10 and p90 boosters: raw outputs now cross everywhere.
    for group in range(len(model.horizon_groups)):
        low, high = f"group_{group}_q10", f"group_{group}_q90"
        model._boosters[low], model._boosters[high] = model._boosters[high], model._boosters[low]

    out = model.predict(build_features(NOISY, ORIGINS[:10], with_label=False))

    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()


def test_point_model_intervals_come_from_held_out_errors() -> None:
    """A boosted model that memorises its training data would get near-zero intervals from
    in-sample errors; held-out errors give realistic widths."""
    overfit = {
        "num_boost_round": 150,
        "num_leaves": 127,
        "min_data_in_leaf": 2,
        "learning_rate": 0.3,
    }
    model = LightGBMDirect(FAST | overfit, horizon_groups=((1,), (2, 3)))
    model.fit(NOISY.filter(pl.col("period_end") <= CUTOFF))

    out = model.predict(build_features(NOISY, ORIGINS, horizons=(1, 2, 3), with_label=False))

    width = out.select((pl.col("p90") - pl.col("p10")).mean()).item()
    assert width > 150  # the 10-90 spread of N(0, 100) noise alone is 256


def test_trained_on_records_the_history_range() -> None:
    model = RidgeForecaster()
    history = NOISY.filter(pl.col("period_end") <= CUTOFF)

    model.fit(history)

    assert model.trained_on == (history["period_end"].min(), CUTOFF)
    assert isinstance(model.trained_on[0], datetime)
