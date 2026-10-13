"""Direct multi-horizon models: one fitted sub-model per horizon group, never recursive.

A recursive model forecasts t+1, feeds that guess back in as if it were data, and compounds its
own errors out to t+48. A direct model learns demand at the target straight from features known
at the origin. Horizons are grouped so that 30-minutes-ahead (where the last observation is
nearly everything) and a day ahead (where it is nearly nothing) get separate models, while
neighbouring horizons share one, with ``horizon`` itself as a feature.

Design choices, fixed rather than tuned:

* Horizon groups 1 | 2-3 | 4-6 | 7-12 | 13-24 | 25-48, roughly doubling in width.
* Training origins every 2 hours. Consecutive half-hourly origins are near-duplicate examples;
  thinning them cuts training time 4x for little information loss. Evaluation still forecasts
  from every half-hour.
* 28 days of warm-up history before the first training origin, so every feature is defined.
* Point models estimate their intervals from errors on the last 28 days before the cutoff,
  predicted by a copy trained without those days, then refit on everything. In-sample errors of
  a boosted model would make its intervals far too narrow.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl

from nem_forecast.backtest.baselines import PREDICTIONS
from nem_forecast.backtest.intervals import ResidualIntervals
from nem_forecast.features.build import (
    DEFAULT_HORIZONS,
    FEATURE_VERSION,
    FEATURES,
    KEYS,
    LABEL,
    assert_no_leakage,
    build_features,
)
from nem_forecast.features.series import PERIOD, TIME_ZONE, span

HORIZON_GROUPS: tuple[tuple[int, ...], ...] = (
    (1,),
    (2, 3),
    (4, 5, 6),
    tuple(range(7, 13)),
    tuple(range(13, 25)),
    tuple(range(25, 49)),
)
TRAIN_ORIGIN_EVERY = timedelta(hours=2)
WARMUP = timedelta(days=28)
CALIBRATION = timedelta(days=28)
METADATA_FILE = "model.json"


def training_frame(
    history: pl.DataFrame,
    *,
    origin_every: timedelta = TRAIN_ORIGIN_EVERY,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pl.DataFrame:
    """Leakage-checked (features, label) rows whose origins and targets all lie inside history."""
    first, last = span(history)
    start = first + WARMUP
    if start >= last:
        raise ValueError(f"need more than {WARMUP.days} days of history, got {last - first}")
    origins = pl.datetime_range(
        start, last - PERIOD, interval=origin_every, time_zone=TIME_ZONE, eager=True
    )
    frame = build_features(history, origins, horizons)
    assert_no_leakage(frame)
    return frame.filter(pl.col(LABEL).is_not_null())


def feature_matrix(frame: pl.DataFrame) -> np.ndarray:
    """Model inputs in FEATURES order as float64; missing values become NaN."""
    return frame.select(pl.col(name).cast(pl.Float64) for name in FEATURES).to_numpy()


class DirectModel(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    kind: ClassVar[str] = "model"
    native_intervals: ClassVar[bool] = False

    def __init__(self, horizon_groups: tuple[tuple[int, ...], ...] = HORIZON_GROUPS) -> None:
        self.horizon_groups = horizon_groups
        self._intervals = ResidualIntervals()
        self.trained_on: tuple[datetime, datetime] | None = None

    # -- subclasses -------------------------------------------------------------------------

    @abstractmethod
    def _fit_group(self, group: int, rows: pl.DataFrame) -> None: ...

    @abstractmethod
    def _predict_group(self, group: int, rows: pl.DataFrame) -> np.ndarray:
        """Shape (n,) for point models, (n, 3) quantiles p10/p50/p90 for native-interval models."""

    @abstractmethod
    def _state(self) -> dict[str, Any]:
        """JSON-serialisable parameters and fitted state, apart from files saved separately."""

    def _save_files(self, directory: Path) -> None:  # noqa: B027 - optional hook
        """Write any fitted state that does not belong in model.json."""

    @classmethod
    @abstractmethod
    def _restore(cls, state: dict[str, Any], directory: Path) -> DirectModel: ...

    # -- Forecaster protocol ----------------------------------------------------------------

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(h for group in self.horizon_groups for h in group)

    def fit(self, history: pl.DataFrame) -> None:
        frame = training_frame(history, horizons=self.horizons)
        if not self.native_intervals:
            first, last = span(history)
            if last - first <= WARMUP + CALIBRATION:
                raise ValueError(
                    f"{self.name}: needs more than {(WARMUP + CALIBRATION).days} days of history "
                    "(warm-up plus the held-out calibration window)"
                )
            calibration_start = last - CALIBRATION
            self.fit_frame(frame.filter(pl.col("target") <= calibration_start))
            held_out = frame.filter(pl.col("origin") >= calibration_start)
            self._intervals.fit(
                held_out.join(self.raw_predictions(held_out), on=list(KEYS)), point="_point"
            )
        self.fit_frame(frame)
        self.trained_on = span(history)

    def fit_frame(self, frame: pl.DataFrame) -> None:
        """Fit every horizon group's sub-model on prepared (features, label) rows."""
        for group, horizons in enumerate(self.horizon_groups):
            rows = frame.filter(pl.col("horizon").is_in(horizons))
            if rows.is_empty():
                raise ValueError(f"{self.name}: no training rows for horizons {horizons}")
            self._fit_group(group, rows)

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        raw = self.raw_predictions(features)
        if not self.native_intervals:
            raw = self._intervals.apply(raw, point="_point")
        return (
            features.select(KEYS).join(raw, on=list(KEYS), how="left").select(*KEYS, *PREDICTIONS)
        )

    def raw_predictions(self, features: pl.DataFrame) -> pl.DataFrame:
        """Keys plus `_point` (point models) or p10/p50/p90 (quantile models), no intervals."""
        unsupported = set(features["horizon"].unique()) - set(self.horizons)
        if unsupported:
            raise ValueError(f"{self.name}: no model for horizons {sorted(unsupported)}")
        parts = []
        for group, horizons in enumerate(self.horizon_groups):
            rows = features.filter(pl.col("horizon").is_in(horizons))
            if rows.is_empty():
                continue
            out = self._predict_group(group, rows)
            keys = rows.select(KEYS)
            if self.native_intervals:
                parts.append(
                    keys.with_columns(
                        p10=pl.Series(out[:, 0]), p50=pl.Series(out[:, 1]), p90=pl.Series(out[:, 2])
                    )
                )
            else:
                parts.append(keys.with_columns(_point=pl.Series(out)))
        return pl.concat(parts)

    # -- persistence ------------------------------------------------------------------------

    def save(self, directory: Path, *, version: str) -> Path:
        """Write the fitted model as JSON plus plain-text files: no pickle, safe to load."""
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": self.name,
            "description": self.description,
            "version": version,
            "feature_version": FEATURE_VERSION,
            "features": list(FEATURES),
            "horizon_groups": [list(g) for g in self.horizon_groups],
            "quantiles": [0.1, 0.5, 0.9],
            "trained_on": [ts.isoformat() for ts in self.trained_on] if self.trained_on else None,
            "saved_at": datetime.now(UTC).isoformat(),
            "intervals": None if self.native_intervals else self._intervals.to_dict(),
            "state": self._state(),
        }
        self._save_files(directory)
        path = directory / METADATA_FILE
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path) -> DirectModel:
        metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        if metadata["name"] != cls.name:
            raise ValueError(f"{directory} holds {metadata['name']}, not {cls.name}")
        if metadata["features"] != list(FEATURES):
            raise ValueError(f"{directory} was trained on other features: {metadata['features']}")
        model = cls._restore(metadata["state"], directory)
        model.horizon_groups = tuple(tuple(g) for g in metadata["horizon_groups"])
        if metadata["intervals"] is not None:
            model._intervals = ResidualIntervals.from_dict(metadata["intervals"])
        if metadata["trained_on"]:
            first, last = metadata["trained_on"]
            model.trained_on = (datetime.fromisoformat(first), datetime.fromisoformat(last))
        return model
