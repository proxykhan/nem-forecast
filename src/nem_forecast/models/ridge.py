"""Ridge regression: the linear reference point for the tree models.

Numeric features are median-imputed and standardised; calendar features are one-hot encoded
(a linear model cannot use "half-hour 37" as a number). One model per horizon group, like the
LightGBM models, so the comparison isolates the learner rather than the setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nem_forecast.features.build import LABEL
from nem_forecast.models.base import HORIZON_GROUPS, DirectModel

NUMERIC = (
    "horizon",
    "demand_at_origin",
    "same_slot_prev_day",
    "same_slot_prev_week",
    "same_slot_mean_4w",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_mean_7d",
)
ONE_HOT = {"half_hour": range(48), "day_of_week": range(7), "month": range(1, 13)}
FLAGS = ("is_public_holiday",)
ALPHA = 1.0


def _design(rows: pl.DataFrame, scaling: dict[str, list[float]]) -> np.ndarray:
    numeric = rows.select(pl.col(c).cast(pl.Float64) for c in NUMERIC).to_numpy()
    numeric = np.where(np.isnan(numeric), np.asarray(scaling["medians"]), numeric)
    numeric = (numeric - np.asarray(scaling["means"])) / np.asarray(scaling["stds"])
    blocks = [numeric]
    for column, categories in ONE_HOT.items():
        values = rows[column].to_numpy()
        blocks.append((values[:, None] == np.asarray(categories)[None, :]).astype(float))
    blocks.append(rows.select(pl.col(c).cast(pl.Float64) for c in FLAGS).to_numpy())
    return np.hstack(blocks)


class RidgeForecaster(DirectModel):
    name = "ridge"
    description = "Ridge regression on standardised lags and one-hot calendar, per horizon group"

    def __init__(self, horizon_groups: tuple[tuple[int, ...], ...] = HORIZON_GROUPS) -> None:
        super().__init__(horizon_groups)
        self._groups: dict[int, dict[str, Any]] = {}

    def _fit_group(self, group: int, rows: pl.DataFrame) -> None:
        numeric = rows.select(pl.col(c).cast(pl.Float64) for c in NUMERIC).to_numpy()
        medians = np.nanmedian(numeric, axis=0)
        filled = np.where(np.isnan(numeric), medians, numeric)
        stds = filled.std(axis=0)
        scaling = {
            "medians": medians.tolist(),
            "means": filled.mean(axis=0).tolist(),
            "stds": np.where(stds > 0, stds, 1.0).tolist(),
        }
        from sklearn.linear_model import Ridge  # training only; prediction is plain numpy

        fitted = Ridge(alpha=ALPHA).fit(_design(rows, scaling), rows[LABEL].to_numpy())
        self._groups[group] = scaling | {
            "coef": fitted.coef_.tolist(),
            "intercept": float(fitted.intercept_),
        }

    def _predict_group(self, group: int, rows: pl.DataFrame) -> np.ndarray:
        state = self._groups[group]
        coef = np.asarray(state["coef"])
        return np.asarray(_design(rows, state) @ coef + state["intercept"], float)

    def _state(self) -> dict[str, Any]:
        return {"alpha": ALPHA, "groups": {str(k): v for k, v in self._groups.items()}}

    @classmethod
    def _restore(cls, state: dict[str, Any], directory: Path) -> DirectModel:
        model = cls()
        model._groups = {int(k): v for k, v in state["groups"].items()}
        return model
