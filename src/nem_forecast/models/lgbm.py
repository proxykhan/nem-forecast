"""LightGBM direct multi-horizon models: a squared-error point model and a quantile model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import lightgbm as lgb
import numpy as np
import polars as pl

from nem_forecast.features.build import FEATURES, LABEL
from nem_forecast.models.base import HORIZON_GROUPS, DirectModel, feature_matrix

DEFAULT_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_boost_round": 400,
}
# Half the logical cores, capped at 8: on Windows, OpenMP across every hyperthread is 2-3x
# slower than this for the small per-group datasets (measured: 16 s vs 6 s per full fit).
_THREADS = max(1, min(8, (os.cpu_count() or 2) // 2))
_FIXED = {"verbosity": -1, "seed": 42, "force_row_wise": True, "num_threads": _THREADS}
QUANTILES = (0.1, 0.5, 0.9)


def train_booster(
    rows: pl.DataFrame, params: dict[str, Any], objective: str, alpha: float | None = None
) -> lgb.Booster:
    settings = {k: v for k, v in params.items() if k != "num_boost_round"}
    settings |= _FIXED | {"objective": objective}
    if alpha is not None:
        settings["alpha"] = alpha
    data = lgb.Dataset(
        feature_matrix(rows), label=rows[LABEL].to_numpy(), feature_name=list(FEATURES)
    )
    return lgb.train(settings, data, num_boost_round=int(params["num_boost_round"]))


class _LightGBMModel(DirectModel):
    def __init__(
        self,
        params: dict[str, Any] | None = None,
        horizon_groups: tuple[tuple[int, ...], ...] = HORIZON_GROUPS,
    ) -> None:
        super().__init__(horizon_groups)
        self.params = DEFAULT_PARAMS | (params or {})
        self._boosters: dict[str, lgb.Booster] = {}

    def _state(self) -> dict[str, Any]:
        return {"params": self.params, "boosters": sorted(self._boosters)}

    def _save_files(self, directory: Path) -> None:
        for key, booster in self._boosters.items():
            booster.save_model(str(directory / f"{key}.txt"))

    @classmethod
    def _restore(cls, state: dict[str, Any], directory: Path) -> DirectModel:
        model = cls(params=state["params"])
        model._boosters = {
            key: lgb.Booster(model_file=str(directory / f"{key}.txt")) for key in state["boosters"]
        }
        return model


class LightGBMDirect(_LightGBMModel):
    name = "lightgbm_direct"
    description = "LightGBM, squared error, one model per horizon group"

    def _fit_group(self, group: int, rows: pl.DataFrame) -> None:
        self._boosters[f"group_{group}"] = train_booster(rows, self.params, "regression")

    def _predict_group(self, group: int, rows: pl.DataFrame) -> np.ndarray:
        return np.asarray(self._boosters[f"group_{group}"].predict(feature_matrix(rows)), float)


class LightGBMQuantile(_LightGBMModel):
    name = "lightgbm_quantile"
    description = "LightGBM quantile regression at 0.1/0.5/0.9, one model per horizon group"
    native_intervals: ClassVar[bool] = True

    def _fit_group(self, group: int, rows: pl.DataFrame) -> None:
        for q in QUANTILES:
            self._boosters[_key(group, q)] = train_booster(rows, self.params, "quantile", alpha=q)

    def _predict_group(self, group: int, rows: pl.DataFrame) -> np.ndarray:
        x = feature_matrix(rows)
        stacked = np.column_stack(
            [np.asarray(self._boosters[_key(group, q)].predict(x), float) for q in QUANTILES]
        )
        # Separately trained quantile models can cross; sorting restores p10 <= p50 <= p90.
        return np.sort(stacked, axis=1)


def _key(group: int, q: float) -> str:
    return f"group_{group}_q{round(q * 100):02d}"
