"""Every trainable model by name, and loading a saved one without knowing its class."""

from __future__ import annotations

import json
from pathlib import Path

from nem_forecast.models.base import METADATA_FILE, DirectModel
from nem_forecast.models.lgbm import LightGBMDirect, LightGBMQuantile
from nem_forecast.models.ridge import RidgeForecaster

MODELS: dict[str, type[DirectModel]] = {
    cls.name: cls for cls in (RidgeForecaster, LightGBMDirect, LightGBMQuantile)
}


def load_model(directory: Path) -> DirectModel:
    name = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))["name"]
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r} in {directory}")
    return MODELS[name].load(directory)
