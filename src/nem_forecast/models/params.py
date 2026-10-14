"""Where tuned hyperparameters live, readable without the tuning stack (Optuna) installed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nem_forecast.schemas import Region

VALIDATION_DAYS = 60
MAX_TRIALS = 50


def params_path(region: Region, directory: Path = Path("config")) -> Path:
    return directory / f"lightgbm_{region.value.lower()}.json"


def load_params(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    params: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))["params"]
    return params
