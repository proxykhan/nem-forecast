from __future__ import annotations

from collections.abc import Iterator

import pytest

from nem_forecast.config import get_settings

_CONFIG_ENV_VARS = (
    "ENVIRONMENT",
    "DATABASE_URL",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "HF_TOKEN",
)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the developer's shell environment and cached settings out of every test."""
    for name in _CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
