from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Keep the developer's environment, .env file and cached settings out of every test.

    Running from an empty directory also means a real .env (with a production DATABASE_URL)
    can never be picked up, and relative paths such as the download cache land in tmp_path.
    """
    monkeypatch.chdir(tmp_path)
    for name in _CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
