"""The forecasting model: downloaded once at startup, cached on disk, never fetched per request.

Loading runs in a background thread so the process answers liveness checks as soon as it can;
until it finishes the service reports itself not ready. A forecast request that arrives while the
model is loading waits for it (up to ``MODEL_WAIT_SECONDS``): on a free-tier cold start the
request that woke the service should get a forecast, not an error.

A download from Hugging Face Hub lands in ``MODEL_CACHE_DIR``; a restarted process with a warm
cache skips the transfer.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl
from huggingface_hub import snapshot_download

from nem_forecast.config import Settings
from nem_forecast.models.base import METADATA_FILE, DirectModel
from nem_forecast.models.catalog import load_model
from nem_forecast.schemas import Region
from nem_forecast.serve.problems import Problem

log = logging.getLogger(__name__)

Downloader = Callable[..., str]


class ModelState(StrEnum):
    NOT_CONFIGURED = "not_configured"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class LoadedModel:
    model: DirectModel
    name: str
    version: str  # the tag the model was published under: a release tag or a 12-character SHA
    region: Region
    max_horizon: int
    trained_to: datetime | None
    source: str


class ModelStore:
    def __init__(
        self,
        settings: Settings,
        *,
        download: Downloader = snapshot_download,
    ) -> None:
        self._settings = settings
        self._download = download
        self._listeners: list[Callable[[LoadedModel], None]] = []
        self._finished = threading.Event()
        self._predict_lock = threading.Lock()
        self.loaded: LoadedModel | None = None
        self.error: str | None = None
        configured = settings.model_dir is not None or settings.model_repo_id is not None
        self.state = ModelState.LOADING if configured else ModelState.NOT_CONFIGURED
        if not configured:
            self._finished.set()

    def on_loaded(self, listener: Callable[[LoadedModel], None]) -> None:
        """Call ``listener`` with the model once loaded (immediately if it already is)."""
        self._listeners.append(listener)
        if self.loaded is not None:
            listener(self.loaded)

    def load(self) -> None:
        """Fetch (or reuse the cached) model and load it; blocking, run off the event loop."""
        if self.state is not ModelState.LOADING:
            return
        try:
            self.loaded = fetch_model(self._settings, download=self._download)
            self.state = ModelState.READY
            log.info(
                "loaded %s %s from %s", self.loaded.name, self.loaded.version, self.loaded.source
            )
            for listener in self._listeners:
                listener(self.loaded)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = ModelState.FAILED
            log.exception("model failed to load")
        finally:
            self._finished.set()

    def require(self, wait: float = 0.0) -> LoadedModel:
        """The loaded model, waiting up to ``wait`` seconds for one that is still loading."""
        if self.loaded is None and wait > 0:
            self._finished.wait(wait)
        if self.loaded is not None:
            return self.loaded
        if self.state is ModelState.LOADING:
            raise Problem(
                503,
                "model-loading",
                "Model is still loading",
                "The service has just started and is loading its model; retry shortly.",
                headers={"Retry-After": "10"},
            )
        raise Problem(
            503,
            "model-unavailable",
            "Model unavailable",
            "No forecasting model could be loaded; see /ready for the reason.",
        )

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        loaded = self.require()
        with self._predict_lock:  # one LightGBM prediction at a time on the shared boosters
            return loaded.model.predict(features)


def fetch_model(settings: Settings, *, download: Downloader = snapshot_download) -> LoadedModel:
    """Load the configured model: ``MODEL_DIR`` if set, else ``MODEL_REPO_ID`` at its revision.

    Shared by the API and the monitoring and retraining jobs, so all three judge the same model.
    """
    directory, source = _locate(settings, download)
    model = load_model(directory)
    metadata: dict[str, Any] = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    trained = metadata.get("trained_on")
    return LoadedModel(
        model=model,
        name=model.name,
        version=model_version(metadata),
        region=Region(metadata.get("region", Region.VIC1.value)),
        max_horizon=max(model.horizons),
        trained_to=datetime.fromisoformat(trained[1]) if trained else None,
        source=source,
    )


def model_version(metadata: dict[str, Any]) -> str:
    """The version predictions are logged under: the release tag, or the SHA's first 12 chars."""
    return str(metadata.get("release") or str(metadata["version"])[:12])


def _locate(settings: Settings, download: Downloader) -> tuple[Path, str]:
    if settings.model_dir is not None:
        return settings.model_dir, f"local:{settings.model_dir}"
    if settings.model_repo_id is None:
        raise ValueError("no model configured: set MODEL_REPO_ID (or MODEL_DIR)")
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    path = download(
        settings.model_repo_id,
        revision=settings.model_revision,
        cache_dir=settings.model_cache_dir,
    )
    return Path(path), f"hf:{settings.model_repo_id}@{settings.model_revision}"
