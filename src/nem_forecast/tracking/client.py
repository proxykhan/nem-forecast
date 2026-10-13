"""Point MLflow at the configured tracking server.

With MLFLOW_TRACKING_URI set (DagsHub in production), MLflow authenticates with HTTP basic auth,
which it reads from the MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD environment
variables. Without it, runs go to a local SQLite store so development never needs credentials.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow

from nem_forecast.config import Settings

log = logging.getLogger(__name__)

LOCAL_STORE = Path("mlflow.db")


def configure_mlflow(settings: Settings) -> str:
    """Set the tracking URI (and credentials) for this process; returns the URI in use."""
    if settings.mlflow_tracking_uri:
        if settings.mlflow_tracking_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
        if settings.mlflow_tracking_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = (
                settings.mlflow_tracking_password.get_secret_value()
            )
        uri = settings.mlflow_tracking_uri
    else:
        # Absolute, because MLflow caches stores by URI and a relative path would stay bound to
        # whichever directory first opened it.
        uri = f"sqlite:///{LOCAL_STORE.resolve().as_posix()}"
        log.warning("MLFLOW_TRACKING_URI is not set; tracking to %s", uri)
    # One "View run at ..." line per run floods the CLI during a backtest; runs are linked from
    # the report and model_runs instead.
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
    mlflow.set_tracking_uri(uri)
    return uri


def experiment_id(name: str) -> str:
    """Id of the named experiment, created on first use."""
    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        return str(existing.experiment_id)
    return str(mlflow.create_experiment(name))
