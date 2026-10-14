"""Runtime configuration, read from environment variables (and a local ``.env`` in development).

Secrets are held as ``SecretStr`` so they never appear in logs, reprs or tracebacks.

Each component checks for the credentials it uses, where it uses them, rather than demanding
every secret up front: the public forecast API should not hold MLflow or Hugging Face write
tokens it never needs.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PSYCOPG_SCHEME = "postgresql+psycopg://"
_PLAIN_POSTGRES_SCHEMES = ("postgres://", "postgresql://")


class Environment(StrEnum):
    LOCAL = "local"
    CI = "ci"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    environment: Environment = Environment.LOCAL
    database_url: SecretStr

    # Experiment tracking (training jobs)
    mlflow_tracking_uri: str | None = None
    mlflow_tracking_username: str | None = None
    mlflow_tracking_password: SecretStr | None = None

    # Model registry: a write token to promote; the serving side reads public models without one
    hf_token: SecretStr | None = None

    # Serving
    model_repo_id: str | None = None  # e.g. "Azharkhan79/nem-forecast-vic1"
    model_revision: str = "main"  # pin to a git-SHA tag in production
    model_dir: Path | None = None  # a local saved model instead of the Hub (development, tests)
    model_cache_dir: Path = Path(".cache/models")
    git_sha: str | None = None  # commit of the running service, set at image build

    @field_validator("database_url", mode="before")
    @classmethod
    def _use_psycopg_driver(cls, value: object) -> object:
        """Neon hands out ``postgresql://`` URLs; SQLAlchemy needs the psycopg 3 driver named."""
        if isinstance(value, str):
            for scheme in _PLAIN_POSTGRES_SCHEMES:
                if value.startswith(scheme):
                    return _PSYCOPG_SCHEME + value.removeprefix(scheme)
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, loaded once. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
