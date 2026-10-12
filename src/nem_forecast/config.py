"""Runtime configuration, read from environment variables (and a local ``.env`` in development).

Secrets are held as ``SecretStr`` so they never appear in logs, reprs or tracebacks.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Self

from pydantic import SecretStr, field_validator, model_validator
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
    )

    environment: Environment = Environment.LOCAL
    database_url: SecretStr
    mlflow_tracking_uri: str | None = None
    mlflow_tracking_username: str | None = None
    mlflow_tracking_password: SecretStr | None = None
    hf_token: SecretStr | None = None

    @field_validator("database_url", mode="before")
    @classmethod
    def _use_psycopg_driver(cls, value: object) -> object:
        """Neon hands out ``postgresql://`` URLs; SQLAlchemy needs the psycopg 3 driver named."""
        if isinstance(value, str):
            for scheme in _PLAIN_POSTGRES_SCHEMES:
                if value.startswith(scheme):
                    return _PSYCOPG_SCHEME + value.removeprefix(scheme)
        return value

    @model_validator(mode="after")
    def _production_needs_every_credential(self) -> Self:
        if self.environment is not Environment.PRODUCTION:
            return self
        required = {
            "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
            "MLFLOW_TRACKING_USERNAME": self.mlflow_tracking_username,
            "MLFLOW_TRACKING_PASSWORD": self.mlflow_tracking_password,
            "HF_TOKEN": self.hf_token,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise ValueError(f"ENVIRONMENT=production requires: {', '.join(missing)}")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, loaded once. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
