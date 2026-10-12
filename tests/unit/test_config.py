from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nem_forecast.config import Environment, Settings, get_settings

NEON_URL = "postgresql://user:s3cret@ep-x.ap-southeast-2.aws.neon.tech/neondb?sslmode=require"


def load() -> Settings:
    """Build Settings from the environment only, ignoring any .env on disk."""
    return Settings(_env_file=None)


def test_loads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", NEON_URL)
    monkeypatch.setenv("ENVIRONMENT", "ci")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/me/nem-forecast.mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "me")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "mlflow-token")
    monkeypatch.setenv("HF_TOKEN", "hf_Xq7pL2mN9vR4")

    settings = load()

    assert settings.environment is Environment.CI
    assert settings.mlflow_tracking_uri == "https://dagshub.com/me/nem-forecast.mlflow"
    assert settings.mlflow_tracking_username == "me"
    assert settings.mlflow_tracking_password is not None
    assert settings.mlflow_tracking_password.get_secret_value() == "mlflow-token"
    assert settings.hf_token is not None
    assert settings.hf_token.get_secret_value() == "hf_Xq7pL2mN9vR4"


def test_defaults_to_local_with_optional_credentials_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", NEON_URL)

    settings = load()

    assert settings.environment is Environment.LOCAL
    assert settings.mlflow_tracking_uri is None
    assert settings.hf_token is None


def test_database_url_is_required() -> None:
    with pytest.raises(ValidationError, match="database_url"):
        load()


@pytest.mark.parametrize(
    ("given", "expected_prefix"),
    [
        ("postgresql://u:p@host/db", "postgresql+psycopg://"),
        ("postgres://u:p@host/db", "postgresql+psycopg://"),
        ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://"),
        ("sqlite:///local.db", "sqlite:///"),
    ],
)
def test_database_url_uses_psycopg_driver(
    monkeypatch: pytest.MonkeyPatch, given: str, expected_prefix: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", given)

    url = load().database_url.get_secret_value()

    assert url.startswith(expected_prefix)
    assert url.endswith(given.split("://", 1)[1])


def test_secrets_never_appear_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", NEON_URL)
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "mlflow-token")
    monkeypatch.setenv("HF_TOKEN", "hf_Xq7pL2mN9vR4")

    rendered = repr(load()) + str(load().model_dump())

    for secret in ("s3cret", "mlflow-token", "hf_Xq7pL2mN9vR4"):
        assert secret not in rendered


def test_production_requires_every_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", NEON_URL)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/me/nem-forecast.mlflow")

    with pytest.raises(ValidationError, match="HF_TOKEN") as excinfo:
        load()

    assert "MLFLOW_TRACKING_PASSWORD" in str(excinfo.value)
    assert "MLFLOW_TRACKING_URI" not in str(excinfo.value)


def test_production_with_every_credential_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "DATABASE_URL": NEON_URL,
        "ENVIRONMENT": "production",
        "MLFLOW_TRACKING_URI": "https://dagshub.com/me/nem-forecast.mlflow",
        "MLFLOW_TRACKING_USERNAME": "me",
        "MLFLOW_TRACKING_PASSWORD": "mlflow-token",
        "HF_TOKEN": "hf_Xq7pL2mN9vR4",
    }.items():
        monkeypatch.setenv(name, value)

    assert load().environment is Environment.PRODUCTION


def test_unknown_environment_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", NEON_URL)
    monkeypatch.setenv("ENVIRONMENT", "staging")

    with pytest.raises(ValidationError, match="environment"):
        load()


def test_reads_dotenv_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"DATABASE_URL={NEON_URL}\nENVIRONMENT=ci\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.environment is Environment.CI
    assert settings.database_url.get_secret_value().startswith("postgresql+psycopg://")


def test_environment_variables_override_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"DATABASE_URL={NEON_URL}\nENVIRONMENT=ci\n", encoding="utf-8")
    monkeypatch.setenv("ENVIRONMENT", "local")

    settings = Settings(_env_file=env_file)

    assert settings.environment is Environment.LOCAL


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setenv("DATABASE_URL", NEON_URL)

    first = get_settings()
    monkeypatch.setenv("DATABASE_URL", "postgresql://other/db")

    assert get_settings() is first
