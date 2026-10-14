"""Model loading and process wiring for the API, without a database."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from nem_forecast.config import Settings
from nem_forecast.models.ridge import RidgeForecaster
from nem_forecast.serve import __main__ as entrypoint
from nem_forecast.serve.model_store import ModelState, ModelStore
from nem_forecast.serve.problems import Problem
from tests.synthetic import series

VERSION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model = RidgeForecaster()
    model.fit(series(days=60, noise_sd=50.0))
    directory = tmp_path_factory.mktemp("model")
    model.save(directory, version=VERSION)
    return directory


def settings(**values: Any) -> Settings:
    return Settings(_env_file=None, database_url=SecretStr("sqlite://"), **values)


def test_loads_a_local_model_directory(saved_model: Path) -> None:
    store = ModelStore(settings(model_dir=saved_model))

    store.load()

    assert store.state is ModelState.READY
    assert store.loaded is not None
    assert (store.loaded.name, store.loaded.version) == ("ridge", VERSION[:12])
    assert store.loaded.max_horizon == 48
    assert store.loaded.source == f"local:{saved_model}"


def test_downloads_the_pinned_revision_into_the_cache(saved_model: Path, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_download(repo_id: str, **kwargs: Any) -> str:
        calls.append((repo_id, kwargs))
        return str(saved_model)

    store = ModelStore(
        settings(
            model_repo_id="someone/nem-forecast-vic1",
            model_revision="fa90c4b4c9a3",
            model_cache_dir=tmp_path / "cache",
        ),
        download=fake_download,
    )
    store.load()
    store.load()  # already loaded: no second download

    assert calls == [
        (
            "someone/nem-forecast-vic1",
            {"revision": "fa90c4b4c9a3", "cache_dir": tmp_path / "cache"},
        )
    ]
    assert store.loaded is not None
    assert store.loaded.source == "hf:someone/nem-forecast-vic1@fa90c4b4c9a3"


def test_a_failed_download_is_reported_not_raised(tmp_path: Path) -> None:
    def broken(repo_id: str, **kwargs: Any) -> str:
        raise ConnectionError("hub unreachable")

    store = ModelStore(
        settings(model_repo_id="someone/x", model_cache_dir=tmp_path), download=broken
    )
    store.load()

    assert store.state is ModelState.FAILED
    assert store.error == "ConnectionError: hub unreachable"
    with pytest.raises(Problem) as excinfo:
        store.require()
    assert (excinfo.value.status, excinfo.value.slug) == (503, "model-unavailable")


def test_listeners_hear_about_the_model_once_it_loads(saved_model: Path) -> None:
    store = ModelStore(settings(model_dir=saved_model))
    heard: list[str] = []
    store.on_loaded(lambda m: heard.append(f"early {m.version}"))

    store.load()
    store.on_loaded(lambda m: heard.append(f"late {m.version}"))

    assert heard == [f"early {VERSION[:12]}", f"late {VERSION[:12]}"]


def test_without_a_model_source_the_store_is_not_configured() -> None:
    store = ModelStore(settings())
    store.load()

    assert store.state is ModelState.NOT_CONFIGURED


def test_region_defaults_to_vic1_for_models_saved_without_one(saved_model: Path) -> None:
    metadata = json.loads((saved_model / "model.json").read_text(encoding="utf-8"))
    assert "region" not in metadata

    store = ModelStore(settings(model_dir=saved_model))
    store.load()

    assert store.loaded is not None
    assert store.loaded.region.value == "VIC1"


def test_entrypoint_runs_the_app_factory_on_port_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: seen.update(kwargs, app=app))
    monkeypatch.setenv("PORT", "10000")

    entrypoint.run()

    assert seen["app"] == "nem_forecast.serve.app:create_app"
    assert seen["factory"] is True
    assert seen["port"] == 10000
    assert seen["host"] == "0.0.0.0"  # noqa: S104 - inside a container


HEAVY = ("mlflow", "matplotlib", "optuna", "sklearn", "evidently", "pandas", "pyarrow")
# Make the training extras look uninstalled, exactly as in the core-only serving image.
# (LightGBM imports pandas and scikit-learn opportunistically when they happen to be present.)
PROBE = f"""
import importlib.abc, sys
class Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {HEAVY!r}:
            raise ModuleNotFoundError(name)
sys.meta_path.insert(0, Absent())
import nem_forecast.serve.app, nem_forecast.serve.__main__, nem_forecast.cli
print("ok")
"""


def test_serving_and_ingestion_work_without_the_training_extras() -> None:
    result = subprocess.run(  # noqa: S603 - fixed probe
        [sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=120, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize(
    ("baked", "render", "expected"),
    [
        ("abc1234", "fromrender", "abc1234"),  # passed at image build
        ("", "fromrender", "fromrender"),  # Render builds without the build argument
        ("unknown", "fromrender", "fromrender"),  # older images baked a placeholder
    ],
)
def test_service_sha_prefers_a_real_baked_sha_then_render(
    monkeypatch: pytest.MonkeyPatch, baked: str, render: str, expected: str
) -> None:
    from nem_forecast.serve.app import service_sha

    monkeypatch.setenv("RENDER_GIT_COMMIT", render)

    assert service_sha(settings(git_sha=baked)) == expected
