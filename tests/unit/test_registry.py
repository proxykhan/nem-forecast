from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import polars as pl
import pytest

from nem_forecast.backtest.baselines import BASELINES
from nem_forecast.backtest.harness import BacktestConfig, run_backtest
from nem_forecast.models import registry
from nem_forecast.models.registry import PromotionError, push_to_hub, render_model_card
from nem_forecast.models.ridge import RidgeForecaster
from nem_forecast.schemas import Region
from tests.synthetic import series

DATA = series(days=75, noise_sd=80.0)


@pytest.fixture(scope="module")
def recorded() -> pl.DataFrame:
    """A backtest in the shape latest_backtest() reads back from model_run_metrics."""
    result = run_backtest(
        DATA,
        [*(b() for b in BASELINES), RidgeForecaster()],
        BacktestConfig(test_days=6, refit_every_days=3),
    )
    return result.metrics.with_columns(
        kind=pl.col("model").replace_strict(result.kinds),
        eval_start=pl.lit(result.eval_start),
        eval_end=pl.lit(result.eval_end),
    )


METADATA: dict[str, Any] = {
    "name": "ridge",
    "description": "Ridge regression on standardised lags and one-hot calendar, per horizon group",
    "version": "0123456789abcdef0123456789abcdef01234567",
    "features": ["horizon", "half_hour"],
    "trained_on": ["2023-10-01T00:30:00+00:00", "2026-10-13T13:00:00+00:00"],
}


def test_card_carries_every_required_section(recorded: pl.DataFrame) -> None:
    card = render_model_card(METADATA, recorded, region=Region.VIC1, repo_id="me/nem-forecast-vic1")

    for section in (
        "## Intended use",
        "## Training data",
        "## Features",
        "## Evaluation",
        "## Known failure modes",
    ):
        assert section in card
    assert "**Version:** `0123456789abcdef0123456789abcdef01234567`" in card
    assert "2023-10-01 10:30 to 2026-10-13 23:00" in card  # trained range in market time
    assert "seasonal naive" in card  # the verdict line, win or lose
    for baseline in BASELINES:
        assert f"`{baseline.name}`" in card
    assert "**`ridge`**" in card
    assert 'snapshot_download("me/nem-forecast-vic1", revision="0123456789ab")' in card
    assert card.startswith("---\nlicense: mit")


def test_card_states_interval_calibration_from_the_numbers(recorded: pl.DataFrame) -> None:
    card = render_model_card(METADATA, recorded, region=Region.VIC1, repo_id="me/x")
    coverage = recorded.filter(
        pl.col("model") == "ridge",
        pl.col("metric") == "coverage_80",
        pl.col("horizon") == 0,
        pl.col("tod_bucket") == "all",
    )["value"][0]

    assert f"contained {coverage:.1f}% of actual demand" in card


def test_card_needs_a_recorded_backtest(recorded: pl.DataFrame) -> None:
    with pytest.raises(PromotionError, match="no recorded backtest for lightgbm_direct"):
        render_model_card(
            METADATA | {"name": "lightgbm_direct"},
            recorded,
            region=Region.VIC1,
            repo_id="me/x",
        )


@pytest.mark.parametrize("version", ["", "unknown", "abc1234-dirty"])
def test_uncommitted_code_is_never_promoted(version: str, tmp_path: Path) -> None:
    with pytest.raises(PromotionError, match="commit your changes"):
        push_to_hub(tmp_path, token="hf_x", region=Region.VIC1, version=version)


class FakeHub:
    calls: ClassVar[list[tuple[Any, ...]]] = []

    def __init__(self, token: str) -> None:
        self.calls.append(("token", token))

    def whoami(self) -> dict[str, str]:
        return {"name": "someone"}

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.calls.append(("create_repo", repo_id, kwargs))

    def upload_folder(self, **kwargs: Any) -> None:
        self.calls.append(("upload_folder", kwargs))

    def create_tag(self, repo_id: str, **kwargs: Any) -> None:
        self.calls.append(("create_tag", repo_id, kwargs))


def test_promotion_uploads_the_folder_and_tags_the_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeHub.calls = []
    monkeypatch.setattr(registry, "HfApi", FakeHub)
    sha = "0123456789abcdef0123456789abcdef01234567"

    url = push_to_hub(tmp_path, token="hf_x", region=Region.VIC1, version=sha)

    assert url == "https://huggingface.co/someone/nem-forecast-vic1/tree/0123456789ab"
    kinds = [c[0] for c in FakeHub.calls]
    assert kinds == ["token", "create_repo", "upload_folder", "create_tag"]
    assert FakeHub.calls[1][2]["private"] is False
    assert FakeHub.calls[2][1]["folder_path"] == tmp_path
    assert FakeHub.calls[3][2]["tag"] == "0123456789ab"
