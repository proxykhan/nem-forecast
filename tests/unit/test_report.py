from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nem_forecast.backtest.baselines import BASELINES
from nem_forecast.backtest.harness import BacktestConfig, run_backtest
from nem_forecast.backtest.report import CHART_NAME, REPORT_NAME, horizon_label, write_report
from nem_forecast.schemas import Region
from tests.synthetic import series


def test_report_has_every_table_and_a_chart(tmp_path: Path) -> None:
    result = run_backtest(
        series(days=60, noise_sd=100.0),
        [b() for b in BASELINES],
        BacktestConfig(test_days=10, refit_every_days=5),
    )
    run_ids = {b.name: uuid.uuid4() for b in BASELINES}

    path = write_report(
        result,
        tmp_path / "out",
        region=Region.VIC1,
        git_sha="abc1234",
        generated_at=datetime(2026, 10, 13, tzinfo=UTC),
        run_ids=run_ids,
    )
    text = path.read_text(encoding="utf-8")

    assert path.name == REPORT_NAME
    for heading in (
        "## Overall",
        "## MAE (MW) by horizon",
        "by horizon",
        "## MAE (MW) by time of day",
    ):
        assert heading in text
    for baseline in BASELINES:
        assert f"`{baseline.name}` | baseline | {baseline.description}" in text
        assert str(run_ids[baseline.name]) in text
    assert f"![MAE by horizon]({CHART_NAME})" in text
    assert "commit `abc1234`" in text
    assert (tmp_path / "out" / CHART_NAME).read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize(("h", "label"), [(1, "30 min"), (2, "1 h"), (3, "1.5 h"), (48, "24 h")])
def testhorizon_labels(h: int, label: str) -> None:
    assert horizon_label(h) == label


def test_report_with_models_states_a_verdict_for_each(tmp_path: Path) -> None:
    from nem_forecast.backtest.report import _ranges, verdict
    from nem_forecast.models.ridge import RidgeForecaster

    result = run_backtest(
        series(days=75, noise_sd=100.0),
        [*(b() for b in BASELINES), RidgeForecaster()],
        BacktestConfig(test_days=6, refit_every_days=3),
    )

    path = write_report(
        result,
        tmp_path,
        region=Region.VIC1,
        git_sha="abc1234",
        generated_at=datetime(2026, 10, 13, tzinfo=UTC),
        mlflow_runs={"ridge": "mlflow-run-1"},
    )
    text = path.read_text(encoding="utf-8")

    assert "## Verdict" in text
    assert f"- `ridge`: {verdict(result, 'ridge')}" in text
    assert "seasonal naive" in verdict(result, "ridge")
    assert "## MAE change against seasonal naive, by horizon" in text
    assert "`mlflow-run-1`" in text
    assert "| `ridge` | model |" in text
    assert _ranges([1, 2, 3, 7, 9, 10]) == "1-3, 7, 9-10"
