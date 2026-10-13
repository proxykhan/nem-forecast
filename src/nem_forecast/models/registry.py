"""Promote a trained model to Hugging Face Hub with a model card built from the recorded backtest.

Every number on the card comes from the latest backtest of each model in model_run_metrics; none
is typed by hand. Versions are git SHAs, and promotion refuses uncommitted code, because a model
whose code cannot be checked out again cannot be audited.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from huggingface_hub import HfApi
from sqlalchemy import Engine, text

from nem_forecast.backtest.metrics import ALL_BUCKETS, ALL_HORIZONS, TOD_BUCKETS
from nem_forecast.backtest.report import horizon_label, verdict_from_metrics
from nem_forecast.schemas import NEM_TZ, Region

GITHUB_REPO = "https://github.com/proxykhan/nem-forecast"
CARD_HORIZONS = (1, 2, 4, 6, 12, 24, 48)
CARD_FILES = ("README.md", "model_card.md")  # the Hub renders README.md as the card

_LATEST_BACKTEST = """
WITH latest AS (
    SELECT DISTINCT ON (model_name) run_id, model_name, model_kind, eval_start, eval_end
    FROM model_runs
    WHERE region = :region AND run_type = 'backtest' AND status = 'succeeded'
    ORDER BY model_name, started_at DESC
)
SELECT l.model_name, l.model_kind, l.eval_start, l.eval_end,
       m.metric, m.horizon, m.tod_bucket, m.value, m.n_obs
FROM latest l JOIN model_run_metrics m USING (run_id)
"""


class PromotionError(RuntimeError):
    pass


def repo_name(region: Region) -> str:
    return f"nem-forecast-{region.value.lower()}"


def latest_backtest(engine: Engine, region: Region) -> pl.DataFrame:
    """Tidy metrics of the most recent successful backtest of every model in a region."""
    with engine.connect() as conn:
        rows = conn.execute(text(_LATEST_BACKTEST), {"region": region.value}).all()
    columns = [
        "model",
        "kind",
        "eval_start",
        "eval_end",
        "metric",
        "horizon",
        "tod_bucket",
        "value",
        "n_obs",
    ]
    return pl.DataFrame([tuple(r) for r in rows], schema=columns, orient="row")


def render_model_card(
    metadata: dict[str, Any], backtest: pl.DataFrame, *, region: Region, repo_id: str
) -> str:
    name = metadata["name"]
    version = metadata["version"]
    if backtest.filter(pl.col("model") == name).is_empty():
        raise PromotionError(f"no recorded backtest for {name}; run `nem-forecast backtest` first")
    kinds = dict(backtest.select("model", "kind").unique().iter_rows())
    baselines = sorted(m for m, k in kinds.items() if k == "baseline")
    recorded = backtest.filter(pl.col("model") == name).row(0, named=True)
    eval_start, eval_end = recorded["eval_start"], recorded["eval_end"]

    def value(
        model: str, metric: str, horizon: int = ALL_HORIZONS, bucket: str = ALL_BUCKETS
    ) -> float:
        row = backtest.filter(
            pl.col("model") == model,
            pl.col("metric") == metric,
            pl.col("horizon") == horizon,
            pl.col("tod_bucket") == bucket,
        )
        return float(row["value"][0]) if row.height else float("nan")

    trained_from, trained_to = (
        datetime.fromisoformat(ts).astimezone(NEM_TZ) for ts in metadata["trained_on"]
    )
    coverage = value(name, "coverage_80")
    buckets = {b: value(name, "mae", bucket=b) for b, _, _ in TOD_BUCKETS}
    worst = max(buckets, key=lambda b: buckets[b])
    best = min(buckets, key=lambda b: buckets[b])
    header = " | ".join(horizon_label(h) for h in CARD_HORIZONS)
    divider = "|".join("--:" for _ in CARD_HORIZONS)

    lines = [
        "---",
        "license: mit",
        "library_name: lightgbm" if name.startswith("lightgbm") else "library_name: scikit-learn",
        "pipeline_tag: time-series-forecasting",
        "tags: [time-series-forecasting, electricity-demand, energy, australia, aemo, nem]",
        "---",
        "",
        f"# nem-forecast: {region} half-hourly electricity demand ({name})",
        "",
        f"{metadata['description']}. Forecasts Victorian (`{region}`) operational demand for each "
        "half-hour from 30 minutes to 24 hours ahead, as a median with a 10th-90th percentile "
        f"interval. Part of [nem-forecast]({GITHUB_REPO}), where the training, evaluation and "
        "serving code lives.",
        "",
        f"**Version:** `{version}`, the git commit of the code that trained it "
        f"([browse]({GITHUB_REPO}/tree/{version})).",
        "",
        "## Intended use",
        "",
        "A portfolio demonstration of a production-style forecasting system: short-term VIC1 "
        "demand forecasts for analysis and for showing how the pipeline behaves on real data. "
        "It is **not** intended for market bidding, trading or operational dispatch decisions, "
        "and it has not been validated for any region other than VIC1.",
        "",
        "## Training data",
        "",
        "- **Source:** AEMO NEMWeb dispatch data and the monthly price-and-demand archive, public.",
        "- **Target:** VIC1 operational demand (`TOTALDEMAND`), 5-minute values averaged to "
        "half-hours. Rooftop solar output is not included in operational demand.",
        f"- **Range:** {trained_from:%Y-%m-%d %H:%M} to {trained_to:%Y-%m-%d %H:%M} AEMO market "
        "time (UTC+10).",
        "",
        "## Features",
        "",
        "Every feature is computable at the forecast origin; the pipeline checks this on every row "
        f"([feature definitions]({GITHUB_REPO}#features-and-backtesting)).",
        "",
        *[f"- `{feature}`" for feature in metadata["features"]],
        "",
        "## Evaluation",
        "",
        f"Walk-forward backtest over {eval_start.astimezone(NEM_TZ):%Y-%m-%d} to "
        f"{eval_end.astimezone(NEM_TZ):%Y-%m-%d} (market time): expanding "
        "window, refit every 30 days, a forecast from every half-hourly origin. The same "
        "forecasts were made by every baseline.",
        "",
        verdict_from_metrics(_wide_mae(backtest), name, baselines),
        "",
        "**MAE (MW) by horizon**",
        "",
        f"| Model | {header} | All |",
        f"|---|{divider}|--:|",
    ]
    for model in [name, *baselines]:
        cells = " | ".join(f"{value(model, 'mae', h):,.0f}" for h in CARD_HORIZONS)
        bold = "**" if model == name else ""
        lines.append(f"| {bold}`{model}`{bold} | {cells} | {value(model, 'mae'):,.0f} |")
    lines += [
        "",
        "**Pinball loss, average of q10/q50/q90 (MW), by horizon**",
        "",
        f"| Model | {header} | All |",
        f"|---|{divider}|--:|",
    ]
    for model in [name, *baselines]:
        cells = " | ".join(f"{value(model, 'pinball_avg', h):,.0f}" for h in CARD_HORIZONS)
        bold = "**" if model == name else ""
        lines.append(f"| {bold}`{model}`{bold} | {cells} | {value(model, 'pinball_avg'):,.1f} |")

    if abs(coverage - 80) <= 3:
        calibration = "close to its nominal 80%"
    elif coverage < 80:
        calibration = (
            "against a nominal 80%: the interval is too narrow, so treat p10 and p90 as "
            "optimistic bounds rather than calibrated ones"
        )
    else:
        calibration = "against a nominal 80%: the interval is wider than it needs to be"
    lines += [
        "",
        "## Known failure modes",
        "",
        f"- **No weather inputs.** Error is highest in the `{worst}` bucket "
        f"({buckets[worst]:,.0f} MW MAE, against {buckets[best]:,.0f} MW in `{best}`). Rooftop "
        "solar depends on cloud cover and demand on temperature, and the model sees neither.",
        f"- **Interval calibration.** The 10-90 interval contained {coverage:.1f}% of actual "
        f"demand in the backtest, {calibration}.",
        "- **Horizon-group seams.** Each group of horizons (1 | 2-3 | 4-6 | 7-12 | 13-24 | 25-48 "
        "half-hours) has its own model, so error can step slightly where one hands over to the "
        "next.",
        "- **Rare days.** Victoria has about 13 public holidays a year, so holiday behaviour rests "
        "on few examples; unusual events (heatwaves, storms, market interventions) on fewer.",
        "- **Structural change.** Growth in rooftop solar and batteries keeps reshaping the "
        "demand curve. The model reflects its training window and needs regular retraining; the "
        "project monitors live error and drift for this reason.",
        "- **Revised data.** AEMO revises published values. The model learned from the values as "
        "they stood when it was trained.",
        "",
        "## Files and loading",
        "",
        "`model.json` holds the metadata, feature list, horizon groups and interval offsets; each "
        "`group_*.txt` is a LightGBM booster in its plain-text format. Nothing is pickled. "
        "Loading and building features need the project's package "
        f"(`pip install git+{GITHUB_REPO}`).",
        "",
        "```python",
        "from pathlib import Path",
        "from huggingface_hub import snapshot_download",
        "from nem_forecast.models.catalog import load_model",
        "",
        f'path = snapshot_download("{repo_id}", revision="{version[:12]}")',
        "model = load_model(Path(path))",
        "```",
        "",
    ]
    return "\n".join(lines)


def push_to_hub(directory: Path, *, token: str, region: Region, version: str) -> str:
    """Upload a saved model folder and tag the commit with the git SHA; returns the URL."""
    if not version or version.endswith("-dirty") or version == "unknown":
        raise PromotionError(f"refusing to promote version {version!r}: commit your changes first")
    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/{repo_name(region)}"
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=directory,
        commit_message=f"Promote {version[:12]}",
        repo_type="model",
    )
    api.create_tag(repo_id, tag=version[:12], repo_type="model", exist_ok=True)
    return f"https://huggingface.co/{repo_id}/tree/{version[:12]}"


def _wide_mae(backtest: pl.DataFrame) -> pl.DataFrame:
    return (
        backtest.filter(
            pl.col("metric") == "mae", pl.col("tod_bucket") == ALL_BUCKETS, pl.col("horizon") > 0
        )
        .pivot(on="model", index="horizon", values="value")
        .sort("horizon")
    )
