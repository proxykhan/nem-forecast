"""Render a backtest as a markdown report plus an error-by-horizon chart, in docs/backtest/.

When models are present the report states, per model and computed from the metrics, whether it
beats seasonal naive and the best baseline at every horizon, so a loss is reported in the same
words as a win.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from nem_forecast.backtest.harness import BacktestResult
from nem_forecast.backtest.metrics import ALL_BUCKETS, ALL_HORIZONS, TOD_BUCKETS
from nem_forecast.schemas import NEM_TZ, Region

REPORT_NAME = "README.md"
CHART_NAME = "mae_by_horizon.png"
REPORT_HORIZONS = (1, 2, 4, 6, 12, 24, 36, 48)
REFERENCE_BASELINE = "seasonal_naive"
BEST_BASELINE = "best baseline"

# Reference categorical palette, slots in fixed order (validated light mode: CVD and
# normal-vision separation pass; aqua and yellow are below 3:1 contrast, so every line is
# direct-labelled and the report carries the numbers as tables).
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def write_report(
    result: BacktestResult,
    out_dir: Path,
    *,
    region: Region,
    git_sha: str | None,
    generated_at: datetime,
    run_ids: dict[str, uuid.UUID] | None = None,
    mlflow_runs: dict[str, str] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_mae_by_horizon(result, out_dir / CHART_NAME, region=region)
    path = out_dir / REPORT_NAME
    path.write_text(
        render_markdown(
            result,
            region=region,
            git_sha=git_sha,
            generated_at=generated_at,
            run_ids=run_ids,
            mlflow_runs=mlflow_runs,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return path


def mae_by_horizon(metrics: pl.DataFrame) -> pl.DataFrame:
    """Wide frame: horizon, then one MAE column per model."""
    return (
        metrics.filter(
            pl.col("metric") == "mae", pl.col("tod_bucket") == ALL_BUCKETS, pl.col("horizon") > 0
        )
        .pivot(on="model", index="horizon", values="value")
        .sort("horizon")
    )


def verdict(result: BacktestResult, model: str) -> str:
    """Plain statement of where ``model`` beats or loses to the baselines, by MAE."""
    baselines = [m for m, kind in result.kinds.items() if kind == "baseline"]
    return verdict_from_metrics(mae_by_horizon(result.metrics), model, baselines)


def verdict_from_metrics(wide: pl.DataFrame, model: str, baselines: Sequence[str]) -> str:
    """The same statement from a wide MAE-by-horizon frame (one column per model)."""
    parts = []
    if REFERENCE_BASELINE in wide.columns:
        parts.append(_versus(wide, model, pl.col(REFERENCE_BASELINE), "seasonal naive"))
    if baselines:
        parts.append(
            _versus(wide, model, pl.min_horizontal(baselines), "the best baseline at each horizon")
        )
    return " ".join(parts)


def _versus(wide: pl.DataFrame, model: str, reference: pl.Expr, name: str) -> str:
    losing = wide.filter(pl.col(model) >= reference)["horizon"].to_list()
    total = wide.height
    if not losing:
        return f"Beats {name} at all {total} horizons."
    return (
        f"**Does not beat {name} at {len(losing)} of {total} horizons** "
        f"({_ranges(losing)}, in half-hours ahead)."
    )


def render_markdown(
    result: BacktestResult,
    *,
    region: Region,
    git_sha: str | None,
    generated_at: datetime,
    run_ids: dict[str, uuid.UUID] | None = None,
    mlflow_runs: dict[str, str] | None = None,
) -> str:
    m = result.metrics
    cfg = result.config
    names = list(result.descriptions)
    models = [n for n in names if result.kinds.get(n) == "model"]
    forecasts = m.filter(
        pl.col("metric") == "mae",
        pl.col("horizon") == ALL_HORIZONS,
        pl.col("tod_bucket") == ALL_BUCKETS,
    )["n_obs"].to_list()

    def value(
        model: str, metric: str, horizon: int = ALL_HORIZONS, bucket: str = ALL_BUCKETS
    ) -> float:
        row = m.filter(
            pl.col("model") == model,
            pl.col("metric") == metric,
            pl.col("horizon") == horizon,
            pl.col("tod_bucket") == bucket,
        )
        return float(row["value"][0]) if row.height else float("nan")

    intro = (
        "Baselines and models are scored on identical forecasts. Baselines are fixed rules and "
        "were not tuned; LightGBM hyperparameters were tuned on a validation split that ends "
        "where this backtest begins."
        if models
        else "These numbers are the bar every model has to clear; nothing here was tuned."
    )
    lines = [
        f"# Backtest: {region}",
        "",
        f"Generated by `nem-forecast backtest` on {generated_at.astimezone(NEM_TZ):%Y-%m-%d %H:%M} "
        f"AEST from commit `{git_sha or 'unknown'}`. {intro}",
        "",
        "## Setup",
        "",
        f"- **Data:** AEMO 5-minute operational demand averaged to half-hours, "
        f"{_day(result.train_start)} to {_day(result.eval_end)}.",
        f"- **Evaluation:** walk-forward over {_day(result.eval_start)} to {_day(result.eval_end)} "
        f"({cfg.test_days} days). Expanding training window, refit every {cfg.refit_every_days} "
        f"days ({len(result.folds)} refits), a forecast from every "
        f"{cfg.origin_every_minutes}-minute origin, {len(cfg.horizons)} half-hour horizons. "
        f"{max(forecasts, default=0):,} scored forecasts per model.",
        "- **Intervals:** baselines use each rule's 10th and 90th error percentiles over the 90 "
        "days before each refit."
        + (
            " Ridge and LightGBM direct use their own errors on the 28 days before each refit, "
            "predicted by a copy trained without those days. LightGBM quantile predicts its "
            "quantiles directly."
            if models
            else ""
        ),
        "- **Time of day:** Melbourne wall-clock time of the target half-hour.",
    ]
    if models:
        lines += ["", "## Verdict", ""]
        lines += [f"- `{model}`: {verdict(result, model)}" for model in models]

    lines += [
        "",
        "## Overall",
        "",
        "| Model | Kind | Description | MAE (MW) | RMSE (MW) | MAPE (%) | Pinball avg (MW) "
        "| 10-90 coverage (%) |",
        "|---|---|---|--:|--:|--:|--:|--:|",
    ]
    for name in names:
        lines.append(
            f"| `{name}` | {result.kinds.get(name, 'model')} | {result.descriptions[name]} | "
            f"{value(name, 'mae'):,.0f} | {value(name, 'rmse'):,.0f} | {value(name, 'mape'):.1f} | "
            f"{value(name, 'pinball_avg'):,.1f} | {value(name, 'coverage_80'):.1f} |"
        )

    header = " | ".join(horizon_label(h) for h in REPORT_HORIZONS)
    divider = "|".join("--:" for _ in REPORT_HORIZONS)
    for title, metric in (
        ("MAE (MW) by horizon", "mae"),
        ("Pinball loss, average of q10/q50/q90 (MW), by horizon", "pinball_avg"),
    ):
        lines += ["", f"## {title}", ""]
        if metric == "mae":
            lines += [f"![MAE by horizon]({CHART_NAME})", ""]
        lines += [f"| Model | {header} |", f"|---|{divider}|"]
        for name in names:
            cells = " | ".join(f"{value(name, metric, h):,.0f}" for h in REPORT_HORIZONS)
            lines.append(f"| `{name}` | {cells} |")

    if models and REFERENCE_BASELINE in names:
        lines += [
            "",
            "## MAE change against seasonal naive, by horizon",
            "",
            "Negative is better than seasonal naive.",
            "",
            f"| Model | {header} |",
            f"|---|{divider}|",
        ]
        for name in [n for n in names if n != REFERENCE_BASELINE]:
            cells = " | ".join(
                f"{100 * (value(name, 'mae', h) / value(REFERENCE_BASELINE, 'mae', h) - 1):+.0f}%"
                for h in REPORT_HORIZONS
            )
            lines.append(f"| `{name}` | {cells} |")

    buckets = [name for name, _, _ in TOD_BUCKETS]
    lines += [
        "",
        "## MAE (MW) by time of day",
        "",
        "| Model | " + " | ".join(buckets) + " |",
        "|---|" + "|".join("--:" for _ in buckets) + "|",
    ]
    for name in names:
        cells = " | ".join(f"{value(name, 'mae', bucket=b):,.0f}" for b in buckets)
        lines.append(f"| `{name}` | {cells} |")

    lines += [
        "",
        "## Where the full numbers live",
        "",
        "Every metric for every horizon and time-of-day bucket is in `model_run_metrics`",
        "(`horizon = 0` and `tod_bucket = 'all'` are the pooled rows). Fold-level runs are in",
        "MLflow, nested under each model's run.",
    ]
    if run_ids or mlflow_runs:
        lines += ["", "| Model | model_runs.run_id | MLflow run |", "|---|---|---|"]
        for name in names:
            db_id = f"`{run_ids[name]}`" if run_ids and name in run_ids else ""
            ml_id = f"`{mlflow_runs[name]}`" if mlflow_runs and name in mlflow_runs else ""
            lines.append(f"| `{name}` | {db_id} | {ml_id} |")
    return "\n".join(lines) + "\n"


README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
README_HORIZONS = (1, 2, 4, 6, 12, 24, 36, 48)
_NAMES = {
    "lightgbm_quantile": "LightGBM quantile",
    "lightgbm_direct": "LightGBM direct",
    "ridge": "Ridge",
    "seasonal_naive": "Seasonal naive",
    "weekly_seasonal_naive": "Weekly seasonal naive",
    "same_weekday_mean_4w": "4-week same-weekday mean",
    "persistence": "Persistence",
}


def render_results_block(result: BacktestResult, *, generated_at: datetime) -> str:
    """The README's results tables, regenerated by every nightly backtest."""
    m = result.metrics

    def value(model: str, metric: str, horizon: int = ALL_HORIZONS) -> float:
        row = m.filter(
            pl.col("model") == model,
            pl.col("metric") == metric,
            pl.col("horizon") == horizon,
            pl.col("tod_bucket") == ALL_BUCKETS,
        )
        return float(row["value"][0]) if row.height else float("nan")

    names = sorted(
        result.descriptions, key=lambda n: (result.kinds.get(n) != "model", value(n, "mae"))
    )
    label = {n: _NAMES.get(n, n) for n in names}
    header = " | ".join(horizon_label(h) for h in README_HORIZONS)
    divider = "|".join("--:" for _ in README_HORIZONS)
    forecasts = m.filter(
        pl.col("metric") == "mae",
        pl.col("horizon") == ALL_HORIZONS,
        pl.col("tod_bucket") == ALL_BUCKETS,
    )["n_obs"].to_list()
    lines = [
        f"Walk-forward backtest over {_day(result.eval_start)} to {_day(result.eval_end)}, "
        f"{max(forecasts, default=0):,} forecasts per model, refreshed nightly "
        f"(last run {generated_at.astimezone(NEM_TZ):%Y-%m-%d}; "
        "[full report](docs/backtest/README.md)).",
        "",
        "| Model | Kind | MAE (MW) | MAPE (%) | Pinball avg (MW) | 10-90 coverage (%) |",
        "|---|---|--:|--:|--:|--:|",
    ]
    for n in names:
        lines.append(
            f"| {label[n]} | {result.kinds.get(n, 'model')} | {value(n, 'mae'):,.0f} | "
            f"{value(n, 'mape'):.1f} | {value(n, 'pinball_avg'):,.1f} | "
            f"{value(n, 'coverage_80'):.1f} |"
        )
    for title, metric in (
        ("MAE (MW) by horizon", "mae"),
        ("Pinball loss, average of q10/q50/q90 (MW), by horizon", "pinball_avg"),
    ):
        lines += ["", f"**{title}**", "", f"| Model | {header} |", f"|---|{divider}|"]
        for n in names:
            cells = " | ".join(f"{value(n, metric, h):,.0f}" for h in README_HORIZONS)
            lines.append(f"| {label[n]} | {cells} |")
    verdicts = [
        f"- **{label[n]}**: {verdict(result, n)}" for n in names if result.kinds.get(n) == "model"
    ]
    if verdicts:
        lines += ["", *verdicts]
    return "\n".join(lines)


def update_readme(path: Path, block: str) -> None:
    """Replace the text between the results markers in ``path``."""
    text = path.read_text(encoding="utf-8")
    start, end = text.find(README_START), text.find(README_END)
    if start < 0 or end < start:
        raise ValueError(f"{path} has no {README_START} ... {README_END} section")
    updated = f"{text[: start + len(README_START)]}\n{block}\n{text[end:]}"
    path.write_text(updated, encoding="utf-8", newline="\n")


def plot_mae_by_horizon(result: BacktestResult, path: Path, *, region: Region) -> None:
    """Models against the best baseline at each horizon; baselines alone when there are none."""
    wide = mae_by_horizon(result.metrics)
    names = list(result.descriptions)
    models = [n for n in names if result.kinds.get(n) == "model"]
    baselines = [n for n in names if result.kinds.get(n) == "baseline"]
    if models and baselines:
        wide = wide.with_columns(pl.min_horizontal(baselines).alias(BEST_BASELINE))
        series = [(name, SERIES_COLOURS[i]) for i, name in enumerate(models)]
        series.append((BEST_BASELINE, TEXT_MUTED))
        title = f"Forecast error by horizon: {region} models against the best baseline"
    else:
        series = [(name, SERIES_COLOURS[i % len(SERIES_COLOURS)]) for i, name in enumerate(names)]
        title = f"Forecast error by horizon: {region} baselines"

    fig = Figure(figsize=(9, 5), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes((0.09, 0.12, 0.66, 0.70), facecolor=SURFACE)
    horizons = wide["horizon"].to_list()
    ends: list[tuple[float, str, str]] = []
    for name, colour in series:
        values = wide[name].to_list()
        ax.plot(horizons, values, color=colour, linewidth=1.6, solid_capstyle="round", label=name)
        ax.plot(
            horizons[-1:],
            values[-1:],
            "o",
            color=colour,
            markersize=6.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
        ends.append((values[-1], name, colour))

    ymax = max((v for name, _ in series for v in wide[name].to_list()), default=1.0) * 1.15
    ax.set_ylim(0, ymax)
    ax.set_xlim(0.5, 48.5)
    _direct_labels(ax, ends, ymax)

    ax.set_xticks([1, 12, 24, 36, 48], ["30 min", "6 h", "12 h", "18 h", "24 h"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Forecast horizon (time ahead of the origin)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("MAE (MW)", color=TEXT_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8.5, length=0)

    fig.text(0.09, 0.94, title, fontsize=12.5, color=TEXT_PRIMARY, weight="bold")
    fig.text(
        0.09,
        0.895,
        "Mean absolute error of the median forecast, walk-forward backtest",
        fontsize=9,
        color=TEXT_SECONDARY,
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=len(series),
        frameon=False,
        fontsize=8,
        labelcolor=TEXT_SECONDARY,
        handlelength=1.6,
        columnspacing=1.4,
        borderaxespad=0.3,
    )
    fig.savefig(path, facecolor=SURFACE)


def _direct_labels(ax: Axes, ends: list[tuple[float, str, str]], ymax: float) -> None:
    """Label each line's end; crowded labels are spread evenly around their lines' mean height,
    each joined to its line end by a thin leader in the series colour."""
    ordered = sorted(ends)
    gap = ymax * 0.07
    positions: list[float] = []
    for value, _, _ in ordered:
        positions.append(max(value, positions[-1] + gap) if positions else value)
    shift = sum(v for v, _, _ in ordered) / len(ordered) - sum(positions) / len(positions)
    shift = max(shift, gap / 2 - positions[0])
    for (value, model, colour), y in zip(ordered, positions, strict=True):
        ax.annotate(
            model,
            xy=(48, value),
            xytext=(50.5, y + shift),
            textcoords="data",
            annotation_clip=False,
            va="center",
            fontsize=8.5,
            color=TEXT_PRIMARY,
            arrowprops={
                "arrowstyle": "-",
                "color": colour,
                "linewidth": 0.9,
                "shrinkA": 2,
                "shrinkB": 5,
                "relpos": (0, 0.5),
            },
        )


def _ranges(horizons: Sequence[int]) -> str:
    """[1, 2, 3, 7, 9, 10] -> '1-3, 7, 9-10'."""
    spans: list[list[int]] = []
    for h in sorted(horizons):
        if spans and h == spans[-1][-1] + 1:
            spans[-1].append(h)
        else:
            spans.append([h])
    return ", ".join(f"{s[0]}-{s[-1]}" if len(s) > 1 else str(s[0]) for s in spans)


def horizon_label(h: int) -> str:
    minutes = h * 30
    return f"{minutes} min" if minutes < 60 else f"{minutes / 60:g} h"


def _day(ts: datetime) -> str:
    return f"{ts.astimezone(NEM_TZ):%Y-%m-%d}"
