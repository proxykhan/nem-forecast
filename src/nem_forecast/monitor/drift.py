"""Drift in what the model reads, what it forecasts, and what demand actually does, with Evidently.

Three comparisons of the last ``CURRENT_DAYS`` against a reference from the served model's
training window:

* **Data drift:** the data-derived features (lags and rolling statistics of demand), one row per
  half-hourly origin. Calendar features are left out: their distribution in any week is fixed by
  the calendar (one week holds one month and seven weekdays), so a "shift" there describes the
  window, not the data. Horizon is left out too: every horizon reads the same series, so one row
  per origin (horizon 1) is the input distribution without 48 copies of it.
* **Prediction drift:** the forecasts the service actually served, at every horizon, against the
  model's own forecasts over the reference.
* **Target drift:** realised demand, one value per half-hour.

**The reference is the training window at the same time of year.** VIC1 demand is strongly
seasonal, so the reference is the training window's origins within ``SEASON_DAYS`` of the current
window's day of year, in every year the model saw: this spring against previous springs. If the
training window has no such days, the whole window is used and the verdict says so.

**Thresholds are calibrated, not defaulted.** Evidently's default for large samples, a normed
Wasserstein distance of 0.1, is far too tight for a weather-driven series measured over one
week: scored against its own seasonal reference, *every one* of the 104 weeks to October 2026
crossed it on at least one feature (``nem-forecast calibrate-drift`` reproduces this). An alert
that fires every week is an alert nobody reads. So each column's threshold is a quantile of its
own weekly score over that history, at the level where at least one feature would have alerted
in ``ALARM_RATE`` of past weeks. They live in ``config/drift_<region>.json``. Without that file
the Evidently default is used, and the verdict says so.

**One method, pinned.** Evidently picks a test by sample size unless told otherwise, and a
p-value (smaller is worse) and a distance (bigger is worse) read in opposite directions. Every
column uses the normed Wasserstein distance: how far apart the distributions are, in reference
standard deviations.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import wasserstein_distance

from nem_forecast.features.build import (
    DATA_FEATURES,
    LABEL,
    LOCAL_TIME_ZONE,
    assert_no_leakage,
    build_features,
)
from nem_forecast.features.series import PERIOD, TIME_ZONE, span
from nem_forecast.models.base import WARMUP, DirectModel
from nem_forecast.schemas import Region

with warnings.catch_warnings():  # evidently's own modules emit SyntaxWarnings on import
    warnings.simplefilter("ignore")
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import ValueDrift
    from evidently.presets import DataDriftPreset

log = logging.getLogger(__name__)

CURRENT_DAYS = 7  # one full weekly cycle, so weekdays and weekends are both represented
SEASON_DAYS = 28  # +- days of year around the current window, in every training year
ORIGIN_EVERY = timedelta(minutes=30)  # every half-hour, like the live service
REFERENCE_FORECAST_ORIGINS = 1_000  # sampled origins x 48 horizons: plenty for a distribution
MIN_ROWS = 48 * 48  # one day of served forecasts at every horizon: below this, no verdict
METHOD = "wasserstein"
DEFAULT_THRESHOLD = 0.1  # Evidently's own default for this method
ALARM_RATE = 0.05  # calibrated so the feature alert would have fired in 1 week in 20
CALIBRATION_WEEKS = 104

FORECAST = "forecast"
DRIFT_FEATURES = DATA_FEATURES
INPUT_COLUMNS = (*DRIFT_FEATURES, LABEL)
COLUMNS = (*INPUT_COLUMNS, FORECAST)


def thresholds_path(region: Region, directory: Path = Path("config")) -> Path:
    return directory / f"drift_{region.value.lower()}.json"


@dataclass(frozen=True)
class Thresholds:
    by_column: dict[str, float]
    source: str  # how they were set, for the verdict

    @classmethod
    def default(cls) -> Thresholds:
        return cls(
            {column: DEFAULT_THRESHOLD for column in INPUT_COLUMNS},
            f"Evidently default {DEFAULT_THRESHOLD} (no calibration file)",
        )

    @classmethod
    def load(cls, path: Path) -> Thresholds:
        if not path.exists():
            log.warning("no drift calibration at %s; using Evidently's default threshold", path)
            return cls.default()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            {k: float(v) for k, v in data["thresholds"].items()},
            f"calibrated {data['computed_at'][:10]}: {data['weeks']} weeks, alarm rate "
            f"{data['achieved_alarm_rate']:.0%} at the {data['quantile']:.3f} quantile",
        )

    def for_forecast(self) -> float:
        # No history of served forecasts exists to calibrate on; a forecast is an estimate of
        # demand, so it is allowed to move as much as demand itself does.
        return self.by_column[LABEL]


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "rows": self.rows}


@dataclass(frozen=True)
class DriftFrames:
    """Inputs (horizon-1 features and demand) and forecasts, for reference and current windows."""

    reference_inputs: pl.DataFrame
    reference_forecasts: pl.DataFrame
    current_inputs: pl.DataFrame
    current_forecasts: pl.DataFrame
    reference_window: Window
    current_window: Window
    seasonal: bool  # False when the training window has no days at this time of year


@dataclass
class DriftResult:
    verdict: dict[str, Any]
    snapshot: Any = field(repr=False)  # evidently Snapshot

    @property
    def data_drift(self) -> bool:
        return bool(self.verdict["data_drift"]["drifted"])

    def save(self, directory: Path) -> tuple[Path, Path]:
        """Write the Evidently HTML report and the JSON verdict; returns both paths."""
        directory.mkdir(parents=True, exist_ok=True)
        html = directory / "drift_report.html"
        verdict = directory / "verdict.json"
        self.snapshot.save_html(str(html))
        verdict.write_text(json.dumps(self.verdict, indent=2) + "\n", encoding="utf-8")
        return html, verdict


def drift_frames(
    series: pl.DataFrame,
    model: DirectModel,
    served: pl.DataFrame,
    *,
    current_days: int = CURRENT_DAYS,
    season_days: int = SEASON_DAYS,
) -> DriftFrames:
    """Reference and current frames, the current window ending at the last half-hour of ``series``.

    ``served`` holds the forecasts the service logged for this model in the current window:
    forecast_origin, horizon, predicted_mw (one row per forecast).
    """
    if model.trained_on is None:
        raise ValueError("the model does not record its training window")
    first, last = span(series)
    current_start = last - timedelta(days=current_days)
    current_inputs = _inputs(series, _origins(current_start + PERIOD, last - PERIOD))

    trained_from, trained_to = model.trained_on
    # Reference origins lie inside the training window, before the current window, and every
    # forecast from them (up to the model's longest horizon) lands inside it too.
    reach = PERIOD * max(model.horizons)
    candidates = _origins(
        _ceil_to_period(max(trained_from, first) + WARMUP),
        min(trained_to, current_start) - reach,
    )
    if candidates.is_empty():
        raise ValueError("the training window is too short to draw a reference from")
    in_season = candidates.filter(
        season_distance(candidates, _midpoint(current_start, last)) <= season_days
    )
    seasonal = not in_season.is_empty()
    if not seasonal:
        log.warning("no training days within %d days of this time of year", season_days)
    origins = in_season if seasonal else candidates

    reference_inputs = _inputs(series, origins)
    sampled = origins.sample(min(REFERENCE_FORECAST_ORIGINS, origins.len()), seed=0).sort()
    features = build_features(series, sampled, model.horizons, with_label=False)
    assert_no_leakage(features)
    reference_forecasts = model.predict(features).select(pl.col("p50").alias(FORECAST))
    current_forecasts = served.select(pl.col("predicted_mw").alias(FORECAST))

    return DriftFrames(
        reference_inputs=reference_inputs,
        reference_forecasts=reference_forecasts,
        current_inputs=current_inputs,
        current_forecasts=current_forecasts,
        reference_window=Window(_first(origins), _last(origins) + reach, reference_inputs.height),
        current_window=Window(current_start, last, current_inputs.height),
        seasonal=seasonal,
    )


def check_drift(
    frames: DriftFrames,
    *,
    thresholds: Thresholds,
    model_name: str,
    model_version: str,
    trained_on: tuple[datetime, datetime],
    generated_at: datetime,
) -> DriftResult:
    """Run Evidently on the frames and turn its output into a machine-readable verdict.

    Inputs and forecasts have different rows (one per half-hour, one per forecast), so each
    side's frame stacks them with the other's columns empty; Evidently ignores missing values
    column by column, so every column is compared on its own rows only.
    """
    served = frames.current_forecasts.height
    landed = int(frames.current_inputs[LABEL].is_not_null().sum())
    features = list(DRIFT_FEATURES)
    metrics: list[Any] = [
        DataDriftPreset(
            columns=features,
            num_method=METHOD,
            per_column_threshold={c: thresholds.by_column[c] for c in features},
        )
    ]
    if landed >= 48:  # a day of half-hours
        metrics.append(
            ValueDrift(column=LABEL, method=METHOD, threshold=thresholds.by_column[LABEL])
        )
    if served >= MIN_ROWS:
        metrics.append(
            ValueDrift(column=FORECAST, method=METHOD, threshold=thresholds.for_forecast())
        )

    definition = DataDefinition(numerical_columns=list(COLUMNS))
    report = Report(metrics, metadata={"model": f"{model_name} {model_version}"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snapshot = report.run(
            Dataset.from_pandas(
                _stacked(frames.current_inputs, frames.current_forecasts),
                data_definition=definition,
            ),
            Dataset.from_pandas(
                _stacked(frames.reference_inputs, frames.reference_forecasts),
                data_definition=definition,
            ),
        )
    scores = column_scores(snapshot.dict())

    data = {column: scores[column] for column in DRIFT_FEATURES}
    verdict = {
        "generated_at": generated_at.isoformat(),
        "model": {
            "name": model_name,
            "version": model_version,
            "trained_from": trained_on[0].isoformat(),
            "trained_to": trained_on[1].isoformat(),
        },
        "method": METHOD,
        "thresholds": thresholds.source,
        "reference": {
            **frames.reference_window.to_dict(),
            "forecast_rows": frames.reference_forecasts.height,
            "seasonal": frames.seasonal,
            "season_days": SEASON_DAYS if frames.seasonal else None,
            "description": (
                f"the training window within {SEASON_DAYS} days of this time of year"
                if frames.seasonal
                else "the whole training window (no training days at this time of year)"
            ),
        },
        "current": {**frames.current_window.to_dict(), "forecast_rows": served},
        "data_drift": {
            "drifted": any(c["drifted"] for c in data.values()),
            "drifted_columns": [name for name, c in data.items() if c["drifted"]],
            "columns": data,
        },
        "prediction_drift": _single(scores, FORECAST, served, MIN_ROWS, "served forecasts"),
        "target_drift": _single(scores, LABEL, landed, 48, "landed half-hours"),
    }
    return DriftResult(verdict=verdict, snapshot=snapshot)


def column_scores(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-column drift score and decision from an Evidently snapshot's ``dict()``."""
    scores: dict[str, dict[str, Any]] = {}
    for metric in snapshot["metrics"]:
        config = metric.get("config", {})
        if config.get("type") != "evidently:metric_v2:ValueDrift":
            continue
        method = str(config.get("method", ""))
        if not method.lower().startswith(METHOD):
            # A p-value method would invert the comparison below; fail rather than misread it.
            raise ValueError(f"unexpected drift method {method!r} for {config.get('column')}")
        score = float(metric["value"])
        threshold = float(config["threshold"])
        scores[config["column"]] = {
            "score": round(score, 4),
            "threshold": round(threshold, 4),
            "drifted": score >= threshold,
        }
    return scores


# -- calibration --------------------------------------------------------------------------------


def normed_wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    """Evidently's score: Wasserstein distance over the reference's standard deviation."""
    return float(wasserstein_distance(reference, current) / max(float(np.std(reference)), 0.001))


def weekly_scores(
    series: pl.DataFrame,
    *,
    weeks: int = CALIBRATION_WEEKS,
    current_days: int = CURRENT_DAYS,
    season_days: int = SEASON_DAYS,
) -> pl.DataFrame:
    """Every past week's drift score per column, each week against its own seasonal reference.

    The reference for a week is everything before it (from the end of the warm-up) within
    ``season_days`` of its day of year: what a model trained just before that week would have
    been compared against. Weeks end where the live window does, at the latest half-hour.
    """
    first, last = span(series)
    rows = _inputs(series, _origins(_ceil_to_period(first + WARMUP), last - PERIOD))
    days = _day_of_year(rows["origin"])
    out: list[dict[str, Any]] = []
    end = last
    for _ in range(weeks):
        start = end - timedelta(days=current_days)
        current = rows.filter(pl.col("origin") > start, pl.col("origin") <= end)
        apart = (days - _day_of_year(pl.Series([_midpoint(start, end)]))[0]).abs()
        near = pl.min_horizontal(apart, 365 - apart) <= season_days
        reference = rows.filter(pl.col("origin") <= start - PERIOD, near)
        if reference.height < 48 * current_days or current.height < 48:
            break  # less than a week of seasonal history before this week
        record: dict[str, Any] = {"week_end": end, "reference_rows": reference.height}
        for column in INPUT_COLUMNS:
            ref = reference[column].drop_nulls().to_numpy()
            cur = current[column].drop_nulls().to_numpy()
            record[column] = normed_wasserstein(ref, cur) if len(ref) and len(cur) else None
        out.append(record)
        end = start
    return pl.DataFrame(out)


def calibrate(scores: pl.DataFrame, *, alarm_rate: float = ALARM_RATE) -> dict[str, Any]:
    """Per-column thresholds at one shared quantile of each column's weekly scores, chosen so
    that on ``scores`` at least one feature would have alerted in about ``alarm_rate`` of weeks.
    """
    if scores.height < 20:
        raise ValueError(f"need at least 20 weeks of history to calibrate, got {scores.height}")
    features = list(DRIFT_FEATURES)

    def rate_at(q: float) -> float:
        limits = {c: scores[c].quantile(q, "linear") for c in features}
        fired = pl.any_horizontal(pl.col(c) > limits[c] for c in features)
        return float(scores.select(fired.mean()).item())

    # With a hundred-odd weeks the achievable rates are coarse steps (each feature's extreme weeks
    # fall in different places), so take the quantile whose rate comes closest to the target, and
    # the lowest such quantile: the most sensitive thresholds with that historical rate.
    grid = [round(0.5 + k / 1000, 3) for k in range(500)]  # 0.500 .. 0.999
    quantile = min(grid, key=lambda q: (abs(rate_at(q) - alarm_rate), q))
    thresholds = {
        c: round(float(scores[c].quantile(quantile, "linear")), 4)  # type: ignore[arg-type]
        for c in INPUT_COLUMNS
    }
    return {
        "quantile": round(quantile, 4),
        "target_alarm_rate": alarm_rate,
        "achieved_alarm_rate": round(rate_at(quantile), 4),
        "default_alarm_rate": round(
            float(
                scores.select(
                    pl.any_horizontal(pl.col(c) > DEFAULT_THRESHOLD for c in features).mean()
                ).item()
            ),
            4,
        ),
        "thresholds": thresholds,
        "weeks": scores.height,
        "first_week_end": _first(scores["week_end"]).isoformat(),
        "last_week_end": _last(scores["week_end"]).isoformat(),
        "median_scores": {c: round(float(scores[c].median()), 4) for c in INPUT_COLUMNS},  # type: ignore[arg-type]
    }


# -- helpers ------------------------------------------------------------------------------------


def _single(
    scores: dict[str, dict[str, Any]], column: str, rows: int, needed: int, what: str
) -> dict[str, Any]:
    if column not in scores:
        return {"drifted": None, "status": f"insufficient data: {rows} {what}, need {needed}"}
    return {**scores[column], "status": "ok", "rows": rows}


def _origins(start: datetime, end: datetime) -> pl.Series:
    return pl.datetime_range(
        start, end, interval=ORIGIN_EVERY, time_zone=TIME_ZONE, eager=True
    ).alias("origin")


def _inputs(series: pl.DataFrame, origins: pl.Series) -> pl.DataFrame:
    """Horizon-1 rows: the data features at each origin, and demand in the next half-hour."""
    frame = build_features(series, origins, horizons=(1,))
    assert_no_leakage(frame)
    return frame.select("origin", *INPUT_COLUMNS)


def _stacked(inputs: pl.DataFrame, forecasts: pl.DataFrame) -> pd.DataFrame:
    """One pandas frame with input rows and forecast rows, each blank in the other's columns."""
    frame = pl.concat(
        [inputs.select(INPUT_COLUMNS), forecasts.select(FORECAST)], how="diagonal_relaxed"
    )
    # Column by column through numpy: no pyarrow needed, and nulls become NaN.
    return pd.DataFrame(
        {name: frame[name].cast(pl.Float64).to_numpy().astype(np.float64) for name in COLUMNS}
    )


def season_distance(origins: pl.Series, centre: datetime) -> pl.Series:
    """Days between each origin's day of year and the centre's, wrapping round the year end."""
    apart = (_day_of_year(origins) - _day_of_year(pl.Series([centre]))[0]).abs()
    return pl.select(pl.min_horizontal(apart, 365 - apart)).to_series()


def _day_of_year(timestamps: pl.Series) -> pl.Series:
    return timestamps.dt.convert_time_zone(LOCAL_TIME_ZONE).dt.ordinal_day().cast(pl.Int32)


def _midpoint(start: datetime, end: datetime) -> datetime:
    return start + (end - start) / 2


def _first(values: pl.Series) -> datetime:
    value = values.min()
    if not isinstance(value, datetime):
        raise ValueError("empty")
    return value


def _last(values: pl.Series) -> datetime:
    value = values.max()
    if not isinstance(value, datetime):
        raise ValueError("empty")
    return value


def _ceil_to_period(ts: datetime) -> datetime:
    floored = ts.replace(minute=ts.minute - ts.minute % 30, second=0, microsecond=0)
    return floored if floored == ts else floored + PERIOD
