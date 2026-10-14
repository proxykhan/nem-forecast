"""Champion/challenger retraining: a new model is promoted only if it beats the served one.

**The holdout.** The last ``HOLDOUT_DAYS`` of data, but never earlier than the end of the
incumbent's training data: a holdout the incumbent had trained on would hand it an in-sample
advantage. If the incumbent was trained so recently that fewer than ``MIN_HOLDOUT_DAYS`` remain,
there is nothing fair to compare on, and retraining is skipped. (It would also be pointless:
the candidate would see almost exactly the incumbent's data.)

**The comparison.** The candidate (the incumbent's model type, with the currently tuned
parameters) is trained on everything before the holdout. Both models then forecast the same
holdout rows from every half-hourly origin, and their MAE is compared within each horizon group
(1 | 2-3 | 4-6 | 7-12 | 13-24 | 25-48 half-hours ahead).

**The gate.** The candidate wins only if its MAE is lower in *every* horizon group. Better on
average while worse at one horizon is not a win: a trading desk using the 30-minute forecast
does not care that the day-ahead one improved.

**What ships.** After the gate passes, the same recipe is refit on all data up to now and that
model is promoted, exactly as the walk-forward backtest refits before each evaluation period.
The promoted artifact is downloaded back from the Hub and must reproduce the in-memory model's
forecasts before anything points at it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import polars as pl

from nem_forecast.features.build import KEYS, LABEL, assert_no_leakage, build_features
from nem_forecast.features.series import PERIOD, TIME_ZONE, span
from nem_forecast.models.base import HORIZON_GROUPS, DirectModel, horizon_group_label
from nem_forecast.schemas import NEM_TZ

log = logging.getLogger(__name__)

HOLDOUT_DAYS = 14  # two weekly cycles: about 640 origins and 30,000 forecasts
MIN_HOLDOUT_DAYS = 7  # one weekly cycle, the least that represents weekdays and weekends
ORIGIN_EVERY = timedelta(minutes=30)


class HoldoutTooShort(ValueError):
    """The incumbent was trained too recently for a fair holdout."""


@dataclass(frozen=True)
class GroupResult:
    group: str
    n: int
    candidate_mae: float
    incumbent_mae: float

    @property
    def improved(self) -> bool:
        return self.candidate_mae < self.incumbent_mae


@dataclass(frozen=True)
class Comparison:
    holdout_start: datetime
    holdout_end: datetime
    groups: tuple[GroupResult, ...]

    @property
    def wins(self) -> bool:
        return bool(self.groups) and all(g.improved for g in self.groups)

    @property
    def regressed(self) -> list[str]:
        return [g.group for g in self.groups if not g.improved]

    def to_dict(self) -> dict[str, Any]:
        return {
            "holdout_start": self.holdout_start.isoformat(),
            "holdout_end": self.holdout_end.isoformat(),
            "wins": self.wins,
            "regressed": self.regressed,
            "groups": [asdict(g) | {"improved": g.improved} for g in self.groups],
        }

    def markdown(self, *, candidate: str = "Candidate", incumbent: str = "Incumbent") -> str:
        """A table of MAE by horizon group, for the model card and issues."""
        lines = [
            f"Holdout {self.holdout_start.astimezone(NEM_TZ):%Y-%m-%d %H:%M} to "
            f"{self.holdout_end.astimezone(NEM_TZ):%Y-%m-%d %H:%M} (market time), forecasts from "
            "every half-hour, MAE in MW:",
            "",
            f"| Horizons | Forecasts | {candidate} | {incumbent} | Change |",
            "|---|--:|--:|--:|--:|",
        ]
        for g in self.groups:
            change = (g.candidate_mae - g.incumbent_mae) / g.incumbent_mae * 100
            mark = "" if g.improved else " **regressed**"
            lines.append(
                f"| {g.group} | {g.n:,} | {g.candidate_mae:,.1f} | {g.incumbent_mae:,.1f} | "
                f"{change:+.1f}%{mark} |"
            )
        return "\n".join(lines)


def plan_holdout(
    series_end: datetime,
    incumbent_trained_to: datetime | None,
    *,
    holdout_days: int = HOLDOUT_DAYS,
    min_days: int = MIN_HOLDOUT_DAYS,
) -> tuple[datetime, datetime]:
    """The holdout ``(start, end]``: recent data the incumbent never trained on."""
    start = series_end - timedelta(days=holdout_days)
    if incumbent_trained_to is not None and incumbent_trained_to > start:
        start = incumbent_trained_to
    if series_end - start < timedelta(days=min_days):
        raise HoldoutTooShort(
            f"the incumbent was trained on data up to "
            f"{incumbent_trained_to:%Y-%m-%d %H:%M} UTC, leaving "
            f"{(series_end - start).total_seconds() / 86400:.1f} days it has not seen; "
            f"a fair holdout needs at least {min_days}"
        )
    return start, series_end


def compare(
    series: pl.DataFrame,
    incumbent: DirectModel,
    candidate: DirectModel,
    *,
    holdout: tuple[datetime, datetime],
) -> Comparison:
    """Fit ``candidate`` on data before the holdout; score both models on identical rows."""
    start, end = holdout
    candidate.fit(series.filter(pl.col("period_end") <= start))
    origins = pl.datetime_range(
        start, end - PERIOD, interval=ORIGIN_EVERY, time_zone=TIME_ZONE, eager=True
    )
    frame = build_features(series, origins, incumbent.horizons)
    assert_no_leakage(frame)
    frame = frame.filter(pl.col(LABEL).is_not_null(), pl.col("target") <= end)
    if frame.is_empty():
        raise ValueError("the holdout has no landed actuals")
    features = frame.drop(LABEL)
    scored = (
        frame.select(*KEYS, LABEL)
        .join(_p50(candidate, features, "candidate"), on=list(KEYS))
        .join(_p50(incumbent, features, "incumbent"), on=list(KEYS))
    )

    groups = []
    for group in HORIZON_GROUPS:
        rows = scored.filter(pl.col("horizon").is_in(group))
        if rows.is_empty():
            continue
        groups.append(
            GroupResult(
                group=horizon_group_label(group),
                n=rows.height,
                candidate_mae=_mae(rows, "candidate"),
                incumbent_mae=_mae(rows, "incumbent"),
            )
        )
    comparison = Comparison(start, end, tuple(groups))
    log.info(
        "candidate %s the incumbent on the holdout%s",
        "beats" if comparison.wins else "does not beat",
        "" if comparison.wins else f" (regressed: {', '.join(comparison.regressed)})",
    )
    return comparison


def holdout_features(
    series: pl.DataFrame, comparison: Comparison, model: DirectModel
) -> pl.DataFrame:
    """Feature rows from the holdout's first day, used to check a downloaded model round-trips."""
    first, _ = span(series)
    start = max(comparison.holdout_start, first)
    origins = pl.datetime_range(
        start, start + timedelta(days=1), interval=ORIGIN_EVERY, time_zone=TIME_ZONE, eager=True
    )
    return build_features(series, origins, model.horizons, with_label=False)


def _p50(model: DirectModel, features: pl.DataFrame, name: str) -> pl.DataFrame:
    return model.predict(features).select(*KEYS, pl.col("p50").alias(name))


def _mae(rows: pl.DataFrame, column: str) -> float:
    value = (rows[column] - rows[LABEL]).abs().mean()
    return round(float(value), 3)  # type: ignore[arg-type]
