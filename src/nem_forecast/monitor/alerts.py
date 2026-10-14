"""Alert thresholds, and the GitHub issue a breach opens.

Three checks, each with a threshold chosen and justified below rather than left to a default.
A check without enough data to decide reports ``insufficient data`` and does not fire: an alert
raised on a handful of points teaches people to ignore alerts.

A breach opens a GitHub issue labelled for its check. While that issue is open, later breaches
of the same check comment on it instead of opening another, so a week of drift is one thread,
not seven issues.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Any, Protocol

import httpx
import polars as pl
from sqlalchemy import Engine, text

from nem_forecast.backtest.metrics import ALL_BUCKETS, ALL_HORIZONS
from nem_forecast.schemas import NEM_TZ, Region

log = logging.getLogger(__name__)

# -- Thresholds ---------------------------------------------------------------------------------

# Realised error is judged over 7 days: one full weekly cycle, so the mix of weekdays and weekends
# matches the backtest's, and long enough that one stormy afternoon cannot decide it alone.
ERROR_WINDOW = timedelta(days=7)
# At least three days of scored forecasts (48 origins x 48 horizons a day): fewer, and a single
# unusual day would be most of the evidence.
MIN_SCORED = 3 * 48 * 48
# The bar is the seasonal naive baseline's backtest MAE ("same half-hour yesterday"), not the
# model's own backtest MAE. A model that does worse than copying yesterday has negative value:
# anyone could replace it with one line of SQL. Its own backtest MAE is a year-long average, and a
# single hot or cloudy week legitimately runs well above it, so that bar would fire on weather.
# Seasonal naive's error barely depends on horizon (every horizon reads the same lag), so the
# comparison holds whatever mix of horizons has landed.
ERROR_BASELINE = "seasonal_naive"

# Drift fires when any data feature's drift score (normed Wasserstein distance against the
# seasonal reference, see drift.py) reaches its Evidently threshold. The thresholds are calibrated
# on two years of weekly history so that this would have fired in 1 week in 20; Evidently's
# default of 0.1 would have fired in every one of those weeks. Features, not forecasts or demand:
# drift in the inputs is the early warning, before errors land; the other two are reported
# alongside for diagnosis.

# Volume fires when no forecast at all was logged for 6 hours. The scheduled client asks every
# half-hour, so 6 hours is 12 missed requests in a row: longer than GitHub's scheduling delays
# (which can skip an hour or two of cron runs), a Render cold start, or a redeploy, so it means
# the service or its client is down, not slow.
VOLUME_GAP = timedelta(hours=6)
# The monitor runs daily; looking back 30 hours lets a gap spanning two runs still be seen whole.
VOLUME_LOOKBACK = timedelta(hours=30)

OK = "ok"
BREACHED = "breached"
INSUFFICIENT = "insufficient data"


@dataclass(frozen=True)
class Check:
    name: str
    title: str
    status: str  # ok | breached | insufficient data
    value: float | None
    threshold: float | None
    detail: str
    retrain: bool  # whether a breach is something retraining can fix

    @property
    def breached(self) -> bool:
        return self.status == BREACHED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def baseline_mae(backtest: pl.DataFrame, model: str = ERROR_BASELINE) -> float | None:
    """Overall backtest MAE of ``model``, from :func:`~.registry.latest_backtest` output."""
    row = backtest.filter(
        pl.col("model") == model,
        pl.col("metric") == "mae",
        pl.col("horizon") == ALL_HORIZONS,
        pl.col("tod_bucket") == ALL_BUCKETS,
    )
    return float(row["value"][0]) if row.height else None


def check_realised_error(scores: pl.DataFrame, baseline: float | None) -> Check:
    """Rolling 7-day MAE of served forecasts against seasonal naive's backtest MAE."""
    title = "Realised 7-day MAE above the seasonal naive baseline"
    if baseline is None:
        return Check(
            "realised_error",
            title,
            INSUFFICIENT,
            None,
            None,
            f"no recorded backtest for {ERROR_BASELINE}; run the nightly backtest",
            retrain=True,
        )
    n = scores.height
    if n < MIN_SCORED:
        return Check(
            "realised_error",
            title,
            INSUFFICIENT,
            None,
            baseline,
            f"{n:,} forecasts scored in the last {ERROR_WINDOW.days} days; need {MIN_SCORED:,}",
            retrain=True,
        )
    mae = float((scores["predicted_mw"] - scores["actual_mw"]).abs().mean())  # type: ignore[arg-type]
    status = BREACHED if mae > baseline else OK
    return Check(
        "realised_error",
        title,
        status,
        round(mae, 1),
        round(baseline, 1),
        f"MAE {mae:,.0f} MW over {n:,} forecasts in the last {ERROR_WINDOW.days} days, against "
        f"{baseline:,.0f} MW for {ERROR_BASELINE} in the backtest",
        retrain=True,
    )


def check_feature_drift(verdict: dict[str, Any]) -> Check:
    """Any data feature's drift score at or above Evidently's threshold."""
    columns: dict[str, dict[str, Any]] = verdict["data_drift"]["columns"]
    worst = max(columns, key=lambda name: columns[name]["score"])
    drifted = verdict["data_drift"]["drifted_columns"]
    detail = (
        f"drifted: {', '.join(drifted)}" if drifted else "no feature reached the threshold"
    ) + f"; largest score {columns[worst]['score']:.3f} ({worst})"
    return Check(
        "feature_drift",
        "Feature drift past the Evidently threshold",
        BREACHED if drifted else OK,
        columns[worst]["score"],
        columns[worst]["threshold"],
        detail,
        retrain=True,
    )


def load_request_times(
    engine: Engine, region: Region, *, start: datetime, end: datetime
) -> list[datetime]:
    """When each logged forecast request in ``(start, end]`` was served."""
    query = """
        SELECT min(served_at) FROM predictions
        WHERE region = :region AND served_at > :start AND served_at <= :end
        GROUP BY request_id ORDER BY 1
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query), {"region": region.value, "start": start, "end": end}).all()
    return [row[0] for row in rows]


def check_volume(served_at: Sequence[datetime], *, now: datetime) -> Check:
    """The longest stretch in the lookback window with no forecast logged at all."""
    start = now - VOLUME_LOOKBACK
    moments = [start, *sorted(t for t in served_at if start < t <= now), now]
    gaps = [(later - earlier, earlier, later) for earlier, later in pairwise(moments)]
    longest, gap_start, gap_end = max(gaps)
    hours = longest.total_seconds() / 3600
    lookback_hours = VOLUME_LOOKBACK.total_seconds() / 3600
    status = BREACHED if longest >= VOLUME_GAP else OK
    return Check(
        "prediction_volume",
        "No forecasts logged for 6 hours",
        status,
        round(hours, 2),
        VOLUME_GAP.total_seconds() / 3600,
        f"{len(moments) - 2} forecasts served in the last {lookback_hours:.0f} hours; longest "
        f"gap {hours:.1f} h, {_market(gap_start)} to {_market(gap_end)}",
        retrain=False,  # an outage is not a model problem
    )


# -- GitHub issues ------------------------------------------------------------------------------


class IssueTracker(Protocol):
    def report(self, *, title: str, label: str, body: str) -> str:
        """Comment on the open issue labelled ``label``, or open a new one; returns its URL."""
        ...


class GitHubIssues:
    """Opens (or comments on) one issue per alert label through the GitHub REST API."""

    API = "https://api.github.com"
    MONITORING_LABEL = "monitoring"

    def __init__(self, repository: str, token: str, *, client: httpx.Client | None = None) -> None:
        self.repository = repository
        self._client = client or httpx.Client(timeout=30)
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def report(self, *, title: str, label: str, body: str) -> str:
        """Comment on the open issue labelled ``label``, or open a new one; returns its URL."""
        for name in (self.MONITORING_LABEL, label):
            self._ensure_label(name)
        found = self._request(
            "GET",
            f"/repos/{self.repository}/issues",
            params={"state": "open", "labels": label, "per_page": 1},
        )
        if found:
            issue = found[0]
            self._request(
                "POST",
                f"/repos/{self.repository}/issues/{issue['number']}/comments",
                json={"body": body},
            )
            log.info("commented on %s", issue["html_url"])
            return str(issue["html_url"])
        created = self._request(
            "POST",
            f"/repos/{self.repository}/issues",
            json={"title": title, "body": body, "labels": [self.MONITORING_LABEL, label]},
        )
        log.info("opened %s", created["html_url"])
        return str(created["html_url"])

    def _ensure_label(self, name: str) -> None:
        response = self._client.post(
            f"{self.API}/repos/{self.repository}/labels",
            headers=self._headers,
            json={"name": name, "color": "d93f0b" if name != self.MONITORING_LABEL else "5319e7"},
        )
        if response.status_code not in (201, 422):  # 422: it already exists
            response.raise_for_status()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._client.request(
            method, f"{self.API}{path}", headers=self._headers, **kwargs
        )
        response.raise_for_status()
        return response.json()


def alert_body(
    check: Check,
    *,
    verdict: dict[str, Any],
    report_url: str | None,
    run_url: str | None,
) -> str:
    """Issue text for a breached check, with the drift verdict and links to the report.

    GitHub's API cannot attach files to an issue, so the Evidently report is linked where it is
    published (GitHub Pages) and where the run keeps a copy (its artifacts).
    """
    checked = datetime.fromisoformat(verdict["generated_at"])
    lines = [
        f"**{check.title}** (`{check.name}`), checked {_market(checked)}.",
        "",
        "| | |",
        "|---|---|",
        f"| Measured | {check.value} |",
        f"| Threshold | {check.threshold} |",
        "",
        check.detail + ".",
        "",
    ]
    if report_url:
        lines.append(f"- Evidently drift report: {report_url}")
    if run_url:
        lines.append(f"- Monitoring run, with the report and verdict as artifacts: {run_url}")
    if check.retrain:
        lines.append("- Retraining was triggered; its result is reported separately.")
    lines += [
        "",
        "<details><summary>Drift verdict (JSON)</summary>",
        "",
        "```json",
        json.dumps(verdict, indent=2),
        "```",
        "",
        "</details>",
    ]
    return "\n".join(lines)


def _market(ts: datetime) -> str:
    return f"{ts.astimezone(NEM_TZ):%Y-%m-%d %H:%M} AEST"
