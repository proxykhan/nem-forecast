"""The daily monitoring job and the retraining job, end to end.

Both take their collaborators as arguments (database engine, the served model, a Hub publisher,
an issue tracker), so the CLI wires in the real ones and tests wire in fakes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl
from sqlalchemy import Engine, text

from nem_forecast.backtest.metrics import ALL_BUCKETS
from nem_forecast.features.series import load_half_hourly, span
from nem_forecast.models.base import METADATA_FILE, DirectModel
from nem_forecast.models.registry import (
    CARD_FILES,
    PromotionError,
    latest_backtest,
    release_tag,
    render_model_card,
)
from nem_forecast.monitor import alerts, retrain, scoring
from nem_forecast.monitor.alerts import Check, IssueTracker
from nem_forecast.monitor.drift import CURRENT_DAYS, Thresholds, check_drift, drift_frames
from nem_forecast.monitor.site import write_site
from nem_forecast.schemas import Region
from nem_forecast.serve.model_store import LoadedModel

log = logging.getLogger(__name__)

REALISED_DAYS = 30  # the dashboard's window for realised error by horizon


# -- Monitoring ---------------------------------------------------------------------------------


@dataclass
class MonitorOutcome:
    status: dict[str, Any]
    checks: list[Check]
    issues: list[str] = field(default_factory=list)

    @property
    def breached(self) -> list[Check]:
        return [c for c in self.checks if c.breached]

    @property
    def retrain(self) -> bool:
        """Whether a breach is one that retraining could fix (error or drift, not an outage)."""
        return any(c.retrain for c in self.breached)


def run_monitor(
    engine: Engine,
    served: LoadedModel,
    *,
    region: Region,
    now: datetime,
    site_dir: Path,
    report_url: str | None = None,
    run_url: str | None = None,
    thresholds: Thresholds,
    issues: IssueTracker | None = None,
    git_sha: str | None = None,
    rescore_days: int = scoring.RESCORE_DAYS,
) -> MonitorOutcome:
    """Score, check drift, evaluate the alerts, publish the site, record the run, raise issues."""
    trained_on = served.model.trained_on
    if trained_on is None:
        raise ValueError("the served model does not record its training window")
    scored = scoring.score_predictions(engine, region, now=now, rescore_days=rescore_days)

    series = load_half_hourly(engine, region)
    last = span(series)[1]
    logged = scoring.load_served(
        engine,
        region,
        model_version=served.version,
        start=last - timedelta(days=CURRENT_DAYS),
        end=last,
    )
    drift = check_drift(
        drift_frames(series, served.model, logged),
        thresholds=thresholds,
        model_name=served.name,
        model_version=served.version,
        trained_on=trained_on,
        generated_at=now,
    )
    report_html, _ = drift.save(site_dir)

    backtest = latest_backtest(engine, region)
    recent = scoring.load_scores(engine, region, start=now - alerts.ERROR_WINDOW, end=now)
    month = scoring.load_scores(engine, region, start=now - timedelta(days=REALISED_DAYS), end=now)
    checks = [
        alerts.check_realised_error(recent, alerts.baseline_mae(backtest)),
        alerts.check_feature_drift(drift.verdict),
        alerts.check_volume(
            alerts.load_request_times(engine, region, start=now - alerts.VOLUME_LOOKBACK, end=now),
            now=now,
        ),
    ]
    status = {
        "generated_at": now.isoformat(),
        "region": region.value,
        "model": {
            "name": served.name,
            "version": served.version,
            "trained_to": trained_on[1].isoformat(),
        },
        "window_days": REALISED_DAYS,
        "scoring": {
            "scored": scored.scored,
            "rescored": scored.rescored,
            "unchanged": scored.unchanged,
            "pending": scored.pending,
        },
        "realised_vs_backtest": _comparison(month, backtest, served.name),
        "realised_by_horizon": _by_horizon(month),
        "checks": [c.to_dict() for c in checks],
        "drift": drift.verdict,
        "report_url": report_url,
    }
    write_site(site_dir, status=status, checks=checks, report_html=report_html)
    _record_monitor_run(engine, status, region=region, git_sha=git_sha)

    outcome = MonitorOutcome(status=status, checks=checks)
    if issues is not None:
        for check in outcome.breached:
            outcome.issues.append(
                issues.report(
                    title=f"Monitoring alert: {check.title}",
                    label=f"alert:{check.name}",
                    body=alerts.alert_body(
                        check, verdict=drift.verdict, report_url=report_url, run_url=run_url
                    ),
                )
            )
    return outcome


def _comparison(scores: pl.DataFrame, backtest: pl.DataFrame, model: str) -> list[dict[str, Any]]:
    if scores.is_empty() or backtest.filter(pl.col("model") == model).is_empty():
        return []
    rows = scoring.compare_with_backtest(scores, backtest, model=model)
    return [
        {
            **row,
            "realised_mae": round(row["realised_mae"], 1),
            "backtest_mae": round(row["backtest_mae"], 1),
            "ratio": round(row["ratio"], 3),
        }
        for row in rows.iter_rows(named=True)
    ]


def _by_horizon(scores: pl.DataFrame) -> list[dict[str, Any]]:
    if scores.is_empty():
        return []
    tidy = scoring.realised_metrics(scores).filter(
        pl.col("metric") == "mae", pl.col("horizon") > 0, pl.col("tod_bucket") == ALL_BUCKETS
    )
    return [
        {"horizon": int(row["horizon"]), "mae": round(row["value"], 1), "n": int(row["n_obs"])}
        for row in tidy.sort("horizon").iter_rows(named=True)
    ]


def _record_monitor_run(
    engine: Engine, status: dict[str, Any], *, region: Region, git_sha: str | None
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO monitor_runs (region, run_at, model_version, drift, alerts, realised, "
                "report_url, git_sha) VALUES (:region, :run_at, :version, CAST(:drift AS JSONB), "
                "CAST(:alerts AS JSONB), CAST(:realised AS JSONB), :report_url, :sha)"
            ),
            {
                "region": region.value,
                "run_at": status["generated_at"],
                "version": status["model"]["version"],
                "drift": json.dumps(status["drift"]),
                "alerts": json.dumps(status["checks"]),
                "realised": json.dumps(
                    {
                        "window_days": status["window_days"],
                        "by_group": status["realised_vs_backtest"],
                        "by_horizon": status["realised_by_horizon"],
                        "scoring": status["scoring"],
                    }
                ),
                "report_url": status["report_url"],
                "sha": git_sha,
            },
        )


# -- Retraining ---------------------------------------------------------------------------------


class Publisher(Protocol):
    repo_id: str

    def publish(self, directory: Path, *, version: str, release: str) -> str:
        """Upload and tag a saved model; returns its URL."""
        ...

    def fetch(self, release: str) -> DirectModel:
        """Download a published model back from the registry."""
        ...


@dataclass
class RetrainOutcome:
    decision: str  # promoted | rejected | skipped | dry-run
    reason: str
    comparison: retrain.Comparison | None = None
    release: str | None = None
    url: str | None = None
    issue: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "release": self.release,
            "url": self.url,
            "issue": self.issue,
            "comparison": self.comparison.to_dict() if self.comparison else None,
        }


def run_retrain(
    engine: Engine,
    incumbent: LoadedModel,
    make_candidate: Callable[[], DirectModel],
    *,
    region: Region,
    started: datetime,
    trigger: str,
    output_dir: Path,
    publisher: Publisher | None,
    issues: IssueTracker | None = None,
    git_sha: str | None = None,
    holdout_days: int = retrain.HOLDOUT_DAYS,
) -> RetrainOutcome:
    """Retrain, compare against the incumbent on a shared holdout, and promote only on a win.

    ``publisher=None`` is a dry run: the comparison is made and reported, nothing is recorded.
    """
    series = load_half_hourly(engine, region)
    try:
        holdout = retrain.plan_holdout(
            span(series)[1], incumbent.trained_to, holdout_days=holdout_days
        )
    except retrain.HoldoutTooShort as exc:
        outcome = RetrainOutcome("skipped", str(exc))
        if publisher is not None:
            _record_retrain(engine, outcome, incumbent, region, started, trigger, git_sha)
        return outcome

    candidate = make_candidate()
    comparison = retrain.compare(series, incumbent.model, candidate, holdout=holdout)
    if not comparison.wins:
        outcome = RetrainOutcome(
            "rejected",
            f"the candidate did not beat {incumbent.version} on: {', '.join(comparison.regressed)}",
            comparison,
        )
        if publisher is not None:
            _record_retrain(engine, outcome, incumbent, region, started, trigger, git_sha)
            if issues is not None:
                outcome.issue = issues.report(
                    title="Retraining rejected: the candidate did not beat the served model",
                    label="retrain:rejected",
                    body=_rejection_body(outcome, comparison, incumbent, trigger),
                )
        return outcome
    if publisher is None:
        return RetrainOutcome("dry-run", "the candidate wins; dry run, not promoted", comparison)
    if not git_sha or git_sha.endswith("-dirty"):
        raise PromotionError("refusing to promote from uncommitted code: commit first")

    # The gate passed: refit the same recipe on everything and ship that.
    final = make_candidate()
    final.fit(series)
    trained_to = span(series)[1]
    release = release_tag(git_sha, trained_to)
    directory = output_dir / f"{region.value.lower()}-{final.name}-{release}"
    final.save(directory, version=git_sha, release=release)
    metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    gate = "\n".join(
        [
            "## Promotion gate",
            "",
            f"Retrained ({trigger}) and promoted only after beating the previous model, "
            f"`{incumbent.version}`, in every horizon group on a holdout neither had trained on:",
            "",
            comparison.markdown(candidate="This model's recipe", incumbent=incumbent.version),
        ]
    )
    card = render_model_card(
        metadata,
        latest_backtest(engine, region),
        region=region,
        repo_id=publisher.repo_id,
        promotion_gate=gate,
    )
    for name in CARD_FILES:
        (directory / name).write_text(card, encoding="utf-8", newline="\n")

    url = publisher.publish(directory, version=git_sha, release=release)
    _check_round_trip(final, publisher.fetch(release), series, comparison)
    outcome = RetrainOutcome(
        "promoted",
        f"the candidate beat {incumbent.version} in every horizon group",
        comparison,
        release=release,
        url=url,
    )
    _record_retrain(engine, outcome, incumbent, region, started, trigger, git_sha)
    _record_model_run(engine, metadata, region, git_sha, started, release)
    return outcome


def _check_round_trip(
    final: DirectModel, fetched: DirectModel, series: pl.DataFrame, comparison: retrain.Comparison
) -> None:
    features = retrain.holdout_features(series, comparison, final)
    expected = final.predict(features).sort("origin", "horizon")
    actual = fetched.predict(features).sort("origin", "horizon")
    for column in ("p10", "p50", "p90"):
        if not np.allclose(expected[column].to_numpy(), actual[column].to_numpy(), atol=1e-6):
            raise PromotionError(
                "the model downloaded from the Hub does not reproduce the trained model's "
                f"{column}; render.yaml was not changed, so nothing serves it"
            )


def _rejection_body(
    outcome: RetrainOutcome,
    comparison: retrain.Comparison,
    incumbent: LoadedModel,
    trigger: str,
) -> str:
    return "\n".join(
        [
            f"Retraining ({trigger}) produced a candidate that did **not** beat the served model "
            f"`{incumbent.name}` `{incumbent.version}` in every horizon group, so it was not "
            "promoted. The served model is unchanged.",
            "",
            f"Regressed: **{', '.join(comparison.regressed)}**.",
            "",
            comparison.markdown(candidate="Candidate", incumbent=incumbent.version),
            "",
            "The candidate is the served model's type with the current tuned parameters, trained "
            "on all data before the holdout. A win must hold in every group: a better day-ahead "
            "forecast does not excuse a worse 30-minute one.",
        ]
    )


def _record_retrain(
    engine: Engine,
    outcome: RetrainOutcome,
    incumbent: LoadedModel,
    region: Region,
    started: datetime,
    trigger: str,
    git_sha: str | None,
) -> None:
    comparison = outcome.comparison
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO retrain_runs (region, started_at, trigger, model_name, "
                "incumbent_version, release, holdout_start, holdout_end, decision, reason, "
                "comparison, git_sha) VALUES (:region, :started, :trigger, :name, :incumbent, "
                ":release, :holdout_start, :holdout_end, :decision, :reason, "
                "CAST(:comparison AS JSONB), :sha)"
            ),
            {
                "region": region.value,
                "started": started,
                "trigger": trigger,
                "name": incumbent.name,
                "incumbent": incumbent.version,
                "release": outcome.release,
                "holdout_start": comparison.holdout_start if comparison else None,
                "holdout_end": comparison.holdout_end if comparison else None,
                "decision": outcome.decision,
                "reason": outcome.reason,
                "comparison": json.dumps(comparison.to_dict() if comparison else {}),
                "sha": git_sha,
            },
        )


def _record_model_run(
    engine: Engine,
    metadata: dict[str, Any],
    region: Region,
    git_sha: str,
    started: datetime,
    release: str,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO model_runs (model_name, model_kind, run_type, region, git_sha, "
                "hf_model_version, feature_version, params, train_start, train_end, status, "
                "started_at, finished_at) VALUES (:name, 'model', 'retrain', :region, :sha, "
                ":release, :feature_version, CAST(:params AS JSONB), :start, :end, 'succeeded', "
                ":started, now())"
            ),
            {
                "name": metadata["name"],
                "region": region.value,
                "sha": git_sha,
                "release": release,
                "feature_version": metadata["feature_version"],
                "params": json.dumps(metadata["state"].get("params", {})),
                "start": metadata["trained_on"][0],
                "end": metadata["trained_on"][1],
                "started": started,
            },
        )
