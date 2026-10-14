"""The static site the daily monitor publishes to GitHub Pages.

* ``index.html``: today's checks, realised error against the backtest, drift by column
* ``drift_report.html``: the full Evidently report
* ``verdict.json``: the machine-readable drift verdict
* ``status.json``: everything on the index page, as JSON

Only the latest run is published; earlier reports stay attached to their workflow runs.
"""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from nem_forecast.monitor.alerts import BREACHED, INSUFFICIENT, Check
from nem_forecast.schemas import NEM_TZ

REPO_URL = "https://github.com/proxykhan/nem-forecast"

_STYLE = """
:root { --bg: #fbfbfa; --fg: #1d1d1b; --muted: #6b6b66; --line: #e3e2de; --card: #ffffff;
        --ok: #1a7f37; --bad: #c4321c; --warn: #9a6700; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #151514; --fg: #ecebe7; --muted: #a09f99; --line: #2f2e2b; --card: #1d1d1b;
          --ok: #4ac26b; --bad: #ff7b63; --warn: #d4a72c; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 32px 0 8px; }
p.meta { color: var(--muted); margin: 0 0 24px; }
section { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
          padding: 16px 18px; margin-top: 16px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line);
         white-space: nowrap; }
td.num, th.num { text-align: right; }
.ok { color: var(--ok); font-weight: 600; } .breached { color: var(--bad); font-weight: 600; }
.insufficient { color: var(--warn); font-weight: 600; }
a { color: inherit; }
ul.links { padding-left: 18px; }
"""


def write_site(
    directory: Path,
    *,
    status: dict[str, Any],
    checks: list[Check],
    report_html: Path,
) -> Path:
    """Write the site into ``directory`` (the Evidently report is moved in); returns index.html."""
    directory.mkdir(parents=True, exist_ok=True)
    if report_html.resolve() != (directory / "drift_report.html").resolve():
        report_html.replace(directory / "drift_report.html")
    (directory / "verdict.json").write_text(
        json.dumps(status["drift"], indent=2) + "\n", encoding="utf-8"
    )
    (directory / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    index = directory / "index.html"
    index.write_text(render_index(status, checks), encoding="utf-8")
    (directory / ".nojekyll").write_text("", encoding="utf-8")
    return index


def render_index(status: dict[str, Any], checks: list[Check]) -> str:
    generated = datetime.fromisoformat(status["generated_at"]).astimezone(NEM_TZ)
    model = status["model"]
    drift = status["drift"]
    breached = [c for c in checks if c.status == BREACHED]
    headline = (
        f'<span class="breached">{len(breached)} alert{"s" if len(breached) != 1 else ""}</span>'
        if breached
        else '<span class="ok">All checks passing</span>'
    )

    check_rows = "".join(
        f"<tr><td>{escape(c.title)}</td><td class='{_css(c.status)}'>{escape(c.status)}</td>"
        f"<td class='num'>{_num(c.value)}</td><td class='num'>{_num(c.threshold)}</td>"
        f"<td>{escape(c.detail)}</td></tr>"
        for c in checks
    )
    compare_rows = (
        "".join(
            f"<tr><td>{escape(r['group'])}</td><td class='num'>{r['n']:,}</td>"
            f"<td class='num'>{r['realised_mae']:,.0f}</td>"
            f"<td class='num'>{r['backtest_mae']:,.0f}</td>"
            f"<td class='num'>{r['ratio']:.2f}</td></tr>"
            for r in status["realised_vs_backtest"]
        )
        or "<tr><td colspan='5'>No forecasts scored in the window yet.</td></tr>"
    )

    drift_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>data</td><td class='num'>{c['score']:.3f}</td>"
        f"<td class='num'>{c['threshold']:.3f}</td>"
        f"<td class='{_css(BREACHED if c['drifted'] else 'ok')}'>"
        f"{'drift' if c['drifted'] else 'no drift'}</td></tr>"
        for name, c in drift["data_drift"]["columns"].items()
    )
    for label, key in (("forecast", "prediction_drift"), ("demand", "target_drift")):
        entry = drift[key]
        if entry["drifted"] is None:
            drift_rows += (
                f"<tr><td>{label}</td><td>{key.split('_')[0]}</td><td class='num'>-</td>"
                f"<td class='num'>-</td>"
                f"<td class='insufficient'>{escape(entry['status'])}</td></tr>"
            )
        else:
            drift_rows += (
                f"<tr><td>{label}</td><td>{key.split('_')[0]}</td>"
                f"<td class='num'>{entry['score']:.3f}</td>"
                f"<td class='num'>{entry['threshold']:.3f}</td>"
                f"<td class='{_css(BREACHED if entry['drifted'] else 'ok')}'>"
                f"{'drift' if entry['drifted'] else 'no drift'}</td></tr>"
            )

    window = status["window_days"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nem-forecast monitoring</title>
<style>{_STYLE}</style>
</head>
<body>
<main>
<h1>nem-forecast monitoring: {escape(status["region"])}</h1>
<p class="meta">{headline} &middot; checked {generated:%Y-%m-%d %H:%M} AEST &middot; model
<code>{escape(model["name"])}</code> <code>{escape(model["version"])}</code>, trained to
{datetime.fromisoformat(model["trained_to"]).astimezone(NEM_TZ):%Y-%m-%d %H:%M} AEST</p>

<h2>Checks</h2>
<section><table>
<tr><th>Check</th><th>Status</th><th class="num">Value</th><th class="num">Threshold</th>
<th>Detail</th></tr>
{check_rows}
</table></section>

<h2>Realised error against the backtest, last {window} days</h2>
<section>
<p class="meta">MAE in MW of served forecasts once their actuals landed, by horizon group. The
backtest column is the backtest's MAE for the same mix of horizons and times of day. A ratio
above 1 means live forecasts are doing worse than the backtest promised.</p>
<table>
<tr><th>Horizons</th><th class="num">Forecasts</th><th class="num">Realised MAE</th>
<th class="num">Backtest MAE</th><th class="num">Ratio</th></tr>
{compare_rows}
</table></section>

<h2>Drift</h2>
<section>
<p class="meta">Normed Wasserstein distance of the last {drift["current"]["rows"]:,}
half-hours against {escape(drift["reference"]["description"])}
({drift["reference"]["rows"]:,} half-hours). Thresholds: {escape(drift["thresholds"])}.</p>
<table>
<tr><th>Column</th><th>Kind</th><th class="num">Score</th><th class="num">Threshold</th>
<th>Verdict</th></tr>
{drift_rows}
</table></section>

<h2>Files</h2>
<section><ul class="links">
<li><a href="drift_report.html">Evidently drift report</a> (HTML)</li>
<li><a href="verdict.json">Drift verdict</a> (JSON)</li>
<li><a href="status.json">Everything on this page</a> (JSON)</li>
<li><a href="{REPO_URL}">Source on GitHub</a></li>
</ul></section>
</main>
</body>
</html>
"""


def _css(status: str) -> str:
    return {BREACHED: "breached", INSUFFICIENT: "insufficient"}.get(status, "ok")


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:,.2f}"
