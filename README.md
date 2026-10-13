# nem-forecast

Half-hourly electricity demand forecasting for Victoria (region `VIC1`) in Australia's National
Electricity Market, built from public AEMO NEMWeb data. It forecasts 48 half-hour periods
(24 hours) ahead with 10th–90th percentile prediction intervals.

This is a production-style ML system on free tiers: ingestion that handles AEMO's revised data,
a backtesting harness built before any model, experiment tracking, a model registry, a served
API that logs every forecast, and monitoring that scores those forecasts against actuals.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold: layout, config, domain schemas, SQL migrations, CLI, CI | ✅ Done |
| 1 | Ingestion: NEMWeb client, record-type parser, idempotent upsert with revision history, backfill | ✅ Done |
| 2 | Features and backtesting: leakage-safe features, baselines, walk-forward harness | ✅ Done |
| 3 | Models and tracking: Ridge, LightGBM, quantile LightGBM, Optuna, DagsHub MLflow, HF Hub registry | ✅ Done |
| 4 | Serving: FastAPI, prediction logging, Docker | ⬜ Not started |
| 5 | Deployment and CI/CD: Render, hardened CI, nightly backtest | ⬜ Not started |
| 6 | Monitoring and retraining: realised-error scoring, Evidently drift, gated retraining, dashboard | ⬜ Not started |

## Architecture

_Placeholder. The full diagram lands in Phase 5._

Planned flow: AEMO NEMWeb → ingestion (GitHub Actions cron) → Neon Postgres (`raw_dispatch`)
→ features → walk-forward backtest → training (tracked in DagsHub MLflow) → model on Hugging
Face Hub → FastAPI on Render (logs to `predictions`) → monitoring (scores against actuals,
drift checks, gated retraining).

| Service | Role |
|---|---|
| GitHub Actions | CI, scheduled ingestion, monitoring and retraining |
| Neon Postgres | Raw dispatch data, features, predictions, run metrics |
| DagsHub MLflow | Experiment tracking |
| Hugging Face Hub | Model registry |
| Render | Inference API |
| Vercel | Dashboard |

## Ingestion

Two AEMO feeds, both public and keyless:

| Feed | Used for | Format |
|---|---|---|
| NEMWeb `Reports/Current/DispatchIS_Reports` | Scheduled ingest every 6 hours (last 24 h) | One zipped MMS report per 5-minute interval; about 48 hours retained |
| aemo.com.au monthly `PRICE_AND_DEMAND_<yyyymm>_<region>.csv` | Backfill, and repairing gaps | Plain CSV, one file per region per month, updated daily |

What the real files showed, and what the code does about it:

- **MMS reports interleave tables.** A DispatchIS file is a stream of `C` (comment), `I`
  (header) and `D` (data) records for seven tables of different widths, each with a version
  number. The parser keys every data row to the latest header for its table and reads columns
  by name, so a new AEMO table version that adds columns does not shift values. Demand comes
  from `DISPATCH.REGIONSUM.TOTALDEMAND`, price from `DISPATCH.PRICE.RRP`, pricing run only
  (`INTERVENTION = 0`).
- **Market time has no daylight saving.** On 5 Oct 2025 Melbourne clocks jumped from 02:00 to
  03:00, yet AEMO's file has all 288 intervals including 02:05-02:55. Timestamps are fixed
  UTC+10 and mark the interval *end*. A test against the real October file pins this.
- **AEMO revises data.** Every load is an upsert on `(settlement_date, region)`. An unchanged
  value is left completely untouched; a changed demand or price is copied to
  `raw_dispatch_history` before being overwritten and `revision` is incremented. Re-running any
  file produces zero changes. The two feeds publish identical values for the same interval, so
  switching source is never mistaken for a revision.
- **Bad rows never sink a batch.** Malformed rows are logged and skipped; missing intervals
  are counted and recorded in `ingest_log` with every batch.

The HTTP client identifies itself in its User-Agent, makes at most one request per second,
retries server errors with exponential backoff (3 retries), and caches downloads locally in
`.cache/nemweb/` so development re-runs never re-download a file.

```bash
uv run nem-forecast backfill                      # 3 years of VIC1, month by month, resumable
uv run nem-forecast ingest                        # last 24 hours of 5-minute data
uv run nem-forecast ingest --start "2026-10-12 04:00" --end "2026-10-13 04:00" --region VIC1
```

Backfill commits each month together with its `ingest_log` row, so an interrupted run loses at
most the month in flight; re-running skips every month already marked complete.

## Features and backtesting

The evaluation harness was built before any model, so it cannot have been bent to flatter one.

**Leakage is checked, not assumed.** A forecast made at an *origin* for 48 half-hours ahead may
only use data up to that origin, which is stricter than "data before the target": at horizon 5,
the demand at t-1 doesn't exist yet. Every data-derived feature carries an `_asof` column holding
the timestamp of the observation it actually used. The value is fetched by joining on that
timestamp, so provenance can't drift from reality. `assert_no_leakage` checks every `_asof`
against the row's cutoff, refuses any column it can't trace, and runs on every fold of every
backtest. A separate test rewrites all data after the origin and asserts that no feature moves.

| Feature | Definition (at horizon h, target t) | Latest data used |
|---|---|---|
| `demand_at_origin` | demand of the period ending at the origin (lag t-1 at h = 1) | origin |
| `same_slot_prev_day` | same half-hour on the latest fully known day (lag t-48) | t - 1 day |
| `same_slot_prev_week` | same half-hour a week earlier (lag t-336) | t - 7 days |
| `same_slot_mean_4w` | mean of the same half-hour on the last 4 same weekdays | t - 7 days |
| `rolling_mean_24h`, `rolling_std_24h`, `rolling_mean_7d` | windows ending at the origin | origin |
| `half_hour`, `day_of_week`, `month`, `is_public_holiday` | calendar of the target, Melbourne wall-clock time, VIC holidays | known in advance |

**Walk-forward backtest.** Expanding window, refit every 30 days, a forecast from *every*
half-hourly origin, 48 horizons, over the most recent 365 days: 842,040 scored forecasts per
model. Forecasting from every origin keeps horizon and time of day independent; with one daily
origin, "horizon 12" would just mean "noon". Metrics are MAE, RMSE, MAPE, pinball loss at the
10th/50th/90th percentiles and 10-90 interval coverage, each by horizon and by time-of-day
bucket. Runs and metrics are saved to `model_runs` and `model_run_metrics`.

**Baselines** ([full report](docs/backtest/README.md)), 2025-10-13 to 2026-10-13:

| Baseline | MAE (MW) | MAPE (%) | Pinball avg (MW) | 10-90 coverage (%) |
|---|--:|--:|--:|--:|
| Seasonal naive: same half-hour yesterday | 506 | 11.0 | 179.0 | 80.0 |
| Weekly seasonal naive: same half-hour last week | 592 | 13.0 | 209.8 | 79.4 |
| Mean of the same half-hour on the last 4 same weekdays | **502** | **10.9** | **177.8** | 74.7 |
| Persistence: last observed value, held flat | 970 | 20.8 | 300.0 | 79.7 |

What the baselines show:

- **The three seasonal baselines are flat across horizons by construction.** Each reads a fixed
  lag of the target, so it ignores everything observed since. Against them alone, any model that
  sees recent demand would look spectacular at short horizons for free. That's why
  **persistence** is included alongside the doc's three: it is the textbook short-horizon
  benchmark, and it wins clearly up to 1.5 hours ahead (170 MW at 30 minutes).
- **The bar a model must clear is therefore horizon-dependent:** persistence for horizons 1-3
  (30-90 minutes), the 4-week same-weekday mean for every horizon from 2 hours to 24 hours.
- **The midday solar trough is the hardest time of day** for every baseline (796-1,068 MW MAE
  from 10:00 to 16:00), 3-4x the seasonal baselines' overnight error. Rooftop solar output depends on cloud
  cover, which none of these rules can see.
- **The 4-week mean is the most accurate but its intervals are the worst calibrated**: 74.7%
  coverage against a nominal 80%. Averaging four weeks smooths the point forecast, which makes
  its recent errors less representative of the next ones.

Re-run with `uv run nem-forecast build-features` and `uv run nem-forecast backtest --no-models`.

## Models and results

Three models, all **direct** multi-horizon (one sub-model per horizon group, never feeding their
own forecasts back in), trained on the same leakage-checked features and scored in the same
walk-forward backtest as the baselines:

- **Ridge regression**: standardised lags and one-hot calendar; the linear reference point.
- **LightGBM direct**: squared error, one model per horizon group (1 | 2-3 | 4-6 | 7-12 | 13-24 |
  25-48 half-hours). Its intervals come from its own errors on the 28 days before each refit,
  predicted by a copy trained without them, so they are out-of-sample.
- **LightGBM quantile**: quantile regression at 0.1 / 0.5 / 0.9, same groups.

Hyperparameters were tuned with Optuna (50 trials) on a validation split, 2025-08-14 to
2025-10-13, that ends where the backtest begins; validation MAE improved from 374.5 to 360.9 MW
([parameters](config/lightgbm_vic1.json)). Every trial, every model and every walk-forward fold
is logged to [MLflow on DagsHub](https://dagshub.com/proxykhan/nem-forecast.mlflow/#/experiments)
with its parameters, metrics by horizon, feature list, data range, git SHA and the report.

**Results**, 2025-10-13 to 2026-10-13, 842,040 forecasts per model
([full report](docs/backtest/README.md)):

| Model | Kind | MAE (MW) | MAPE (%) | Pinball avg (MW) | 10-90 coverage (%) |
|---|---|--:|--:|--:|--:|
| **LightGBM quantile** | model | **351** | **7.6** | **114.3** | 68.5 |
| LightGBM direct | model | 353 | 7.6 | 124.1 | 77.0 |
| Ridge | model | 383 | 8.3 | 131.3 | 77.8 |
| 4-week same-weekday mean | baseline | 502 | 10.9 | 177.8 | 74.7 |
| Seasonal naive | baseline | 506 | 11.0 | 179.0 | 80.0 |
| Weekly seasonal naive | baseline | 592 | 13.0 | 209.8 | 79.4 |
| Persistence | baseline | 970 | 20.8 | 300.0 | 79.7 |

**MAE (MW) by horizon**

| Model | 30 min | 1 h | 2 h | 3 h | 6 h | 12 h | 18 h | 24 h |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| LightGBM quantile | **103** | 162 | 246 | 289 | **360** | **370** | **377** | **387** |
| LightGBM direct | **103** | **160** | **242** | **285** | **360** | 375 | 380 | 393 |
| Ridge | 132 | 209 | 297 | 358 | 408 | 406 | 404 | 404 |
| Seasonal naive | 507 | 507 | 507 | 507 | 507 | 506 | 506 | 506 |
| Weekly seasonal naive | 592 | 592 | 592 | 592 | 592 | 592 | 592 | 593 |
| 4-week same-weekday mean | 502 | 502 | 502 | 502 | 502 | 502 | 502 | 502 |
| Persistence | 170 | 324 | 600 | 830 | 1,181 | 996 | 1,196 | 506 |

**Pinball loss, average of q10/q50/q90 (MW), by horizon**

| Model | 30 min | 1 h | 2 h | 3 h | 6 h | 12 h | 18 h | 24 h |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| LightGBM quantile | **36** | **54** | **80** | **94** | **117** | **121** | **123** | **126** |
| LightGBM direct | **36** | 55 | 84 | 100 | 127 | 131 | 134 | 139 |
| Ridge | 45 | 71 | 101 | 121 | 139 | 139 | 139 | 139 |
| Seasonal naive | 179 | 179 | 179 | 179 | 179 | 179 | 179 | 179 |
| Weekly seasonal naive | 210 | 210 | 210 | 210 | 210 | 210 | 210 | 210 |
| 4-week same-weekday mean | 178 | 178 | 178 | 178 | 178 | 178 | 178 | 178 |
| Persistence | 53 | 101 | 186 | 255 | 359 | 307 | 367 | 179 |

![MAE by horizon](docs/backtest/mae_by_horizon.png)

What this does and does not show:

- **Every model beats seasonal naive, and the best baseline, at all 48 horizons.** The report
  computes this per horizon rather than asserting it. The margin narrows with distance: LightGBM
  is 80% better than seasonal naive at 30 minutes and 24% better a day ahead, where, without
  weather inputs, there is little to know beyond the calendar and the recent level.
- **The LightGBM quantile intervals are too narrow.** Its 10-90 interval held 68.5% of actual
  demand against a nominal 80%. It still has the best pinball loss because its intervals are
  sharp, but its p10 and p90 are optimistic. LightGBM direct's held-out intervals are better
  calibrated (77.0%).
- **Tuning bought little.** 50 trials improved validation MAE by 3.6%; the features matter far
  more than the hyperparameters.
- **The midday solar trough is still the hardest period**, at about 600 MW MAE against 171 MW
  overnight. Cloud cover drives rooftop solar, and no model here can see it.
- **Error steps slightly at horizon-group boundaries**, a side effect of one model per group.

**Registry.** The LightGBM quantile model, trained on all data to 2026-10-13, is published on
Hugging Face as [Azharkhan79/nem-forecast-vic1](https://huggingface.co/Azharkhan79/nem-forecast-vic1),
tagged with the git SHA of the code that trained it. Its
[model card](https://huggingface.co/Azharkhan79/nem-forecast-vic1) is generated from the recorded
backtest metrics: training range, features, comparison with every baseline by horizon, known
failure modes and intended use. The files are JSON plus LightGBM's plain-text booster format;
nothing is pickled. Promotion refuses uncommitted code.

```bash
uv run nem-forecast tune                                         # Optuna, logged to MLflow
uv run nem-forecast backtest                                     # baselines + models, saved and tracked
uv run nem-forecast train --model lightgbm_quantile --promote    # train on all data, publish
```

## Database

Schema changes are plain numbered SQL files in [`migrations/`](migrations/), applied in order by
`nem-forecast migrate` and recorded in `schema_migrations` with a checksum. Applied migrations
are never edited; a change is always a new file.

| Table | Purpose |
|---|---|
| `raw_dispatch` | AEMO demand and price, unique on `(settlement_date, region)`, with a `revision` counter because AEMO republishes revised values |
| `raw_dispatch_history` | Every superseded value, with the file that replaced it |
| `ingest_log` | One row per load batch: period, counts, missing intervals, completeness |
| `features` | Horizon-1 feature rows (`v1`); multi-horizon frames are built in memory by the same code, since storing every (origin, horizon) pair would be 48x larger |
| `features` | Versioned feature rows with a `data_cutoff` used to guard against leakage |
| `predictions` | Every forecast served, the ground truth for monitoring |
| `model_runs`, `model_run_metrics` | Backtest and training runs, with metrics by horizon and time of day |

All timestamps are `TIMESTAMPTZ`. AEMO market time is fixed AEST (UTC+10) and does not follow
Victorian daylight saving.

## Local setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 itself if needed.

```bash
git clone https://github.com/proxykhan/nem-forecast.git && cd nem-forecast
uv sync --all-groups
cp .env.example .env        # then fill in real values
uv run nem-forecast migrate
uv run pytest --cov
```

## Configuration

All configuration comes from environment variables (see [`.env.example`](.env.example)). Secrets
live only in GitHub Actions secrets and the hosting provider's environment settings, never in
this repository.

| Variable | Required | Used for |
|---|---|---|
| `DATABASE_URL` | yes | Neon Postgres |
| `ENVIRONMENT` | no (`local`) | `local`, `ci` or `production`; production requires every credential below |
| `MLFLOW_TRACKING_URI` | production | DagsHub MLflow server |
| `MLFLOW_TRACKING_USERNAME` | production | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | production | DagsHub token |
| `HF_TOKEN` | production | Hugging Face Hub model registry |

The scheduled [Ingest workflow](.github/workflows/ingest.yml) reads `DATABASE_URL` from a
repository secret: `gh secret set DATABASE_URL` and paste the Neon connection string.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest --cov
```

Integration tests need a PostgreSQL server; locally they skip without one, in CI a Postgres
service container runs them all. Start a throwaway server with Docker and they will find it on
`localhost:5432`:

```bash
docker run --rm -d --name nem-pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:17
```

Override the connection with `TEST_PG_HOST`, `TEST_PG_PORT`, `TEST_PG_USER` and
`TEST_PG_PASSWORD`.

## License

MIT
