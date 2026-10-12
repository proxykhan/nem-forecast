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
| 2 | Features and backtesting: leakage-safe features, three baselines, walk-forward harness | ⬜ Not started |
| 3 | Models and tracking: Ridge, LightGBM, quantile LightGBM, Optuna, DagsHub MLflow, HF Hub registry | ⬜ Not started |
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

## Database

Schema changes are plain numbered SQL files in [`migrations/`](migrations/), applied in order by
`nem-forecast migrate` and recorded in `schema_migrations` with a checksum. Applied migrations
are never edited; a change is always a new file.

| Table | Purpose |
|---|---|
| `raw_dispatch` | AEMO demand and price, unique on `(settlement_date, region)`, with a `revision` counter because AEMO republishes revised values |
| `raw_dispatch_history` | Every superseded value, with the file that replaced it |
| `ingest_log` | One row per load batch: period, counts, missing intervals, completeness |
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
