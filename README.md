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
| 1 | Ingestion: NEMWeb client, record-type parser, idempotent upsert with revision history, backfill | ⬜ Not started |
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

## Database

Schema changes are plain numbered SQL files in [`migrations/`](migrations/), applied in order by
`nem-forecast migrate` and recorded in `schema_migrations` with a checksum. Applied migrations
are never edited; a change is always a new file.

| Table | Purpose |
|---|---|
| `raw_dispatch` | AEMO demand and price, unique on `(settlement_date, region)`, with a `revision` counter because AEMO republishes revised values |
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

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest --cov                 # unit tests; integration tests skip without Postgres
```

Integration tests need a PostgreSQL server. Start a throwaway one with Docker and they will find
it on `localhost:5432`:

```bash
docker run --rm -d --name nem-pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:17
```

Override the connection with `TEST_PG_HOST`, `TEST_PG_PORT`, `TEST_PG_USER` and
`TEST_PG_PASSWORD`.

## License

MIT
