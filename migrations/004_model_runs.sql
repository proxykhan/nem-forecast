-- One row per backtest / training run, for baselines and models alike.
CREATE TABLE IF NOT EXISTS model_runs (
    run_id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name       TEXT        NOT NULL,             -- 'seasonal_naive', 'lightgbm_quantile', ...
    model_kind       TEXT        NOT NULL CHECK (model_kind IN ('baseline', 'model')),
    run_type         TEXT        NOT NULL CHECK (run_type IN ('backtest', 'train', 'retrain')),
    region           TEXT        NOT NULL,
    git_sha          TEXT,
    mlflow_run_id    TEXT,
    hf_model_version TEXT,
    feature_version  TEXT,
    params           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    train_start      TIMESTAMPTZ,
    train_end        TIMESTAMPTZ,
    eval_start       TIMESTAMPTZ,
    eval_end         TIMESTAMPTZ,
    status           TEXT        NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at      TIMESTAMPTZ
);

-- Tidy metrics: one row per (run, metric, horizon, time-of-day bucket), so results by horizon
-- are a plain GROUP BY rather than JSON unpacking.
CREATE TABLE IF NOT EXISTS model_run_metrics (
    run_id     UUID             NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    metric     TEXT             NOT NULL,              -- mae, rmse, mape, pinball_q10, pinball_q50, pinball_q90
    horizon    SMALLINT         NOT NULL,              -- 1..96; 0 = all horizons combined
    tod_bucket TEXT             NOT NULL DEFAULT 'all',
    value      DOUBLE PRECISION NOT NULL,
    n_obs      INTEGER          NOT NULL,
    PRIMARY KEY (run_id, metric, horizon, tod_bucket)
);
