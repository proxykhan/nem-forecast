-- One row per daily monitoring run: the drift verdict, every alert check (breached or not) and
-- realised error against the backtest, as the JSON the run published. The dashboard reads the
-- latest row.
CREATE TABLE IF NOT EXISTS monitor_runs (
    id            BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    region        TEXT        NOT NULL,
    run_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version TEXT        NOT NULL,
    drift         JSONB       NOT NULL,
    alerts        JSONB       NOT NULL,
    realised      JSONB       NOT NULL,
    report_url    TEXT,
    git_sha       TEXT
);
CREATE INDEX IF NOT EXISTS monitor_runs_region_run_at_idx ON monitor_runs (region, run_at DESC);

-- One row per retraining attempt, promoted or not: the champion/challenger comparison on the
-- shared holdout is the record of why a model was (or was not) put into production.
CREATE TABLE IF NOT EXISTS retrain_runs (
    id                BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    region            TEXT        NOT NULL,
    started_at        TIMESTAMPTZ NOT NULL,
    finished_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    trigger           TEXT        NOT NULL,                  -- 'manual' or the alert that fired
    model_name        TEXT        NOT NULL,
    incumbent_version TEXT        NOT NULL,
    release           TEXT,                                  -- the new model's tag, if promoted
    holdout_start     TIMESTAMPTZ,
    holdout_end       TIMESTAMPTZ,
    decision          TEXT        NOT NULL CHECK (decision IN ('promoted', 'rejected', 'skipped')),
    reason            TEXT        NOT NULL,
    comparison        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    git_sha           TEXT
);
CREATE INDEX IF NOT EXISTS retrain_runs_region_started_idx ON retrain_runs (region, started_at DESC);
