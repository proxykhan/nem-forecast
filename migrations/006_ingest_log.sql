-- One row per committed load batch (a backfill month, or one scheduled ingest run). Written in
-- the same transaction as the data, so a row here means the batch is fully in raw_dispatch.
-- Backfill resumes by skipping months that have a `complete` batch.
CREATE TABLE IF NOT EXISTS ingest_log (
    id                BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source            TEXT        NOT NULL CHECK (source IN ('dispatch_is', 'price_and_demand')),
    regions           TEXT        NOT NULL,            -- comma-separated, e.g. 'VIC1'
    period_start      TIMESTAMPTZ NOT NULL,            -- exclusive: first interval ends 5 min later
    period_end        TIMESTAMPTZ NOT NULL,            -- inclusive interval-ending time
    files             INTEGER     NOT NULL CHECK (files >= 0),
    rows_parsed       INTEGER     NOT NULL,
    rows_skipped      INTEGER     NOT NULL,
    rows_inserted     INTEGER     NOT NULL,
    rows_updated      INTEGER     NOT NULL,
    rows_unchanged    INTEGER     NOT NULL,
    missing_intervals INTEGER     NOT NULL,
    -- The whole period had ended and every interval in it arrived; nothing left to fetch.
    complete          BOOLEAN     NOT NULL,
    loaded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (period_end > period_start)
);

CREATE INDEX IF NOT EXISTS ingest_log_source_period_idx ON ingest_log (source, regions, period_start);
