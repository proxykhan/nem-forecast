-- Superseded raw_dispatch values. When AEMO republishes an interval with a different value,
-- the loader copies the old row here before overwriting it, so every revision stays auditable
-- and monitoring can rescore forecasts against the actual that was known at the time.
CREATE TABLE IF NOT EXISTS raw_dispatch_history (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    raw_dispatch_id    BIGINT           NOT NULL REFERENCES raw_dispatch (id) ON DELETE CASCADE,
    settlement_date    TIMESTAMPTZ      NOT NULL,
    region             TEXT             NOT NULL,
    revision           INTEGER          NOT NULL,      -- the revision this value held
    demand_mw          DOUBLE PRECISION NOT NULL,
    rrp                DOUBLE PRECISION,
    source_file        TEXT             NOT NULL,      -- file that supplied this value
    valid_from         TIMESTAMPTZ      NOT NULL,      -- when this value was written
    superseded_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    superseded_by_file TEXT             NOT NULL,
    CONSTRAINT raw_dispatch_history_revision_uq UNIQUE (raw_dispatch_id, revision)
);

CREATE INDEX IF NOT EXISTS raw_dispatch_history_region_time_idx
    ON raw_dispatch_history (region, settlement_date);
