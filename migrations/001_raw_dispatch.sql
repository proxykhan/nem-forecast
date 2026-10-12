-- Raw regional demand and price from AEMO NEMWeb.
--
-- AEMO market time is fixed AEST (UTC+10, no daylight saving). The parser attaches +10:00
-- before insert, so TIMESTAMPTZ stores the true instant.
--
-- AEMO republishes revised values, so rows are upserted on (settlement_date, region) and
-- `revision` counts how many distinct values a row has held (1 = never revised).
CREATE TABLE IF NOT EXISTS raw_dispatch (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    settlement_date TIMESTAMPTZ      NOT NULL,
    region          TEXT             NOT NULL CHECK (region IN ('NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1')),
    demand_mw       DOUBLE PRECISION NOT NULL,
    rrp             DOUBLE PRECISION,                  -- $/MWh, can be negative
    source_file     TEXT             NOT NULL,
    revision        INTEGER          NOT NULL DEFAULT 1 CHECK (revision >= 1),
    ingested_at     TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT raw_dispatch_settlement_region_uq UNIQUE (settlement_date, region)
);

CREATE INDEX IF NOT EXISTS raw_dispatch_region_time_idx ON raw_dispatch (region, settlement_date DESC);
