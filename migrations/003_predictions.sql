-- Every forecast the API serves, one row per target period. Monitoring scores these rows
-- against actuals in raw_dispatch, so nothing is ever updated or deleted here.
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_id      UUID             NOT NULL,         -- groups the periods of one /forecast call
    region          TEXT             NOT NULL,
    forecast_origin TIMESTAMPTZ      NOT NULL,
    target_period   TIMESTAMPTZ      NOT NULL,
    horizon         SMALLINT         NOT NULL CHECK (horizon BETWEEN 1 AND 96),
    predicted_mw    DOUBLE PRECISION NOT NULL,         -- p50
    lower_mw        DOUBLE PRECISION,                  -- p10
    upper_mw        DOUBLE PRECISION,                  -- p90
    model_version   TEXT             NOT NULL,
    served_at       TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CHECK (target_period > forecast_origin),
    CHECK (lower_mw IS NULL OR upper_mw IS NULL OR lower_mw <= upper_mw)
);

CREATE INDEX IF NOT EXISTS predictions_region_target_idx ON predictions (region, target_period);
CREATE INDEX IF NOT EXISTS predictions_model_served_idx ON predictions (model_version, served_at);
