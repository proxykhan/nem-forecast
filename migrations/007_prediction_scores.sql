-- Realised error of served forecasts, written by the monitoring job once actuals land.
--
-- A forecast is (region, model_version, forecast_origin, target_period). The service may serve
-- the same forecast several times (every request is logged); the first-served values are the
-- ones scored, and prediction_id points at that row.
--
-- AEMO revises published data. When the actual behind a score changes, the old score is kept
-- with superseded_at set and a new score is added, so every number monitoring ever reported can
-- be reproduced. At most one score per forecast is current (superseded_at IS NULL).
CREATE TABLE IF NOT EXISTS prediction_scores (
    id              BIGINT           GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prediction_id   BIGINT           NOT NULL REFERENCES predictions (id) ON DELETE CASCADE,
    region          TEXT             NOT NULL,
    model_version   TEXT             NOT NULL,
    forecast_origin TIMESTAMPTZ      NOT NULL,
    target_period   TIMESTAMPTZ      NOT NULL,
    horizon         SMALLINT         NOT NULL CHECK (horizon BETWEEN 1 AND 96),
    predicted_mw    DOUBLE PRECISION NOT NULL,
    lower_mw        DOUBLE PRECISION,
    upper_mw        DOUBLE PRECISION,
    actual_mw       DOUBLE PRECISION NOT NULL,             -- half-hour mean of six 5-minute values
    error_mw        DOUBLE PRECISION GENERATED ALWAYS AS (predicted_mw - actual_mw) STORED,
    scored_at       TIMESTAMPTZ      NOT NULL DEFAULT now(),
    superseded_at   TIMESTAMPTZ,                            -- set when a revised actual rescored it
    CHECK (target_period > forecast_origin),
    CHECK (superseded_at IS NULL OR superseded_at >= scored_at)
);

CREATE UNIQUE INDEX IF NOT EXISTS prediction_scores_current_idx
    ON prediction_scores (region, model_version, forecast_origin, target_period)
    WHERE superseded_at IS NULL;
CREATE INDEX IF NOT EXISTS prediction_scores_target_idx ON prediction_scores (region, target_period);
CREATE INDEX IF NOT EXISTS prediction_scores_prediction_idx ON prediction_scores (prediction_id);
