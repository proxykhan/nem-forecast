-- Engineered features, one row per (settlement_date, region, feature_version).
--
-- Features live in JSONB under a version tag so adding a feature does not need a migration;
-- bump feature_version when a definition changes.
--
-- data_cutoff is the latest input timestamp used to build the row. assert_no_leakage() checks
-- against it, and the CHECK below is the last line of defence.
CREATE TABLE IF NOT EXISTS features (
    settlement_date TIMESTAMPTZ NOT NULL,
    region          TEXT        NOT NULL,
    feature_version TEXT        NOT NULL,
    data_cutoff     TIMESTAMPTZ NOT NULL,
    feature_values  JSONB       NOT NULL,
    built_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (settlement_date, region, feature_version),
    CHECK (data_cutoff <= settlement_date)
);
