// Everything the dashboard shows, read straight from Neon on the server with a read-only login
// (`nem-forecast grant-reader`). The page is regenerated at most every five minutes, so a visit
// never waits on the forecast API, and the database sees a handful of queries per five minutes.

import { neon, type NeonQueryFunction } from "@neondatabase/serverless";

export const REGION = "VIC1";
const HOUR = 3_600_000;

export type ForecastPoint = { t: number; p10: number | null; p50: number; p90: number | null };
export type Forecast = {
  origin: number;
  servedAt: number;
  version: string;
  points: ForecastPoint[];
};
export type Actual = { t: number; mw: number };
export type HorizonError = {
  horizon: number;
  realised: number | null;
  n: number;
  backtest: number | null;
};
export type Check = {
  name: string;
  title: string;
  status: "ok" | "breached" | "insufficient data";
  value: number | null;
  threshold: number | null;
  detail: string;
};
export type ColumnDrift = { score: number; threshold: number; drifted: boolean };
export type SingleDrift = {
  drifted: boolean | null;
  status: string;
  score?: number;
  threshold?: number;
};
export type Monitor = {
  runAt: number;
  modelVersion: string;
  checks: Check[];
  dataDrift: { drifted: boolean; drifted_columns: string[]; columns: Record<string, ColumnDrift> };
  predictionDrift: SingleDrift;
  targetDrift: SingleDrift;
  thresholds: string;
  reportUrl: string | null;
  // Realised against backtest MAE for the same mix of horizons and times of day, by group.
  comparison: { group: string; n: number; realised_mae: number; backtest_mae: number; ratio: number }[];
};
export type Retrain = {
  startedAt: number;
  decision: "promoted" | "rejected" | "skipped";
  release: string | null;
  reason: string;
  trigger: string;
};
export type Model = { version: string; name: string; trainedTo: number | null };

export type Dashboard = {
  generatedAt: number;
  latestData: number | null;
  model: Model | null;
  forecast: Forecast | null;
  previous: Forecast | null; // the forecast issued about a day before the latest one
  actuals: Actual[];
  errors: HorizonError[];
  realised30d: { mae: number | null; n: number; origins: number };
  monitor: Monitor | null;
  retrain: Retrain | null;
};

type Row = Record<string, unknown>;

function ms(value: unknown): number {
  return value instanceof Date ? value.getTime() : Date.parse(String(value));
}

function num(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function json<T>(value: unknown): T {
  return (typeof value === "string" ? JSON.parse(value) : value) as T;
}

export async function loadDashboard(databaseUrl: string): Promise<Dashboard> {
  const sql = neon(databaseUrl);

  const [latestData, latestForecast, monitorRows, retrainRows, errorRows, summaryRows] =
    await Promise.all([
      sql`SELECT max(settlement_date) AS t FROM raw_dispatch WHERE region = ${REGION}`,
      sql`SELECT forecast_origin, served_at, model_version FROM predictions
          WHERE region = ${REGION} ORDER BY forecast_origin DESC, served_at ASC LIMIT 1`,
      sql`SELECT run_at, model_version, drift, alerts, realised, report_url FROM monitor_runs
          WHERE region = ${REGION} ORDER BY run_at DESC LIMIT 1`,
      sql`SELECT started_at, decision, release, reason, trigger FROM retrain_runs
          WHERE region = ${REGION} ORDER BY started_at DESC LIMIT 1`,
      sql`SELECT horizon, avg(abs(error_mw))::float8 AS mae, count(*)::int AS n
          FROM prediction_scores
          WHERE region = ${REGION} AND superseded_at IS NULL
            AND target_period > now() - interval '30 days'
          GROUP BY horizon ORDER BY horizon`,
      sql`SELECT avg(abs(error_mw))::float8 AS mae, count(*)::int AS n,
                 count(DISTINCT forecast_origin)::int AS origins
          FROM prediction_scores
          WHERE region = ${REGION} AND superseded_at IS NULL
            AND target_period > now() - interval '30 days'`,
    ]);

  const head = latestForecast[0] as Row | undefined;
  const forecast = head
    ? await loadForecast(sql, ms(head.forecast_origin), String(head.model_version))
    : null;
  const previous = forecast ? await loadPrevious(sql, forecast.origin) : null;
  const model = forecast ? await loadModel(sql, forecast.version) : null;

  const windowStart = (previous?.origin ?? (forecast ? forecast.origin - 24 * HOUR : Date.now()));
  const actualRows = await sql`
    SELECT date_bin('30 minutes', settlement_date - interval '1 microsecond',
                    TIMESTAMPTZ '2000-01-01 00:00:00+00') + interval '30 minutes' AS period_end,
           avg(demand_mw)::float8 AS mw
    FROM raw_dispatch
    WHERE region = ${REGION} AND settlement_date > ${new Date(windowStart).toISOString()}
    GROUP BY 1 HAVING count(*) = 6 ORDER BY 1`;

  const backtest = model ? await loadBacktest(sql, model.name) : new Map<number, number>();
  const realised = new Map(
    (errorRows as Row[]).map((r) => [Number(r.horizon), { mae: num(r.mae), n: Number(r.n) }]),
  );
  const horizons = Array.from({ length: 48 }, (_, i) => i + 1);
  const summary = summaryRows[0] as Row;

  const monitorRow = monitorRows[0] as Row | undefined;
  const retrainRow = retrainRows[0] as Row | undefined;

  return {
    generatedAt: Date.now(),
    latestData: latestData[0]?.t ? ms((latestData[0] as Row).t) : null,
    model,
    forecast,
    previous,
    actuals: (actualRows as Row[]).map((r) => ({ t: ms(r.period_end), mw: Number(r.mw) })),
    errors: horizons.map((h) => ({
      horizon: h,
      realised: realised.get(h)?.mae ?? null,
      n: realised.get(h)?.n ?? 0,
      backtest: backtest.get(h) ?? null,
    })),
    realised30d: {
      mae: num(summary?.mae),
      n: Number(summary?.n ?? 0),
      origins: Number(summary?.origins ?? 0),
    },
    monitor: monitorRow ? toMonitor(monitorRow) : null,
    retrain: retrainRow
      ? {
          startedAt: ms(retrainRow.started_at),
          decision: retrainRow.decision as Retrain["decision"],
          release: (retrainRow.release as string | null) ?? null,
          reason: String(retrainRow.reason),
          trigger: String(retrainRow.trigger),
        }
      : null,
  };
}

type Sql = NeonQueryFunction<false, false>;

async function loadForecast(sql: Sql, origin: number, version: string): Promise<Forecast> {
  const rows = (await sql`
    SELECT DISTINCT ON (target_period) target_period, predicted_mw, lower_mw, upper_mw, served_at
    FROM predictions
    WHERE region = ${REGION} AND model_version = ${version}
      AND forecast_origin = ${new Date(origin).toISOString()}
    ORDER BY target_period, served_at`) as Row[];
  return {
    origin,
    version,
    servedAt: Math.min(...rows.map((r) => ms(r.served_at))),
    points: rows.map((r) => ({
      t: ms(r.target_period),
      p10: num(r.lower_mw),
      p50: Number(r.predicted_mw),
      p90: num(r.upper_mw),
    })),
  };
}

async function loadPrevious(sql: Sql, origin: number): Promise<Forecast | null> {
  // The forecast issued closest to 24 hours before the latest one, within two hours either way.
  const target = new Date(origin - 24 * HOUR).toISOString();
  const rows = (await sql`
    SELECT forecast_origin, model_version FROM predictions
    WHERE region = ${REGION}
      AND forecast_origin BETWEEN ${target}::timestamptz - interval '2 hours'
                              AND ${target}::timestamptz + interval '2 hours'
    ORDER BY abs(extract(epoch FROM forecast_origin - ${target}::timestamptz)), served_at
    LIMIT 1`) as Row[];
  if (!rows[0]) return null;
  return loadForecast(sql, ms(rows[0].forecast_origin), String(rows[0].model_version));
}

async function loadModel(sql: Sql, version: string): Promise<Model> {
  const rows = (await sql`
    SELECT model_name, train_end FROM model_runs
    WHERE hf_model_version = ${version} AND run_type IN ('train', 'retrain')
    ORDER BY started_at DESC LIMIT 1`) as Row[];
  return {
    version,
    name: rows[0] ? String(rows[0].model_name) : "lightgbm_quantile",
    trainedTo: rows[0]?.train_end ? ms(rows[0].train_end) : null,
  };
}

async function loadBacktest(sql: Sql, model: string): Promise<Map<number, number>> {
  const rows = (await sql`
    WITH run AS (
      SELECT run_id FROM model_runs
      WHERE region = ${REGION} AND run_type = 'backtest' AND status = 'succeeded'
        AND model_name = ${model}
      ORDER BY started_at DESC LIMIT 1
    )
    SELECT horizon, value::float8 AS mae FROM model_run_metrics
    WHERE run_id = (SELECT run_id FROM run)
      AND metric = 'mae' AND tod_bucket = 'all' AND horizon > 0`) as Row[];
  return new Map(rows.map((r) => [Number(r.horizon), Number(r.mae)]));
}

function toMonitor(row: Row): Monitor {
  const drift = json<{
    data_drift: Monitor["dataDrift"];
    prediction_drift: SingleDrift;
    target_drift: SingleDrift;
    thresholds?: string;
  }>(row.drift);
  return {
    runAt: ms(row.run_at),
    modelVersion: String(row.model_version),
    checks: json<Check[]>(row.alerts),
    dataDrift: drift.data_drift,
    predictionDrift: drift.prediction_drift,
    targetDrift: drift.target_drift,
    thresholds: drift.thresholds ?? "",
    reportUrl: (row.report_url as string | null) ?? null,
    comparison: json<{ by_group?: Monitor["comparison"] }>(row.realised).by_group ?? [],
  };
}
