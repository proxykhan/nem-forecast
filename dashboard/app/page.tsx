import type { ReactNode } from "react";

import ErrorChart from "@/components/ErrorChart";
import ForecastChart from "@/components/ForecastChart";
import { type Check, type Dashboard, loadDashboard } from "@/lib/data";
import { ago, date, mw, when } from "@/lib/format";

// Regenerated at most every 5 minutes; visitors get the last good page in between.
export const revalidate = 300;

const REPO = "https://github.com/proxykhan/nem-forecast";
const API = "https://nem-forecast-api.onrender.com";
const PAGES = "https://proxykhan.github.io/nem-forecast/";
const MODEL = "https://huggingface.co/Azharkhan79/nem-forecast-vic1";

export default async function Page() {
  const url = process.env.DATABASE_URL;
  let data: Dashboard | null = null;
  let problem: string | null = null;
  if (!url) {
    problem = "DATABASE_URL is not set, so there is nothing to show.";
  } else {
    try {
      data = await loadDashboard(url);
    } catch (error) {
      console.error("dashboard query failed", error);
      problem = "The database could not be reached. The page will retry within five minutes.";
    }
  }

  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 sm:py-10">
      <Header />
      {data ? <Body data={data} /> : <Notice>{problem}</Notice>}
      <Footer generatedAt={data?.generatedAt} />
    </main>
  );
}

function Header() {
  return (
    <header className="mb-8">
      <p className="text-sm font-medium tracking-wide text-[var(--muted)] uppercase">
        nem-forecast · VIC1
      </p>
      <h1 className="mt-1 text-2xl font-semibold sm:text-3xl">
        Victorian electricity demand: live forecast and monitoring
      </h1>
      <p className="mt-2 max-w-3xl text-[var(--muted)]">
        Half-hourly demand forecasts from a public API, logged as they are served and checked
        every day against what actually happened. Data: AEMO NEMWeb, ingested every half-hour.
      </p>
      <nav className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-sm">
        <Link href={REPO}>Source on GitHub</Link>
        <Link href={`${API}/docs`}>Forecast API</Link>
        <Link href={PAGES}>Daily monitoring report</Link>
        <Link href={MODEL}>Model card</Link>
      </nav>
    </header>
  );
}

function Body({ data }: { data: Dashboard }) {
  const { forecast, monitor, retrain, model } = data;
  const scored = data.errors.filter((e) => e.realised !== null && e.backtest !== null);
  const weight = scored.reduce((sum, e) => sum + e.n, 0);
  // The daily monitor's estimate weights the backtest by horizon and time of day; before its
  // first run, weight by horizon alone.
  const pooled = monitor?.comparison.find((row) => row.group === "all");
  const expected = pooled
    ? { mae: pooled.backtest_mae, basis: "the same horizons and times of day" }
    : weight > 0
      ? {
          mae: scored.reduce((sum, e) => sum + e.n * (e.backtest ?? 0), 0) / weight,
          basis: "the same horizons",
        }
      : null;

  return (
    <div className="flex flex-col gap-6">
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Tile label="Latest data">
          {data.latestData ? (
            <>
              <Big>{ago(data.latestData, data.generatedAt)}</Big>
              <Small>{when(data.latestData)}</Small>
            </>
          ) : (
            <Big>none</Big>
          )}
        </Tile>
        <Tile label="Model in production">
          <Big mono>{model?.version ?? "–"}</Big>
          <Small>
            {model?.name ?? ""}
            {model?.trainedTo ? `, trained on data to ${date(model.trainedTo)}` : ""}
          </Small>
        </Tile>
        <Tile label="Realised MAE, last 30 days">
          <Big>{mw(data.realised30d.mae)}</Big>
          <Small>
            {data.realised30d.n > 0
              ? `${data.realised30d.n.toLocaleString("en-AU")} forecasts from ${data.realised30d.origins} origins; the backtest expects ${mw(expected?.mae)} for ${expected?.basis}`
              : "No forecasts scored yet"}
          </Small>
        </Tile>
        <Tile label="Drift and retraining">
          <DriftHeadline data={data} />
          <Small>
            {retrain
              ? `Last retrain ${date(retrain.startedAt)}: ${retrain.decision}${retrain.release ? ` (${retrain.release})` : ""}`
              : model?.trainedTo
                ? `No retrain yet; model trained ${date(model.trainedTo)}`
                : "No retrain yet"}
          </Small>
        </Tile>
      </section>

      <Card
        title="Latest forecast"
        subtitle={
          forecast
            ? `From the half-hour ending ${when(forecast.origin)}, served ${ago(forecast.servedAt, data.generatedAt)}. Shaded: the 80% interval (10th to 90th percentile). Dashed: the forecast issued a day earlier, against the demand that actually arrived.`
            : undefined
        }
      >
        {forecast ? (
          <ForecastChart actuals={data.actuals} forecast={forecast} previous={data.previous} />
        ) : (
          <Empty>No forecast has been served yet.</Empty>
        )}
      </Card>

      <Card
        title="Realised error by horizon"
        subtitle="Mean absolute error of served forecasts whose actual demand has landed, over the last 30 days, against the 365-day walk-forward backtest. A forecast is scored once, from the first time it was served; revised actuals are rescored."
      >
        {weight > 0 ? (
          <ErrorChart errors={data.errors} />
        ) : (
          <Empty>No served forecast has been scored yet.</Empty>
        )}
      </Card>

      <Card
        title="Monitoring checks"
        subtitle={
          monitor
            ? `Daily run ${when(monitor.runAt)} on model ${monitor.modelVersion}. Drift thresholds: ${monitor.thresholds}.`
            : undefined
        }
      >
        {monitor ? (
          <div className="flex flex-col gap-5">
            <ul className="flex flex-col gap-3">
              {monitor.checks.map((check) => (
                <CheckRow key={check.name} check={check} />
              ))}
            </ul>
            <DriftTable data={data} />
            {monitor.reportUrl && (
              <p className="text-sm">
                <Link href={monitor.reportUrl}>Full Evidently drift report</Link>
                <span className="text-[var(--muted)]"> · </span>
                <Link href={PAGES}>Monitoring summary</Link>
              </p>
            )}
          </div>
        ) : (
          <Empty>The daily monitor has not run yet.</Empty>
        )}
      </Card>
    </div>
  );
}

function DriftHeadline({ data }: { data: Dashboard }) {
  const monitor = data.monitor;
  if (!monitor) return <Big>not checked yet</Big>;
  const drifted = monitor.dataDrift.drifted_columns;
  return (
    <>
      <Big tone={drifted.length ? "bad" : "ok"}>
        {drifted.length ? `Drift in ${drifted.length} feature${drifted.length > 1 ? "s" : ""}` : "No feature drift"}
      </Big>
      <Small>checked {when(monitor.runAt)}</Small>
    </>
  );
}

function DriftTable({ data }: { data: Dashboard }) {
  const monitor = data.monitor!;
  const rows: { name: string; kind: string; score?: number; threshold?: number; verdict: string; tone: Tone }[] =
    Object.entries(monitor.dataDrift.columns)
      .sort(([a], [b]) => featureOrder(a) - featureOrder(b))
      .map(([name, c]) => ({
      name,
      kind: "feature",
      score: c.score,
      threshold: c.threshold,
      verdict: c.drifted ? "drift" : "no drift",
      tone: c.drifted ? "bad" : "ok",
    }));
  for (const [name, kind, entry] of [
    ["forecast", "prediction", monitor.predictionDrift],
    ["demand", "target", monitor.targetDrift],
  ] as const) {
    rows.push({
      name,
      kind,
      score: entry.score,
      threshold: entry.threshold,
      verdict: entry.drifted === null ? entry.status : entry.drifted ? "drift" : "no drift",
      tone: entry.drifted === null ? "warn" : entry.drifted ? "bad" : "ok",
    });
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] text-sm tabular-nums">
        <thead className="text-left text-[var(--muted)]">
          <tr className="border-b border-[var(--line)]">
            <th className="py-2 pr-4 font-medium">Column</th>
            <th className="py-2 pr-4 font-medium">Kind</th>
            <th className="py-2 pr-4 text-right font-medium">Drift score</th>
            <th className="py-2 pr-4 text-right font-medium">Threshold</th>
            <th className="py-2 font-medium">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.name} className="border-b border-[var(--line)] last:border-0">
              <td className="py-2 pr-4 font-mono text-[13px]">{row.name}</td>
              <td className="py-2 pr-4">{row.kind}</td>
              <td className="py-2 pr-4 text-right">{row.score?.toFixed(3) ?? "–"}</td>
              <td className="py-2 pr-4 text-right">{row.threshold?.toFixed(3) ?? "–"}</td>
              <td className={`py-2 ${TONE[row.tone]}`}>{row.verdict}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// The model's feature order; JSONB does not keep the order the monitor wrote them in.
const FEATURES = [
  "demand_at_origin",
  "same_slot_prev_day",
  "same_slot_prev_week",
  "same_slot_mean_4w",
  "rolling_mean_24h",
  "rolling_std_24h",
  "rolling_mean_7d",
];
const featureOrder = (name: string) =>
  FEATURES.includes(name) ? FEATURES.indexOf(name) : FEATURES.length;

type Tone = "ok" | "bad" | "warn";
const TONE: Record<Tone, string> = {
  ok: "text-[var(--ok)]",
  bad: "text-[var(--bad)] font-medium",
  warn: "text-[var(--warn)]",
};

function CheckRow({ check }: { check: Check }) {
  const tone: Tone =
    check.status === "breached" ? "bad" : check.status === "ok" ? "ok" : "warn";
  return (
    <li className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:gap-3">
      <span
        className={`inline-flex w-fit shrink-0 items-center rounded-full border px-2 py-0.5 text-xs font-medium ${TONE[tone]} border-current`}
      >
        {check.status}
      </span>
      <span>
        <span className="font-medium">{check.title}.</span>{" "}
        <span className="text-[var(--muted)]">{check.detail}.</span>
      </span>
    </li>
  );
}

function Card({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-[var(--line)] bg-[var(--card)] p-4 sm:p-6">
      <h2 className="text-lg font-semibold">{title}</h2>
      {subtitle && <p className="mt-1 mb-4 text-sm text-[var(--muted)]">{subtitle}</p>}
      {!subtitle && <div className="mb-4" />}
      {children}
    </section>
  );
}

function Tile({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--card)] p-4">
      <div className="text-sm text-[var(--muted)]">{label}</div>
      <div className="mt-1 flex flex-col gap-1">{children}</div>
    </div>
  );
}

function Big({ children, mono, tone }: { children: ReactNode; mono?: boolean; tone?: Tone }) {
  return (
    <div
      className={`text-xl font-semibold break-words ${mono ? "font-mono text-lg" : ""} ${tone ? TONE[tone] : ""}`}
    >
      {children}
    </div>
  );
}

function Small({ children }: { children: ReactNode }) {
  return <div className="text-sm text-[var(--muted)]">{children}</div>;
}

function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-[var(--line)] p-8 text-center text-[var(--muted)]">
      {children}
    </div>
  );
}

function Notice({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--card)] p-6 text-[var(--muted)]">
      {children}
    </div>
  );
}

function Link({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      href={href}
      className="underline decoration-[var(--line)] underline-offset-4 hover:decoration-current"
      target="_blank"
      rel="noreferrer"
    >
      {children}
    </a>
  );
}

function Footer({ generatedAt }: { generatedAt?: number }) {
  return (
    <footer className="mt-10 border-t border-[var(--line)] pt-4 text-sm text-[var(--muted)]">
      {generatedAt ? `Page generated ${when(generatedAt)}; refreshed at most every 5 minutes. ` : ""}
      Times are AEMO market time (AEST, UTC+10). Demand data © AEMO, used under its copyright
      permissions. Not for trading or operational decisions.
    </footer>
  );
}
