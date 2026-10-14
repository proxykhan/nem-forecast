"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Actual, Forecast } from "@/lib/data";
import { clock, mw, when } from "@/lib/format";

type Row = {
  t: number;
  actual?: number;
  p50?: number;
  band?: [number, number];
  previous?: number;
};

const COLORS = {
  actual: "var(--chart-actual)",
  forecast: "var(--chart-forecast)",
  band: "var(--chart-band)",
  previous: "var(--chart-previous)",
  grid: "var(--grid)",
  muted: "var(--muted)",
};

export default function ForecastChart({
  actuals,
  forecast,
  previous,
}: {
  actuals: Actual[];
  forecast: Forecast;
  previous: Forecast | null;
}) {
  const rows = new Map<number, Row>();
  const at = (t: number) => rows.get(t) ?? rows.set(t, { t }).get(t)!;
  for (const a of actuals) at(a.t).actual = a.mw;
  for (const p of forecast.points) {
    const row = at(p.t);
    row.p50 = p.p50;
    if (p.p10 !== null && p.p90 !== null) row.band = [p.p10, p.p90];
  }
  // Anchor the forecast to the last actual so the lines join at the origin.
  const anchor = rows.get(forecast.origin);
  if (anchor?.actual !== undefined) {
    anchor.p50 = anchor.actual;
    anchor.band = [anchor.actual, anchor.actual];
  }
  for (const p of previous?.points ?? []) at(p.t).previous = p.p50;
  const data = [...rows.values()].sort((a, b) => a.t - b.t);

  return (
    <ResponsiveContainer width="100%" height={340}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        <XAxis
          dataKey="t"
          type="number"
          scale="time"
          domain={["dataMin", "dataMax"]}
          tickFormatter={clock}
          tick={{ fill: COLORS.muted, fontSize: 12 }}
          stroke={COLORS.grid}
          minTickGap={40}
        />
        <YAxis
          tickFormatter={(v: number) => `${(v / 1000).toFixed(1)} GW`}
          tick={{ fill: COLORS.muted, fontSize: 12 }}
          stroke={COLORS.grid}
          width={56}
          domain={["auto", "auto"]}
        />
        <Tooltip content={<ForecastTooltip />} />
        <Legend wrapperStyle={{ fontSize: 13, paddingTop: 8 }} />
        <Area
          dataKey="band"
          name="80% interval"
          fill={COLORS.band}
          stroke="none"
          isAnimationActive={false}
          connectNulls
        />
        {previous && (
          <Line
            dataKey="previous"
            name="Forecast issued a day earlier"
            stroke={COLORS.previous}
            strokeDasharray="5 4"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        )}
        <Line
          dataKey="actual"
          name="Actual demand"
          stroke={COLORS.actual}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          dataKey="p50"
          name="Latest forecast"
          stroke={COLORS.forecast}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
          connectNulls
        />
        <ReferenceLine
          x={forecast.origin}
          stroke={COLORS.muted}
          strokeDasharray="2 3"
          label={{ value: "origin", position: "insideTopRight", fill: COLORS.muted, fontSize: 12 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function ForecastTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: Row }[];
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-md border border-[var(--line)] bg-[var(--card)] px-3 py-2 text-sm shadow-sm">
      <div className="mb-1 font-medium">{when(row.t)}</div>
      {row.actual !== undefined && <div>Actual: {mw(row.actual)}</div>}
      {row.p50 !== undefined && <div>Forecast: {mw(row.p50)}</div>}
      {row.band && (
        <div className="text-[var(--muted)]">
          80% interval: {mw(row.band[0])} to {mw(row.band[1])}
        </div>
      )}
      {row.previous !== undefined && <div>Day-earlier forecast: {mw(row.previous)}</div>}
    </div>
  );
}
