"use client";

import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { HorizonError } from "@/lib/data";
import { horizonLabel, mw } from "@/lib/format";

export default function ErrorChart({ errors }: { errors: HorizonError[] }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={errors} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis
          dataKey="horizon"
          tickFormatter={horizonLabel}
          ticks={[1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]}
          tick={{ fill: "var(--muted)", fontSize: 12 }}
          stroke="var(--grid)"
        />
        <YAxis
          tickFormatter={(v: number) => `${v}`}
          tick={{ fill: "var(--muted)", fontSize: 12 }}
          stroke="var(--grid)"
          width={48}
          label={{
            value: "MAE (MW)",
            angle: -90,
            position: "insideLeft",
            fill: "var(--muted)",
            fontSize: 12,
          }}
        />
        <Tooltip content={<ErrorTooltip />} />
        <Legend wrapperStyle={{ fontSize: 13, paddingTop: 8 }} />
        <Bar
          dataKey="realised"
          name="Realised, last 30 days"
          fill="var(--chart-forecast)"
          isAnimationActive={false}
        />
        <Line
          dataKey="backtest"
          name="Backtest (365 days)"
          stroke="var(--chart-actual)"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function ErrorTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: HorizonError }[];
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-md border border-[var(--line)] bg-[var(--card)] px-3 py-2 text-sm shadow-sm">
      <div className="mb-1 font-medium">{horizonLabel(row.horizon)} ahead</div>
      <div>
        Realised: {mw(row.realised)} ({row.n.toLocaleString("en-AU")} forecasts)
      </div>
      <div>Backtest: {mw(row.backtest)}</div>
    </div>
  );
}
