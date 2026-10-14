// AEMO publishes in market time, a fixed UTC+10 with no daylight saving: Brisbane's clock.
const MARKET_TIME = "Australia/Brisbane";

const time = new Intl.DateTimeFormat("en-AU", {
  timeZone: MARKET_TIME,
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});
const dayTime = new Intl.DateTimeFormat("en-AU", {
  timeZone: MARKET_TIME,
  weekday: "short",
  day: "numeric",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});
const day = new Intl.DateTimeFormat("en-AU", {
  timeZone: MARKET_TIME,
  day: "numeric",
  month: "short",
  year: "numeric",
});

export const clock = (t: number) => time.format(t);
export const when = (t: number) => `${dayTime.format(t)} AEST`;
export const date = (t: number) => day.format(t);
export const mw = (value: number | null | undefined, digits = 0) =>
  value === null || value === undefined
    ? "–"
    : `${value.toLocaleString("en-AU", { maximumFractionDigits: digits })} MW`;

export function ago(t: number, now: number): string {
  const minutes = Math.max(0, Math.round((now - t) / 60_000));
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours} h ago`;
  return `${Math.round(hours / 24)} days ago`;
}

export const horizonLabel = (h: number) =>
  h % 2 === 0 ? `${h / 2} h` : h === 1 ? "30 min" : `${(h / 2).toFixed(1)} h`;
