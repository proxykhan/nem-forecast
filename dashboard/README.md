# nem-forecast dashboard

The monitoring view of [nem-forecast](../README.md): the latest VIC1 demand forecast with its
80% interval against actuals as they land, realised error by horizon over the last 30 days
against the backtest, the daily drift and alert checks, and the last retraining decision.

A Next.js app that reads Neon directly on the server, through a read-only login, and regenerates
the page at most every five minutes. It never calls the forecast API, so a visit never waits for
the free Render instance to wake up.

## Run it locally

```bash
npm ci
npm run dev:root-env   # reads DATABASE_URL from the repository's ../.env
```

or put `DATABASE_URL` in `dashboard/.env.local` and use `npm run dev`.

## Deploy (Vercel, free tier)

1. Create the read-only login: `uv run nem-forecast grant-reader` from the repository root. It
   prints a connection string once.
2. Import the repository on Vercel with **Root Directory** `dashboard`, and add that string as
   the `DATABASE_URL` environment variable.
