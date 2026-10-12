# Architecture

_Placeholder: the architecture diagram and design notes are written in Phase 5, once ingestion,
modelling and serving exist._

Contracts fixed in Phase 0, which later phases build on:

- **Time.** Every timestamp is timezone-aware. AEMO market time is fixed UTC+10
  (`nem_forecast.schemas.NEM_TZ`). Dispatch data sits on 5-minute boundaries and forecasts on
  30-minute boundaries.
- **Domain objects.** `DispatchInterval`, `ForecastRequest`, `ForecastResponse` and
  `ModelMetadata` in `nem_forecast.schemas`.
- **Schema.** Numbered SQL files in `migrations/`, forward-only, checksummed.
- **Config.** `nem_forecast.config.Settings`, reading environment variables only.
