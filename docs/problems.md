# API problem types

Every error from the forecast API is an [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457)
problem: a JSON body with media type `application/problem+json` whose `type` is a link to one of
the sections below. `title` is fixed for each type; `detail` explains this occurrence; `instance`
is the request path. Some types add fields, listed with them.

```json
{
  "type": "https://github.com/proxykhan/nem-forecast/blob/main/docs/problems.md#horizon-not-supported",
  "title": "Horizon beyond what the model was evaluated for",
  "status": 422,
  "detail": "The served model (lightgbm_quantile fa90c4b4c9a3) was trained and backtested for up to 48 half-hours ahead; 60 was requested.",
  "instance": "/forecast",
  "max_horizon": 48
}
```

## Request problems (422)

### invalid-request

The body failed schema validation: a missing or unknown field, a timestamp without a time zone or
off the 30-minute grid, a horizon outside 1-96, or a region that is not in the NEM. The `errors`
field lists each failure with its `location`, `message` and `type`.

### region-not-supported

The region is a NEM region, but the served model forecasts another one. `supported_regions`
lists what it does serve.

### origin-in-future

`forecast_origin` is later than the current time. A forecast can only start from data that
exists.

### origin-too-old

`forecast_origin` is more than 7 days in the past.

### horizon-not-supported

The horizon is within the API's limit of 96 half-hours but beyond what the served model was
trained and backtested for. `max_horizon` gives the model's limit. The API will not return
forecasts nobody has evaluated.

### origin-not-ingested

The origin is valid, but demand for the half-hour ending at it has not been ingested yet (the
scheduled ingest runs every half-hour). `latest_available` gives the latest complete half-hour;
forecast from that, or use `GET /forecast/latest`.

## Service problems (503)

### model-loading

The service has just started and is still downloading or loading its model. Retry after the
number of seconds in the `Retry-After` header.

### model-unavailable

No model could be loaded. `GET /ready` reports why (no model configured, or the download failed).

### prediction-not-recorded

The forecast was computed but could not be written to the `predictions` table, so it is not
returned: every forecast the API serves must be scorable later. Retry after `Retry-After`
seconds.

### no-recent-data

There is no demand data for the region in the last 29 days, so there is nothing to forecast from.

## Other problems

### not-found

No such endpoint (404).

### method-not-allowed

The endpoint exists but not for this HTTP method (405).

### http-error

Any other HTTP-level error.

### internal-error

An unexpected failure (500). The details are logged on the server and never returned.
