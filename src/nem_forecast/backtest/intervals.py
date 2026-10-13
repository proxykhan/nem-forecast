"""Prediction intervals from a point forecaster's own errors.

The 10th and 90th percentiles of (actual - point), per horizon, are added to the point forecast.
The errors must be out-of-sample for the forecaster that produced them; otherwise the interval
reflects how well it memorised its training data and is far too narrow.

A biased forecaster can have both percentiles on one side of zero. Intervals are clipped so they
always contain the point forecast: p10 <= p50 <= p90.
"""

from __future__ import annotations

import polars as pl

from nem_forecast.features.build import LABEL

LOWER_Q = 0.1
UPPER_Q = 0.9


class ResidualIntervals:
    def __init__(self) -> None:
        self._offsets: pl.DataFrame | None = None

    @property
    def fitted(self) -> bool:
        return self._offsets is not None

    def fit(self, scored: pl.DataFrame, point: str) -> None:
        """Learn offsets from rows holding ``horizon``, the label and the ``point`` forecast."""
        errors = scored.drop_nulls([LABEL, point]).select(
            "horizon", (pl.col(LABEL) - pl.col(point)).alias("error")
        )
        if errors.is_empty():
            raise ValueError("no scored rows to estimate intervals from")
        self._offsets = errors.group_by("horizon").agg(
            pl.col("error").quantile(LOWER_Q).alias("_lower"),
            pl.col("error").quantile(UPPER_Q).alias("_upper"),
        )

    def apply(self, frame: pl.DataFrame, point: str) -> pl.DataFrame:
        """``frame`` with p10, p50 and p90 added around its ``point`` column."""
        if self._offsets is None:
            raise RuntimeError("fit the intervals before applying them")
        p50 = pl.col(point)
        return (
            frame.join(self._offsets, on="horizon", how="left")
            .with_columns(
                p50=p50,
                p10=pl.min_horizontal(p50 + pl.col("_lower"), p50),
                p90=pl.max_horizontal(p50 + pl.col("_upper"), p50),
            )
            .drop("_lower", "_upper")
        )

    def to_dict(self) -> dict[str, list[float]]:
        if self._offsets is None:
            raise RuntimeError("fit the intervals before saving them")
        rows = self._offsets.sort("horizon")
        return {
            "horizon": rows["horizon"].cast(pl.Float64).to_list(),
            "lower": rows["_lower"].to_list(),
            "upper": rows["_upper"].to_list(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> ResidualIntervals:
        intervals = cls()
        intervals._offsets = pl.DataFrame(
            {
                "horizon": [int(h) for h in data["horizon"]],
                "_lower": data["lower"],
                "_upper": data["upper"],
            },
            schema={"horizon": pl.Int16, "_lower": pl.Float64, "_upper": pl.Float64},
        )
        return intervals
