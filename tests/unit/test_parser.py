from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from nem_forecast.ingest.parser import (
    ParseError,
    find_missing_intervals,
    parse_aemo_time,
    parse_amount,
    parse_csv_text,
    parse_file,
)
from nem_forecast.schemas import NEM_TZ, Region
from tests.aemo_files import (
    DISPATCH_IS_TIME,
    DISPATCH_IS_ZIP,
    DST_DAY_CSV,
    VIC1_DEMAND,
    dispatch_is_text,
    replace_once,
    with_vic1_demand,
    zipped,
)

INTERVAL_END = datetime(2026, 10, 13, 4, 40, tzinfo=NEM_TZ)


def vic1_line(text: str, table: str) -> str:
    return next(
        line
        for line in text.splitlines()
        if line.startswith(f"D,DISPATCH,{table}") and ",VIC1," in line
    )


class TestDispatchIs:
    def test_reads_demand_and_price_for_every_region(self) -> None:
        result = parse_file(DISPATCH_IS_ZIP, zipped(dispatch_is_text()))

        assert result.skipped_rows == 0
        assert [r.region for r in result.records] == list(Region)
        vic1 = next(r for r in result.records if r.region is Region.VIC1)
        assert vic1.settlement_date == INTERVAL_END
        assert vic1.demand_mw == float(VIC1_DEMAND)
        assert vic1.rrp == 0.01
        assert vic1.source_file == DISPATCH_IS_ZIP

    def test_prices_are_normalised_to_cents(self) -> None:
        # DispatchIS publishes QLD1 at 152.67215 $/MWh; the monthly file says 152.67.
        assert "152.67215" in dispatch_is_text()

        result = parse_csv_text(dispatch_is_text(), "f", regions={Region.QLD1})

        assert result.records[0].rrp == 152.67

    def test_keeps_only_requested_regions(self) -> None:
        result = parse_file(DISPATCH_IS_ZIP, zipped(dispatch_is_text()), regions={Region.VIC1})

        assert [r.region for r in result.records] == [Region.VIC1]

    def test_accepts_bare_csv_as_well_as_zip(self) -> None:
        result = parse_file("report.CSV", dispatch_is_text().encode())

        assert len(result.records) == 5

    def test_ignores_tables_it_does_not_use(self) -> None:
        text = dispatch_is_text()
        assert "D,DISPATCH,INTERCONNECTORRES" in text
        assert "D,DISPATCH,CASE_SOLUTION" in text

        assert parse_csv_text(text, "f").skipped_rows == 0

    def test_columns_are_found_by_name_not_position(self) -> None:
        # A future table version inserting a column before TOTALDEMAND must not shift values.
        text = dispatch_is_text()
        header = next(line for line in text.splitlines() if line.startswith("I,DISPATCH,REGIONSUM"))
        text = replace_once(text, header, header.replace("INTERVENTION,", "INTERVENTION,NEWCOL,"))
        for line in [ln for ln in text.splitlines() if ln.startswith("D,DISPATCH,REGIONSUM")]:
            fields = line.split(",")
            text = replace_once(text, line, ",".join([*fields[:9], "999", *fields[9:]]))

        result = parse_csv_text(text, "f", regions={Region.VIC1})

        assert result.records[0].demand_mw == float(VIC1_DEMAND)
        assert result.skipped_rows == 0

    def test_intervention_run_is_ignored(self) -> None:
        text = dispatch_is_text()
        pricing_run = vic1_line(text, "REGIONSUM")
        physical_run = pricing_run.replace(f",0,{VIC1_DEMAND},", ",1,9999.9,")
        text = replace_once(text, pricing_run, f"{pricing_run}\n{physical_run}")

        result = parse_csv_text(text, "f", regions={Region.VIC1})

        assert [r.demand_mw for r in result.records] == [float(VIC1_DEMAND)]

    def test_missing_price_leaves_rrp_empty(self) -> None:
        text = dispatch_is_text()
        text = replace_once(text, vic1_line(text, "PRICE") + "\n", "")

        result = parse_csv_text(text, "f", regions={Region.VIC1})

        assert result.records[0].rrp is None

    def test_malformed_rows_are_logged_and_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        text = dispatch_is_text()
        good = vic1_line(text, "REGIONSUM")
        nsw = next(line for line in text.splitlines() if line.startswith("D,DISPATCH,REGIONSUM"))
        bad_rows = [
            good.rsplit(",", 3)[0],  # truncated: wrong column count
            good.replace(DISPATCH_IS_TIME, "2026/13/45 04:40:00"),  # impossible date
            good.replace(f",{VIC1_DEMAND},", ",not-a-number,"),
            nsw.replace("NSW1", "WA1"),  # not a NEM region
            "X,DISPATCH,REGIONSUM,9,garbage",  # unknown record type
        ]
        text = replace_once(text, good, "\n".join([good, *bad_rows]))

        with caplog.at_level(logging.WARNING):
            result = parse_csv_text(text, "f")

        assert result.skipped_rows == len(bad_rows)
        assert len(result.records) == 5
        assert len(caplog.records) == len(bad_rows)

    def test_data_row_before_its_header_is_skipped(self) -> None:
        text = dispatch_is_text()
        header = next(line for line in text.splitlines() if line.startswith("I,DISPATCH,REGIONSUM"))
        text = replace_once(text, header + "\n", "")

        result = parse_csv_text(text, "f")

        assert result.records == []
        assert result.skipped_rows == 5

    @pytest.mark.parametrize("bad", ["nan", "inf", "-Infinity", "1e999999999"])
    def test_non_finite_demand_is_skipped(self, bad: str) -> None:
        result = parse_csv_text(with_vic1_demand(dispatch_is_text(), bad), "f")

        assert Region.VIC1 not in {r.region for r in result.records}
        assert result.skipped_rows == 1

    def test_non_csv_zip_members_are_ignored(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = parse_file("x.zip", zipped("readme", member="README.txt"))

        assert result.records == []
        assert "README.txt" in caplog.text

    def test_byte_order_mark_is_tolerated(self) -> None:
        result = parse_file("f.CSV", b"\xef\xbb\xbf" + dispatch_is_text().encode())

        assert len(result.records) == 5


class TestPriceAndDemand:
    def test_reads_a_real_monthly_file(self) -> None:
        result = parse_file(DST_DAY_CSV.name, DST_DAY_CSV.read_bytes())

        assert result.skipped_rows == 0
        assert len(result.records) == 289
        assert {r.region for r in result.records} == {Region.VIC1}
        assert result.records[0].settlement_date == datetime(2025, 10, 5, tzinfo=NEM_TZ)
        assert result.records[-1].settlement_date == datetime(2025, 10, 6, tzinfo=NEM_TZ)
        assert result.records[0].rrp is not None

    def test_market_time_ignores_victorian_daylight_saving(self) -> None:
        """AEMO market time stays at UTC+10 when Victoria moves to daylight saving.

        On 5 Oct 2025 Melbourne clocks jumped from 02:00 to 03:00, so 02:05-02:55 local time
        never happened. The real AEMO file still has those intervals, with no gap, so its
        timestamps cannot be Melbourne local time.
        """
        records = parse_file(DST_DAY_CSV.name, DST_DAY_CSV.read_bytes()).records
        times = [r.settlement_date for r in records]

        assert find_missing_intervals(times, times[0] - timedelta(minutes=5), times[-1]) == []
        assert all(t.utcoffset() == timedelta(hours=10) for t in times)
        assert len([t for t in times if t.day == 5 and t.hour == 2]) == 12

        melbourne = ZoneInfo("Australia/Melbourne")
        two_thirty = next(t for t in times if (t.hour, t.minute) == (2, 30))
        assert two_thirty.astimezone(melbourne).strftime("%H:%M %Z") == "03:30 AEDT"
        assert times[0].astimezone(melbourne).utcoffset() == timedelta(hours=10)
        assert times[-1].astimezone(melbourne).utcoffset() == timedelta(hours=11)

    def test_missing_required_column_rejects_file(self) -> None:
        with pytest.raises(ParseError, match="TOTALDEMAND"):
            parse_csv_text("REGION,SETTLEMENTDATE,RRP\nVIC1,2025/10/05 00:05:00,1\n", "f")

    def test_malformed_rows_are_skipped(self) -> None:
        text = (
            "REGION,SETTLEMENTDATE,TOTALDEMAND,RRP,PERIODTYPE\n"
            "VIC1,2025/10/05 00:05:00,4000.5,12.1,TRADE\n"
            "VIC1,2025/10/05 00:10:00,,12.1,TRADE\n"
            "VIC1,yesterday,4000.5,12.1,TRADE\n"
            "VIC1,2025/10/05 00:20:00\n"
            "VIC1,2025/10/05 00:25:00,4100.0,,TRADE\n"
        )

        result = parse_csv_text(text, "f")

        assert [r.demand_mw for r in result.records] == [4000.5, 4100.0]
        assert result.records[1].rrp is None
        assert result.skipped_rows == 3


def test_unrecognised_format_is_rejected() -> None:
    with pytest.raises(ParseError, match="unrecognised format"):
        parse_csv_text("<html>Service unavailable</html>", "f")


def test_aemo_time_is_utc_plus_10_even_in_summer() -> None:
    assert parse_aemo_time("2026/01/15 12:00:00") == datetime(2026, 1, 15, 2, tzinfo=UTC)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("31.64139", 31.64),
        ("0.00041", 0.0),
        ("-12.145", -12.15),  # half away from zero, not banker's rounding
        ("4660.9", 4660.9),
        (" 17500 ", 17500.0),
    ],
)
def test_parse_amount_rounds_half_up_to_cents(text: str, expected: float) -> None:
    assert parse_amount(text) == expected


@pytest.mark.parametrize("text", [None, "", "  ", "abc"])
def test_parse_amount_rejects_non_numbers(text: str | None) -> None:
    with pytest.raises(ValueError, match=r"empty value|not a number"):
        parse_amount(text)


def test_find_missing_intervals_snaps_an_off_grid_start_to_the_grid() -> None:
    start = datetime(2026, 1, 1, 4, 58, 54, 998246, tzinfo=NEM_TZ)
    grid = [datetime(2026, 1, 1, 5, m, tzinfo=NEM_TZ) for m in (0, 5, 10)]

    assert find_missing_intervals(grid, start, grid[-1]) == []
    assert find_missing_intervals(grid[1:], start, grid[-1]) == [grid[0]]


def test_find_missing_intervals() -> None:
    start = datetime(2026, 1, 1, tzinfo=NEM_TZ)
    present = [start + timedelta(minutes=m) for m in (5, 10, 20, 30)]

    missing = find_missing_intervals(present, start, start + timedelta(minutes=30))

    assert missing == [start + timedelta(minutes=15), start + timedelta(minutes=25)]
