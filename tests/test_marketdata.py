"""Tests for the market data layer.

These run against a hand-built fixture rather than a downloaded file, so they
are fast, offline, and assert on cases that are rare in real data but expensive
to get wrong: a table version changing mid-file, an intervention run duplicating
a timestamp, and a missing interval.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bess.marketdata.mms import MMSParseError, declared_record_count, extract_table
from bess.marketdata.prices import NEM_TIMEZONE, _clean

# A miniature DispatchIS file. Two tables, so the parser has to ignore one, and
# the columns of each are declared only by their I row.
FIXTURE = """C,NEMP.WORLD,DISPATCHIS,AEMO,PUBLIC,2026/10/08,00:00:08,0000000541700377
I,DISPATCH,CASE_SOLUTION,2,SETTLEMENTDATE,RUNNO,INTERVENTION
D,DISPATCH,CASE_SOLUTION,2,"2026/10/08 00:05:00",1,0
I,DISPATCH,PRICE,5,SETTLEMENTDATE,RUNNO,REGIONID,INTERVENTION,RRP,APCFLAG
D,DISPATCH,PRICE,5,"2026/10/08 00:05:00",1,NSW1,0,104.55,0
D,DISPATCH,PRICE,5,"2026/10/08 00:05:00",1,VIC1,0,88.10,0
D,DISPATCH,PRICE,5,"2026/10/08 00:10:00",1,NSW1,0,-25.40,0
C,"END OF REPORT",7
"""


class TestMMSParser:
    def test_extracts_only_the_requested_table(self) -> None:
        table = extract_table(FIXTURE, "DISPATCH", "PRICE")
        assert len(table) == 3
        assert list(table.columns) == [
            "SETTLEMENTDATE", "RUNNO", "REGIONID", "INTERVENTION", "RRP", "APCFLAG",
        ]

    def test_reads_columns_from_the_header_row(self) -> None:
        """Column positions differ between tables and between versions.

        CASE_SOLUTION has three columns here and PRICE has six. Anything that
        assumed a fixed offset would read the wrong field.
        """
        case = extract_table(FIXTURE, "DISPATCH", "CASE_SOLUTION")
        assert list(case.columns) == ["SETTLEMENTDATE", "RUNNO", "INTERVENTION"]

    def test_absent_table_gives_an_empty_frame(self) -> None:
        assert extract_table(FIXTURE, "DISPATCH", "REGIONSUM").empty

    def test_handles_a_version_change_mid_file(self) -> None:
        """AEMO adds columns over time; older rows must not be misaligned."""
        text = (
            'I,DISPATCH,PRICE,5,SETTLEMENTDATE,REGIONID,RRP\n'
            'D,DISPATCH,PRICE,5,"2026/10/08 00:05:00",NSW1,100.0\n'
            'I,DISPATCH,PRICE,6,SETTLEMENTDATE,REGIONID,RRP,NEWFIELD\n'
            'D,DISPATCH,PRICE,6,"2026/10/08 00:10:00",NSW1,110.0,X\n'
        )
        table = extract_table(text, "DISPATCH", "PRICE")
        assert len(table) == 2
        assert list(table["RRP"]) == ["100.0", "110.0"]
        assert pd.isna(table["NEWFIELD"].iloc[0])  # v5 row has no such column

    def test_data_row_before_its_header_is_an_error(self) -> None:
        with pytest.raises(MMSParseError, match="before its header"):
            extract_table('D,DISPATCH,PRICE,5,"2026/10/08 00:05:00",NSW1\n',
                          "DISPATCH", "PRICE")

    def test_reads_the_footer_record_count(self) -> None:
        assert declared_record_count(FIXTURE) == 7


def _raw(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).astype(str)


class TestCleaning:
    def test_timestamps_are_aest_with_no_daylight_saving(self) -> None:
        """The NEM runs on UTC+10 year-round. A named zone would shift twice a year."""
        series = _clean(extract_table(FIXTURE, "DISPATCH", "PRICE")
                        .loc[lambda d: d.REGIONID == "NSW1"], "NSW1")
        assert series.prices.index.tz == NEM_TIMEZONE
        assert series.prices.index[0].utcoffset() == pd.Timedelta(hours=10)

    def test_prefers_the_normal_run_over_an_intervention_run(self) -> None:
        raw = _raw([
            {"SETTLEMENTDATE": "2026/10/08 00:05:00", "REGIONID": "NSW1",
             "RUNNO": 1, "INTERVENTION": 1, "RRP": 999.0},
            {"SETTLEMENTDATE": "2026/10/08 00:05:00", "REGIONID": "NSW1",
             "RUNNO": 1, "INTERVENTION": 0, "RRP": 100.0},
        ])
        series = _clean(raw, "NSW1")
        assert series.prices.iloc[0] == pytest.approx(100.0)
        assert series.quality.intervention_rows_dropped == 1

    def test_falls_back_to_the_intervention_run_when_alone(self) -> None:
        raw = _raw([
            {"SETTLEMENTDATE": "2026/10/08 00:05:00", "REGIONID": "NSW1",
             "RUNNO": 1, "INTERVENTION": 1, "RRP": 500.0},
        ])
        series = _clean(raw, "NSW1")
        assert series.prices.iloc[0] == pytest.approx(500.0)

    def test_keeps_the_latest_solve_of_a_republished_interval(self) -> None:
        raw = _raw([
            {"SETTLEMENTDATE": "2026/10/08 00:05:00", "REGIONID": "NSW1",
             "RUNNO": 1, "INTERVENTION": 0, "RRP": 100.0},
            {"SETTLEMENTDATE": "2026/10/08 00:05:00", "REGIONID": "NSW1",
             "RUNNO": 2, "INTERVENTION": 0, "RRP": 120.0},
        ])
        series = _clean(raw, "NSW1")
        assert series.prices.iloc[0] == pytest.approx(120.0)
        assert series.quality.duplicate_rows_dropped == 1

    def test_a_missing_interval_is_marked_untradable_not_filled(self) -> None:
        """A stale price would produce a trade that never happened."""
        raw = _raw([
            {"SETTLEMENTDATE": "2026/10/08 00:05:00", "REGIONID": "NSW1",
             "RUNNO": 1, "INTERVENTION": 0, "RRP": 100.0},
            # 00:10 is absent.
            {"SETTLEMENTDATE": "2026/10/08 00:15:00", "REGIONID": "NSW1",
             "RUNNO": 1, "INTERVENTION": 0, "RRP": 300.0},
        ])
        series = _clean(raw, "NSW1")

        assert len(series) == 3  # the hole stays in the index
        assert list(series.tradable) == [True, False, True]
        assert series.quality.missing_intervals == 1
        assert series.prices.iloc[1] != 100.0  # not carried forward

    def test_negative_prices_survive_cleaning(self) -> None:
        raw = _raw([
            {"SETTLEMENTDATE": "2026/10/08 00:05:00", "REGIONID": "NSW1",
             "RUNNO": 1, "INTERVENTION": 0, "RRP": -87.5},
        ])
        series = _clean(raw, "NSW1")
        assert series.prices.iloc[0] == pytest.approx(-87.5)
        assert series.quality.negative_price_intervals == 1

    def test_a_price_below_the_market_floor_is_rejected(self) -> None:
        """The floor is fixed in the Rules, so nothing below it was cleared."""
        raw = _raw([
            {"SETTLEMENTDATE": "2026/10/08 00:05:00", "REGIONID": "NSW1",
             "RUNNO": 1, "INTERVENTION": 0, "RRP": -5000.0},
        ])
        with pytest.raises(ValueError, match="market floor"):
            _clean(raw, "NSW1")

    def test_a_scarcity_price_above_an_old_cap_is_accepted(self) -> None:
        """The Market Price Cap is indexed yearly and rises on a legislated
        schedule, so the parser deliberately sets no ceiling. Bounding against
        a stale cap would reject the real prices of a scarcity event.
        """
        raw = _raw([
            {"SETTLEMENTDATE": "2026/10/08 00:05:00", "REGIONID": "NSW1",
             "RUNNO": 1, "INTERVENTION": 0, "RRP": 23200.0},
        ])
        series = _clean(raw, "NSW1")
        assert series.prices.iloc[0] == pytest.approx(23200.0)
