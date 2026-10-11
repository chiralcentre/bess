"""Tests for the dashboard figures.

Charts are easy to break silently -- a renamed column leaves an empty panel
rather than an error. These build every figure from a real backtest and check
the traces carry data, so a broken chart fails in CI instead of in front of a
reviewer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bess.asset.battery import BatterySpec
from bess.backtest import run_backtest
from bess.dashboard.charts import (
    daily_revenue_figure,
    day_figure,
    price_duration_figure,
)
from bess.marketdata.prices import NEM_TIMEZONE, DataQualityReport, PriceSeries
from bess.reporting.metrics import daily_summary
from bess.strategies.foresight_lp import ForesightLP


@pytest.fixture
def backtest(spec: BatterySpec):
    """Two full trading days: cheap overnight, expensive evening."""
    values = ([20.0] * 144 + [200.0] * 144) * 2
    index = pd.date_range(
        "2026-09-09 00:05", periods=len(values), freq="5min", tz=NEM_TIMEZONE
    )
    prices = pd.Series(values, index=index)
    quality = DataQualityReport(
        region="NSW1", start=index[0], end=index[-1],
        intervals_expected=len(values), intervals_present=len(values),
        intervention_rows_dropped=0, duplicate_rows_dropped=0,
        administered_price_intervals=0, market_suspended_intervals=0,
        negative_price_intervals=0, price_min=20.0, price_max=200.0,
        price_mean=110.0,
    )
    series = PriceSeries(prices, np.ones(len(values), dtype=bool), quality)
    return run_backtest(spec, series, ForesightLP(), 4)


class TestDayFigure:
    def test_has_all_three_panels_with_data(self, backtest) -> None:
        day = backtest.intervals["trading_day"].iloc[0]
        frame = backtest.intervals[backtest.intervals["trading_day"] == day]

        figure = day_figure(frame)
        names = [t.name for t in figure.data]

        assert "RRP" in names
        assert "Internal value of stored energy" in names
        assert "State of charge" in names
        assert "Cumulative revenue" in names
        assert all(len(t.y) == len(frame) for t in figure.data)

    def test_hover_text_explains_each_interval(self, backtest) -> None:
        day = backtest.intervals["trading_day"].iloc[0]
        frame = backtest.intervals[backtest.intervals["trading_day"] == day]

        figure = day_figure(frame)
        rrp = next(t for t in figure.data if t.name == "RRP")

        assert len(rrp.customdata) == len(frame)
        assert any("Bought" in text for text in rrp.customdata)
        assert any("Sold" in text for text in rrp.customdata)

    def test_shades_one_band_per_block_not_per_interval(self, backtest) -> None:
        """288 overlapping rectangles would look wrong and render slowly."""
        day = backtest.intervals["trading_day"].iloc[0]
        frame = backtest.intervals[backtest.intervals["trading_day"] == day]

        figure = day_figure(frame)
        bands = [s for s in figure.layout.shapes if s.type == "rect"]
        active = (frame["action"] != "idle").sum()

        assert 0 < len(bands) < active

    def test_survives_a_day_with_no_trades(self, spec: BatterySpec) -> None:
        index = pd.date_range(
            "2026-09-09 04:05", periods=288, freq="5min", tz=NEM_TIMEZONE
        )
        idle = pd.DataFrame(
            {
                "price_aud_mwh": 80.0, "charge_mwh": 0.0, "discharge_mwh": 0.0,
                "soc_after_mwh": 0.0, "net_revenue_aud": 0.0,
                "action": "idle", "binding_constraint": "none",
                "water_value_aud_mwh": np.nan,
            },
            index=index,
        )
        figure = day_figure(idle)
        assert len([s for s in figure.layout.shapes if s.type == "rect"]) == 0


class TestOverviewFigures:
    def test_daily_revenue_has_a_bar_per_day(self, backtest) -> None:
        daily = daily_summary(backtest.intervals)
        figure = daily_revenue_figure(daily)

        assert len(figure.data[0].x) == len(daily)
        assert len(figure.layout.shapes) == 1  # the median line

    def test_price_duration_is_sorted_descending(self, backtest) -> None:
        prices = backtest.intervals["price_aud_mwh"]
        figure = price_duration_figure(prices)
        values = np.asarray(figure.data[0].y)

        assert np.all(np.diff(values) <= 1e-9)
        assert values[0] == pytest.approx(prices.max())
        assert values[-1] == pytest.approx(prices.min())
