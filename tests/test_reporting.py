"""Tests for trading-day splitting, the trade log and the metrics.

The reporting layer is where a wrong number becomes a wrong decision, so the
tests here check the two things most likely to be silently off: the
interval-ending boundary when days are cut, and whether the trade log still
adds up to what the simulator said.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from bess.asset.battery import BatterySpec
from bess.asset.simulator import Action, simulate
from bess.backtest import assign_trading_days, run_backtest
from bess.marketdata.prices import NEM_TIMEZONE, DataQualityReport, PriceSeries
from bess.reporting.metrics import daily_summary, summarise
from bess.reporting.trades import build_trade_log, explain_interval
from bess.strategies.foresight_lp import ForesightLP

FIVE_MIN = timedelta(minutes=5)


def _series(values: list[float], start: str = "2026-09-09 00:05") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="5min", tz=NEM_TIMEZONE)
    return pd.Series(values, index=index)


def _price_series(values: list[float], start: str = "2026-09-09 00:05") -> PriceSeries:
    prices = _series(values, start)
    quality = DataQualityReport(
        region="NSW1", start=prices.index[0], end=prices.index[-1],
        intervals_expected=len(prices), intervals_present=len(prices),
        intervention_rows_dropped=0, duplicate_rows_dropped=0,
        administered_price_intervals=0, market_suspended_intervals=0,
        negative_price_intervals=int((prices < 0).sum()),
        price_min=float(prices.min()), price_max=float(prices.max()),
        price_mean=float(prices.mean()),
    )
    return PriceSeries(
        prices=prices, tradable=np.ones(len(prices), dtype=bool), quality=quality
    )


def _two_days() -> list[float]:
    """Two full trading days: cheap overnight, expensive evening."""
    day = [20.0] * 144 + [200.0] * 144
    return day * 2


class TestTradingDayBoundary:
    def test_interval_ending_at_the_boundary_belongs_to_the_closing_day(self) -> None:
        """04:00 covers 03:55-04:00, so it ends the old day, not starts the new.

        Getting this wrong puts exactly one interval of every day in the wrong
        bucket, which is small enough to miss and large enough to matter.
        """
        index = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-09-10 03:55", tz=NEM_TIMEZONE),
                pd.Timestamp("2026-09-10 04:00", tz=NEM_TIMEZONE),
                pd.Timestamp("2026-09-10 04:05", tz=NEM_TIMEZONE),
            ]
        )
        days = assign_trading_days(index, start_hour=4, interval=FIVE_MIN)

        assert days.iloc[0].date() == pd.Timestamp("2026-09-09").date()
        assert days.iloc[1].date() == pd.Timestamp("2026-09-09").date()
        assert days.iloc[2].date() == pd.Timestamp("2026-09-10").date()

    def test_midnight_boundary_is_configurable(self) -> None:
        index = pd.DatetimeIndex([pd.Timestamp("2026-09-10 02:00", tz=NEM_TIMEZONE)])
        at_four = assign_trading_days(index, 4, FIVE_MIN).iloc[0]
        at_midnight = assign_trading_days(index, 0, FIVE_MIN).iloc[0]

        assert at_four.date() == pd.Timestamp("2026-09-09").date()
        assert at_midnight.date() == pd.Timestamp("2026-09-10").date()


class TestRunBacktest:
    def test_drops_partial_days(self, spec: BatterySpec) -> None:
        """A part-day's revenue is not comparable with a whole day's."""
        prices = _price_series(_two_days())
        result = run_backtest(spec, prices, ForesightLP(), trading_day_start_hour=4)

        assert result.partial_days_dropped > 0
        counts = result.intervals.groupby("trading_day").size()
        assert (counts == 288).all()

    def test_every_day_starts_and_ends_empty(self, spec: BatterySpec) -> None:
        prices = _price_series(_two_days())
        result = run_backtest(spec, prices, ForesightLP(), 4)

        closing = result.intervals.groupby("trading_day")["soc_after_mwh"].last()
        assert closing.abs().max() == pytest.approx(0.0, abs=1e-6)

    def test_cumulative_revenue_is_continuous_across_days(
        self, spec: BatterySpec
    ) -> None:
        prices = _price_series(_two_days())
        result = run_backtest(spec, prices, ForesightLP(), 4)

        total = result.intervals["net_revenue_aud"].sum()
        assert result.intervals["cumulative_net_revenue_aud"].iloc[-1] == pytest.approx(
            total, abs=1e-6
        )

    def test_a_window_with_no_complete_day_is_an_error(
        self, spec: BatterySpec
    ) -> None:
        with pytest.raises(ValueError, match="no complete trading days"):
            run_backtest(spec, _price_series([50.0] * 10), ForesightLP(), 4)


class TestTradeLog:
    def test_collapses_consecutive_intervals_into_one_trade(
        self, spec: BatterySpec
    ) -> None:
        prices = _series([-10.0] * 6 + [500.0] * 6)
        e = spec.max_charge_mwh
        charge = np.array([e] * 6 + [0.0] * 6)
        discharge = np.array([0.0] * 6 + [e] * 6)

        log = build_trade_log(simulate(spec, prices, charge, discharge).intervals)

        assert len(log) == 2
        assert log["action"].tolist() == [Action.CHARGE, Action.DISCHARGE]
        # Six intervals of buying fill the tank with 22.5 MWh, but selling
        # drains 1/eta_d per MWh delivered, so it empties in five rather than
        # six. The round trip is asymmetric: you always buy for longer than you
        # sell, and the sixth interval has nothing left to sell.
        assert log["intervals"].tolist() == [6, 5]

    def test_trade_totals_match_the_simulator(self, spec: BatterySpec) -> None:
        """The log is a view of the same money, not a second calculation."""
        prices = _price_series(_two_days())
        result = run_backtest(spec, prices, ForesightLP(), 4)
        log = build_trade_log(result.intervals)

        assert log["net_revenue_aud"].sum() == pytest.approx(
            result.intervals["net_revenue_aud"].sum(), abs=1e-6
        )
        assert log["energy_mwh"].sum() == pytest.approx(
            result.intervals["charge_mwh"].sum()
            + result.intervals["discharge_mwh"].sum(),
            abs=1e-6,
        )

    def test_a_trade_never_spans_two_trading_days(self, spec: BatterySpec) -> None:
        prices = _price_series(_two_days())
        result = run_backtest(spec, prices, ForesightLP(), 4)
        log = build_trade_log(result.intervals)

        for _, trade in log.iterrows():
            same_day = result.intervals.loc[trade["start"] : trade["end"], "trading_day"]
            assert same_day.nunique() == 1

    def test_idle_intervals_produce_no_trades(self, spec: BatterySpec) -> None:
        prices = _series([50.0] * 12)
        log = build_trade_log(
            simulate(spec, prices, np.zeros(12), np.zeros(12)).intervals
        )
        assert log.empty

    def test_reason_names_the_binding_constraint(self, spec: BatterySpec) -> None:
        prices = _series([-10.0] * 6 + [500.0] * 100)
        e = spec.max_charge_mwh
        charge = np.array([e] * 6 + [0.0] * 100)
        discharge = np.array([0.0] * 6 + [e] * 100)

        log = build_trade_log(simulate(spec, prices, charge, discharge).intervals)
        sell = log[log["action"] == Action.DISCHARGE].iloc[0]

        assert "emptying" in sell["reason"]

    def test_explain_interval_reads_as_a_sentence(self, spec: BatterySpec) -> None:
        prices = _series([-25.0])
        result = simulate(
            spec, prices, np.array([spec.max_charge_mwh]), np.array([0.0])
        )
        text = explain_interval(result.intervals.iloc[0])

        assert "Bought" in text and "$-25.00" in text and "SoC" in text


class TestMetrics:
    def test_average_prices_are_volume_weighted(self, spec: BatterySpec) -> None:
        """A 0.1 MWh interval must not count the same as a full one."""
        prices = _series([10.0, 100.0, 500.0])
        e = spec.max_charge_mwh
        # Buy a lot at $10 and a sliver at $100: the VWAP must sit near $10.
        charge = np.array([e, 0.01, 0.0])
        discharge = np.array([0.0, 0.0, e])

        metrics = summarise(
            simulate(spec, prices, charge, discharge).intervals, spec
        )
        assert metrics.average_buy_price == pytest.approx(10.2, abs=0.5)

    def test_gross_and_net_differ_by_degradation(self, spec: BatterySpec) -> None:
        prices = _price_series(_two_days())
        result = run_backtest(spec, prices, ForesightLP(), 4)
        metrics = summarise(result.intervals, spec)

        assert metrics.gross_revenue_aud - metrics.degradation_cost_aud == pytest.approx(
            metrics.net_revenue_aud, abs=1e-6
        )

    def test_spread_ratio_survives_a_negative_average_buy_price(
        self, spec: BatterySpec
    ) -> None:
        """Being paid to charge is normal in NSW1 and must not break a ratio."""
        prices = _series([-50.0] * 6 + [300.0] * 6)
        e = spec.max_charge_mwh
        metrics = summarise(
            simulate(
                spec, prices, np.array([e] * 6 + [0.0] * 6),
                np.array([0.0] * 6 + [e] * 6),
            ).intervals,
            spec,
        )

        assert metrics.average_buy_price < 0
        assert metrics.spread_ratio == float("inf")
        assert metrics.captured_spread == pytest.approx(350.0, abs=1.0)

    def test_daily_summary_has_one_row_per_trading_day(
        self, spec: BatterySpec
    ) -> None:
        prices = _price_series(_two_days())
        result = run_backtest(spec, prices, ForesightLP(), 4)
        daily = daily_summary(result.intervals)

        assert len(daily) == result.trading_days
        assert (daily["spread"] >= 0).all()

    def test_summarising_nothing_is_an_error(self, spec: BatterySpec) -> None:
        with pytest.raises(ValueError, match="empty"):
            summarise(pd.DataFrame(), spec)


class TestHorizons:
    """Daily resets against one continuous optimisation.

    The daily reset is bookkeeping, not physics: a real battery does not empty
    itself at 04:00. These tests pin down what changes when it is lifted.
    """

    def test_continuous_never_earns_less_than_daily(self, spec: BatterySpec) -> None:
        """A property, not a fixture.

        Every daily-reset schedule is feasible in the continuous problem -- it
        simply happens to touch zero at each boundary -- so the continuous
        optimum can never be worse. If this fails, the continuous formulation
        has lost a degree of freedom somewhere.
        """
        prices = _price_series(_two_days())

        daily = run_backtest(spec, prices, ForesightLP(), 4, horizon="daily")
        continuous = run_backtest(spec, prices, ForesightLP(), 4, horizon="continuous")

        assert continuous.net_revenue_aud >= daily.net_revenue_aud - 1e-6

    def test_both_horizons_cover_the_same_intervals(self, spec: BatterySpec) -> None:
        """Otherwise the revenue comparison is measuring two different windows."""
        prices = _price_series(_two_days())

        daily = run_backtest(spec, prices, ForesightLP(), 4, horizon="daily")
        continuous = run_backtest(spec, prices, ForesightLP(), 4, horizon="continuous")

        assert daily.intervals.index.equals(continuous.intervals.index)
        assert daily.trading_days == continuous.trading_days

    def test_daily_closes_every_day_empty(self, spec: BatterySpec) -> None:
        prices = _price_series(_two_days())
        result = run_backtest(spec, prices, ForesightLP(), 4, horizon="daily")

        closing = result.intervals.groupby("trading_day")["soc_after_mwh"].last()
        assert closing.abs().max() == pytest.approx(0.0, abs=1e-6)
        assert result.days_are_round_trips

    def test_continuous_carries_charge_across_the_boundary(
        self, spec: BatterySpec
    ) -> None:
        """The whole point: energy bought on one day can be sold on the next."""
        # Cheap all through day one, expensive all through day two, so the only
        # profitable trade spans the boundary.
        values = [10.0] * 288 + [400.0] * 288
        prices = _price_series(values + values[:288])
        result = run_backtest(spec, prices, ForesightLP(), 4, horizon="continuous")

        closing = result.intervals.groupby("trading_day")["soc_after_mwh"].last()
        assert (closing > 1.0).any()
        assert not result.days_are_round_trips

    def test_continuous_still_finishes_empty(self, spec: BatterySpec) -> None:
        """Only the ends are pinned, so nothing is liquidated that was not bought."""
        prices = _price_series(_two_days())
        result = run_backtest(spec, prices, ForesightLP(), 4, horizon="continuous")

        assert result.intervals["soc_after_mwh"].iloc[-1] == pytest.approx(
            0.0, abs=1e-6
        )

    def test_unknown_horizon_is_rejected(self, spec: BatterySpec) -> None:
        with pytest.raises(ValueError, match="horizon must be"):
            run_backtest(
                spec, _price_series(_two_days()), ForesightLP(), 4, horizon="weekly"
            )

    def test_daily_is_the_default(self, spec: BatterySpec) -> None:
        prices = _price_series(_two_days())
        assert run_backtest(spec, prices, ForesightLP(), 4).horizon == "daily"
