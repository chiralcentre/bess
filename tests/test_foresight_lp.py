"""Tests for the perfect-hindsight LP.

The load-bearing test is :meth:`test_lp_objective_matches_the_simulator`. The
LP and the simulator compute revenue by completely different routes -- one from
a constraint matrix, the other by replaying the schedule interval by interval.
If the constraint matrix has a sign error or an efficiency the wrong way round,
the two numbers diverge. Agreement validates both.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bess.asset.battery import BatterySpec
from bess.asset.simulator import simulate
from bess.strategies.foresight_lp import ForesightLP, InfeasibleScheduleError


def _prices(values: list[float]) -> pd.Series:
    return pd.Series(
        values, index=pd.date_range("2026-10-08", periods=len(values), freq="5min")
    )


def _day(seed: int = 0, n: int = 288) -> pd.Series:
    """A synthetic day with a solar trough and an evening peak."""
    rng = np.random.default_rng(seed)
    hours = np.arange(n) * 5 / 60
    shape = 90 - 70 * np.cos((hours - 18) * np.pi / 12) ** 4 * 3
    return _prices(list(shape + rng.normal(0, 25, n)))


class TestForesightLP:
    def test_lp_objective_matches_the_simulator(self, spec: BatterySpec) -> None:
        prices = _day()

        schedule = ForesightLP().plan(spec, prices)
        result = simulate(spec, prices, schedule.charge_mwh, schedule.discharge_mwh)

        assert schedule.claimed_net_revenue_aud == pytest.approx(
            result.net_revenue_aud, abs=1e-6
        )

    def test_lp_schedule_is_physically_feasible(self, spec: BatterySpec) -> None:
        """The simulator should never have to clip an LP schedule."""
        prices = _day(seed=3)

        schedule = ForesightLP().plan(spec, prices)
        result = simulate(spec, prices, schedule.charge_mwh, schedule.discharge_mwh)

        assert result.clipped_mwh == pytest.approx(0.0, abs=1e-9)

    def test_finishes_on_the_terminal_state_of_charge(self, spec: BatterySpec) -> None:
        spec = replace(spec, initial_soc_mwh=0.0)
        prices = _day(seed=7)

        schedule = ForesightLP(terminal_soc_mwh=50.0).plan(spec, prices)
        result = simulate(spec, prices, schedule.charge_mwh, schedule.discharge_mwh)

        assert result.intervals["soc_after_mwh"].iloc[-1] == pytest.approx(
            50.0, abs=1e-6
        )

    def test_does_nothing_when_no_spread_clears_the_hurdle(
        self, spec: BatterySpec
    ) -> None:
        """A flat price series has no arbitrage, so the battery should idle."""
        schedule = ForesightLP().plan(spec, _prices([80.0] * 96))

        assert schedule.charge_mwh.sum() == pytest.approx(0.0, abs=1e-6)
        assert schedule.discharge_mwh.sum() == pytest.approx(0.0, abs=1e-6)

    def test_ignores_a_spread_too_narrow_to_cover_losses(
        self, spec: BatterySpec
    ) -> None:
        """81% round-trip needs a ~23.5% spread before a trade is worth doing."""
        spec = replace(spec, degradation_cost_per_mwh=0.0)
        # 100 -> 110 is only a 10% spread: losing money once losses are paid for.
        schedule = ForesightLP().plan(spec, _prices([100.0] * 48 + [110.0] * 48))

        assert schedule.charge_mwh.sum() == pytest.approx(0.0, abs=1e-6)

    def test_takes_a_spread_that_does_clear_the_hurdle(self, spec: BatterySpec) -> None:
        spec = replace(spec, degradation_cost_per_mwh=0.0)
        schedule = ForesightLP().plan(spec, _prices([100.0] * 48 + [300.0] * 48))

        assert schedule.charge_mwh.sum() > 0
        assert schedule.discharge_mwh.sum() > 0

    def test_buys_into_negative_prices(self, spec: BatterySpec) -> None:
        """Being paid to charge needs no special case: the sign does the work."""
        prices = _prices([-50.0] * 60 + [200.0] * 60)

        schedule = ForesightLP().plan(spec, prices)
        result = simulate(spec, prices, schedule.charge_mwh, schedule.discharge_mwh)

        charged = result.intervals[result.intervals.action == "charge"]
        assert (charged["price_aud_mwh"] < 0).all()
        assert charged["gross_revenue_aud"].sum() > 0  # paid to fill up

    def test_water_value_sits_between_the_buy_and_sell_prices(
        self, spec: BatterySpec
    ) -> None:
        """The dual is the battery's internal value of a stored MWh.

        It must lie between what the battery pays and what it sells for,
        otherwise the sign convention is wrong.
        """
        prices = _day(seed=11)

        schedule = ForesightLP().plan(spec, prices)
        result = simulate(spec, prices, schedule.charge_mwh, schedule.discharge_mwh)
        iv = result.intervals

        buy = iv.loc[iv.action == "charge", "price_aud_mwh"].mean()
        sell = iv.loc[iv.action == "discharge", "price_aud_mwh"].mean()
        water = schedule.water_value_aud_mwh

        assert buy < water.mean() < sell

    def test_tradable_mask_blocks_trading(self, spec: BatterySpec) -> None:
        """Gaps in the price data must not be traded on."""
        prices = _prices([-50.0] * 60 + [200.0] * 60)
        tradable = np.ones(len(prices), dtype=bool)
        tradable[:60] = False  # the cheap half is unavailable

        schedule = ForesightLP(tradable=tradable).plan(spec, prices)

        assert schedule.charge_mwh[:60].sum() == pytest.approx(0.0, abs=1e-6)
        assert schedule.discharge_mwh[:60].sum() == pytest.approx(0.0, abs=1e-6)

    def test_unreachable_terminal_state_is_reported_clearly(
        self, spec: BatterySpec
    ) -> None:
        spec = replace(spec, initial_soc_mwh=0.0)
        # Four intervals can move at most ~17 MWh, nowhere near 200.
        with pytest.raises(InfeasibleScheduleError, match="unreachable"):
            ForesightLP(terminal_soc_mwh=200.0).plan(spec, _prices([10.0] * 4))

    def test_rejects_an_empty_price_series(self, spec: BatterySpec) -> None:
        with pytest.raises(ValueError, match="empty"):
            ForesightLP().plan(spec, _prices([]))


class TestSimultaneousFlow:
    """Documents the one behaviour the LP is deliberately not forbidden from.

    At a sufficiently negative price a full battery will buy and sell at once,
    holding its state of charge flat while being paid for the energy its own
    round-trip losses consume. This is physically reachable by switching
    direction inside the interval, so it is priced rather than banned.
    """

    def test_full_battery_burns_energy_at_deeply_negative_prices(
        self, spec: BatterySpec
    ) -> None:
        spec = replace(spec, initial_soc_mwh=200.0, degradation_cost_per_mwh=0.0)
        prices = _prices([-1000.0] * 24)

        schedule = ForesightLP(terminal_soc_mwh=200.0).plan(spec, prices)
        result = simulate(spec, prices, schedule.charge_mwh, schedule.discharge_mwh)

        assert result.simultaneous_flow_intervals > 0
        assert result.gross_revenue_aud > 0

    def test_degradation_cost_suppresses_it_above_the_threshold(
        self, spec: BatterySpec
    ) -> None:
        """Below ~-$47.63/MWh it pays at delta=$5; above it, it does not."""
        spec = replace(spec, initial_soc_mwh=200.0, degradation_cost_per_mwh=5.0)
        assert spec.simultaneous_flow_price_threshold == pytest.approx(-47.63, abs=0.01)

        mild = _prices([-20.0] * 24)  # above the threshold: not worth the wear
        schedule = ForesightLP(terminal_soc_mwh=200.0).plan(spec, mild)
        result = simulate(spec, mild, schedule.charge_mwh, schedule.discharge_mwh)

        assert result.simultaneous_flow_intervals == 0
