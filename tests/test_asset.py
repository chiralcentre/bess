"""Tests for the battery spec and the simulator.

The simulator is the only place revenue is computed, so these tests are the
foundation everything else rests on. They check arithmetic that was worked by
hand first, and invariants that must hold for any schedule at all.

The ``spec`` fixture is the real configuration from ``config.toml``; variants
use :func:`dataclasses.replace`, which re-runs validation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bess.asset.battery import BatterySpec
from bess.asset.simulator import Action, BindingConstraint, simulate


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-10-08", periods=n, freq="5min")


class TestBatterySpec:
    def test_derived_quantities_match_the_brief(self, spec: BatterySpec) -> None:
        assert spec.duration_hours == pytest.approx(4.0)
        assert spec.max_charge_mwh == pytest.approx(50.0 / 12)
        assert spec.round_trip_efficiency == pytest.approx(0.81)
        # 1/0.81: a ~23.5% spread is needed before a trade breaks even.
        assert spec.breakeven_price_ratio == pytest.approx(1.2346, abs=1e-4)

    def test_burn_threshold_matches_the_closed_form(self, spec: BatterySpec) -> None:
        # -delta * (1 + rte) / (1 - rte) = -5 * 1.81 / 0.19
        tuned = replace(spec, degradation_cost_per_mwh=5.0)
        assert tuned.simultaneous_flow_price_threshold == pytest.approx(-47.63, abs=0.01)

    def test_a_lossless_battery_never_profits_from_burning_energy(
        self, spec: BatterySpec
    ) -> None:
        perfect = replace(spec, charge_efficiency=1.0, discharge_efficiency=1.0)
        assert perfect.simultaneous_flow_price_threshold == float("-inf")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"capacity_mwh": 0},
            {"capacity_mwh": -200},
            {"max_charge_mw": 0},
            {"charge_efficiency": 1.5},
            {"discharge_efficiency": 0},
            {"initial_soc_mwh": 500},
            {"degradation_cost_per_mwh": -1},
            {"interval_minutes": 0},
        ],
    )
    def test_rejects_impossible_specs(self, spec: BatterySpec, kwargs: dict) -> None:
        """Values arrive from TOML, CLI flags and dashboard sliders, not just defaults."""
        with pytest.raises(ValueError):
            replace(spec, **kwargs)


class TestSimulator:
    def test_hand_worked_example(self, spec: BatterySpec) -> None:
        """Buy into two negative intervals, sell into two expensive ones.

        Worked by hand before the code existed: buy 8.333 MWh, sell 6.750 MWh
        (81% of it), for $1,433.33. The final interval is cut short because the
        tank empties, not because of the price.
        """
        spec = replace(spec, degradation_cost_per_mwh=0.0)
        prices = pd.Series([-10.0, -10.0, 200.0, 200.0], index=_index(4))
        e = spec.max_charge_mwh

        r = simulate(spec, prices, np.array([e, e, 0, 0]), np.array([0, 0, e, e]))

        assert r.energy_bought_mwh == pytest.approx(8.3333, abs=1e-3)
        assert r.energy_sold_mwh == pytest.approx(6.75, abs=1e-3)
        assert r.gross_revenue_aud == pytest.approx(1433.33, abs=0.01)
        assert r.intervals["binding_constraint"].iloc[3] == BindingConstraint.SOC_FLOOR

    def test_charging_at_a_negative_price_earns_money(self, spec: BatterySpec) -> None:
        spec = replace(spec, degradation_cost_per_mwh=0.0)
        prices = pd.Series([-100.0], index=_index(1))
        e = spec.max_charge_mwh

        r = simulate(spec, prices, np.array([e]), np.array([0.0]))

        assert r.gross_revenue_aud > 0
        assert r.gross_revenue_aud == pytest.approx(100.0 * e)

    def test_energy_conservation(self, spec: BatterySpec) -> None:
        """Every MWh sold must be traceable to a MWh bought, net of losses."""
        rng = np.random.default_rng(0)
        n = 200
        prices = pd.Series(rng.normal(80, 150, n), index=_index(n))
        e = spec.max_charge_mwh

        r = simulate(spec, prices, rng.uniform(0, e, n), rng.uniform(0, e, n))

        stored = spec.charge_efficiency * r.energy_bought_mwh
        drained = r.energy_sold_mwh / spec.discharge_efficiency
        final_soc = r.intervals["soc_after_mwh"].iloc[-1]
        assert spec.initial_soc_mwh + stored - drained == pytest.approx(
            final_soc, abs=1e-6
        )

    def test_soc_stays_within_bounds_for_any_request(self, spec: BatterySpec) -> None:
        """Even a deliberately impossible schedule cannot break the physics."""
        rng = np.random.default_rng(1)
        n = 300
        prices = pd.Series(rng.normal(0, 300, n), index=_index(n))

        # Request far more than the connection or the tank could ever allow.
        r = simulate(spec, prices, np.full(n, 999.0), np.full(n, 999.0))

        soc = r.intervals["soc_after_mwh"]
        assert soc.min() >= -1e-9
        assert soc.max() <= spec.capacity_mwh + 1e-9
        assert (r.intervals["charge_mwh"] + r.intervals["discharge_mwh"]).max() <= (
            spec.max_charge_mwh + 1e-9
        )
        assert r.clipped_mwh > 0  # and it tells us it had to intervene

    def test_feasible_schedule_is_never_clipped(self, spec: BatterySpec) -> None:
        n = 48
        prices = pd.Series(np.full(n, 50.0), index=_index(n))
        # Charge gently for the whole window: always within every limit.
        r = simulate(spec, prices, np.full(n, spec.max_charge_mwh), np.zeros(n))
        assert r.clipped_mwh == pytest.approx(0.0, abs=1e-9)

    def test_degradation_reduces_net_but_not_gross(self, spec: BatterySpec) -> None:
        # Needs energy in the tank: a battery that starts empty cannot sell.
        spec = replace(spec, degradation_cost_per_mwh=5.0, initial_soc_mwh=100.0)
        prices = pd.Series([100.0], index=_index(1))
        e = spec.max_discharge_mwh

        r = simulate(spec, prices, np.array([0.0]), np.array([e]))

        assert r.gross_revenue_aud == pytest.approx(100.0 * e)
        assert r.degradation_cost_aud == pytest.approx(5.0 * e)
        assert r.net_revenue_aud == pytest.approx(95.0 * e)

    def test_rejects_mismatched_schedule_length(self, spec: BatterySpec) -> None:
        prices = pd.Series([10.0, 20.0], index=_index(2))
        with pytest.raises(ValueError, match="length mismatch"):
            simulate(spec, prices, np.array([1.0]), np.array([0.0, 0.0]))


class TestResultSchema:
    """The DataFrame is a public interface, so it is checked on construction."""

    def test_every_promised_column_is_present(self, spec: BatterySpec) -> None:
        from bess.asset.simulator import REQUIRED_COLUMNS

        prices = pd.Series([50.0] * 4, index=_index(4))
        r = simulate(spec, prices, np.zeros(4), np.zeros(4))
        assert set(REQUIRED_COLUMNS) <= set(r.intervals.columns)

    def test_a_frame_missing_a_column_is_rejected(self, spec: BatterySpec) -> None:
        from bess.asset.simulator import SimulationResult

        with pytest.raises(ValueError, match="missing required column"):
            SimulationResult(intervals=pd.DataFrame({"price_aud_mwh": [1.0]}), spec=spec)

    def test_action_values_come_from_the_enum(self, spec: BatterySpec) -> None:
        prices = pd.Series([-10.0, 200.0], index=_index(2))
        e = spec.max_charge_mwh
        r = simulate(spec, prices, np.array([e, 0.0]), np.array([0.0, 0.0]))

        assert r.intervals["action"].iloc[0] == Action.CHARGE
        assert r.intervals["action"].iloc[1] == Action.IDLE
        assert set(r.intervals["action"].cat.categories) == {a.value for a in Action}
