"""Replay a schedule through the battery and record what happened.

The simulator is the only place in the project that turns a schedule into state of charge and money.

Infeasible requests are clipped to what the battery could really do, with the
removed energy recorded in ``clipped_mwh``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from bess.asset.battery import BatterySpec

# Flows below this are treated as zero. Guards against float dust in the LP solution being reported as a trade.
TOL_MWH = 1e-9


class Action(StrEnum):
    """Descibes what the battery did in an interval."""

    CHARGE = "charge"
    DISCHARGE = "discharge"
    BOTH = "both"
    IDLE = "idle"


class BindingConstraint(StrEnum):
    """Describes which limit, if any, stopped the battery doing more in an interval.

    This allows for explainability: "it discharged 2.6 MWh of a possible 4.2
    because the tank ran dry" is a complete answer which comes from here.
    """

    NONE = "none"
    CONNECTION = "connection"
    SOC_CEILING = "soc_ceiling"
    SOC_FLOOR = "soc_floor"

# Columns every ``SimulationResult`` is guaranteed to carry.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "price_aud_mwh",
    "charge_mwh",
    "discharge_mwh",
    "net_mwh",
    "requested_charge_mwh",
    "requested_discharge_mwh",
    "clipped_mwh",
    "soc_before_mwh",
    "soc_after_mwh",
    "gross_revenue_aud",
    "degradation_aud",
    "net_revenue_aud",
    "cumulative_net_revenue_aud",
    "action",
    "binding_constraint",
)

_ACTION_DTYPE = pd.CategoricalDtype(categories=[a.value for a in Action])
_CONSTRAINT_DTYPE = pd.CategoricalDtype(categories=[c.value for c in BindingConstraint])


@dataclass(frozen=True)
class SimulationResult:
    """Per-interval record of what the battery did, plus headline totals."""

    intervals: pd.DataFrame
    spec: BatterySpec

    def __post_init__(self) -> None:
        """Check the frame carries what everything downstream expects in REQUIRED_COMMENTS.
        Failing here names the missing column, close to where the mistake was made.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in self.intervals.columns]
        if missing:
            raise ValueError(
                f"SimulationResult is missing required column(s): {', '.join(missing)}"
            )

    @property
    def gross_revenue_aud(self) -> float:
        """Market cashflow: what settlement would pay, before modelled wear."""
        return float(self.intervals["gross_revenue_aud"].sum())

    @property
    def degradation_cost_aud(self) -> float:
        return float(self.intervals["degradation_aud"].sum())

    @property
    def net_revenue_aud(self) -> float:
        """Gross revenue less modelled degradation. This is what the LP maximises."""
        return float(self.intervals["net_revenue_aud"].sum())

    @property
    def energy_bought_mwh(self) -> float:
        return float(self.intervals["charge_mwh"].sum())

    @property
    def energy_sold_mwh(self) -> float:
        return float(self.intervals["discharge_mwh"].sum())

    @property
    def equivalent_full_cycles(self) -> float:
        """Energy discharged, expressed as multiples of a full tank."""
        drained = self.energy_sold_mwh / self.spec.discharge_efficiency
        return drained / self.spec.capacity_mwh

    @property
    def clipped_mwh(self) -> float:
        """Total energy removed from the request to make it physically possible."""
        return float(self.intervals["clipped_mwh"].sum())

    @property
    def simultaneous_flow_intervals(self) -> int:
        """Intervals where the battery both bought and sold.

        Reachable by switching direction inside the 300-second interval, and
        worth doing only at deeply negative prices against a full battery.
        Reported rather than forbidden; the README has the derivation.
        """
        return int((self.intervals["action"] == Action.BOTH).sum())


def simulate(
    spec: BatterySpec,
    prices: pd.Series,
    charge_mwh: np.ndarray,
    discharge_mwh: np.ndarray,
) -> SimulationResult:
    """Apply a schedule to the battery, enforcing its physical limits.

    Args:
        spec: the battery being operated.
        prices: RRP in AUD/MWh, indexed by interval-ending timestamp.
        charge_mwh: requested grid-side energy bought per interval.
        discharge_mwh: requested grid-side energy sold per interval.

    Returns:
        A :class:`SimulationResult` with one row per interval.
    """
    n = len(prices)
    if len(charge_mwh) != n or len(discharge_mwh) != n:
        raise ValueError(
            f"schedule length mismatch: prices={n}, "
            f"charge={len(charge_mwh)}, discharge={len(discharge_mwh)}"
        )

    requested_c = np.asarray(charge_mwh, dtype=float)
    requested_d = np.asarray(discharge_mwh, dtype=float)
    price = prices.to_numpy(dtype=float)

    # A single grid connection, so one shared rating governs flow in either direction. 
    # If the charge and discharge ratings differ, the connection must be sized for the larger of the two.
    connection_limit = max(spec.max_charge_mwh, spec.max_discharge_mwh)

    out_c = np.zeros(n)
    out_d = np.zeros(n)
    soc_before = np.zeros(n)
    soc_after = np.zeros(n)
    binding = np.empty(n, dtype=object)

    soc = spec.initial_soc_mwh

    for i in range(n):
        c = max(0.0, requested_c[i])
        d = max(0.0, requested_d[i])
        reason = BindingConstraint.NONE

        # 1. Per-direction power ratings.
        if c > spec.max_charge_mwh:
            c, reason = spec.max_charge_mwh, BindingConstraint.CONNECTION
        if d > spec.max_discharge_mwh:
            d, reason = spec.max_discharge_mwh, BindingConstraint.CONNECTION

        # 2. Shared connection rating. Scale both legs proportionally so the intended balance between them is preserved.
        total = c + d
        if total > connection_limit + TOL_MWH:
            scale = connection_limit / total
            c, d, reason = c * scale, d * scale, BindingConstraint.CONNECTION

        # 3. The tank must neither overflow nor go negative. The limit applies
        #    to the NET change across the interval, not to each leg on its own:
        #    a sell in the same interval frees room for a buy. Clipping the
        #    legs independently would throw away a buy against a full battery
        #    even when a simultaneous sell had already made space for it.
        closing = soc + spec.charge_efficiency * c - d / spec.discharge_efficiency

        if closing > spec.capacity_mwh:
            excess = closing - spec.capacity_mwh
            c = max(0.0, c - excess / spec.charge_efficiency)
            reason = BindingConstraint.SOC_CEILING
        elif closing < 0.0:
            shortfall = -closing
            d = max(0.0, d - shortfall * spec.discharge_efficiency)
            reason = BindingConstraint.SOC_FLOOR

        stored = spec.charge_efficiency * c
        drained = d / spec.discharge_efficiency

        soc_before[i] = soc
        soc = min(spec.capacity_mwh, max(0.0, soc + stored - drained))
        soc_after[i] = soc

        out_c[i], out_d[i], binding[i] = c, d, reason.value

    clipped = (requested_c - out_c) + (requested_d - out_d)

    gross = price * (out_d - out_c)
    degradation = spec.degradation_cost_per_mwh * (out_c + out_d)

    buying = out_c > TOL_MWH
    selling = out_d > TOL_MWH
    action = np.select(
        [buying & selling, buying, selling],
        [Action.BOTH.value, Action.CHARGE.value, Action.DISCHARGE.value],
        default=Action.IDLE.value,
    )

    intervals = pd.DataFrame(
        {
            "price_aud_mwh": price,
            "charge_mwh": out_c,
            "discharge_mwh": out_d,
            "net_mwh": out_d - out_c,
            "requested_charge_mwh": requested_c,
            "requested_discharge_mwh": requested_d,
            "clipped_mwh": clipped,
            "soc_before_mwh": soc_before,
            "soc_after_mwh": soc_after,
            "gross_revenue_aud": gross,
            "degradation_aud": degradation,
            "net_revenue_aud": gross - degradation,
            # Categorical with the enum as its categories, so an unexpected
            # value becomes NaN here, where it is visible.
            "action": pd.Categorical(action, dtype=_ACTION_DTYPE),
            "binding_constraint": pd.Categorical(binding, dtype=_CONSTRAINT_DTYPE),
        },
        index=prices.index,
    )
    intervals["cumulative_net_revenue_aud"] = intervals["net_revenue_aud"].cumsum()

    return SimulationResult(intervals=intervals, spec=spec)
