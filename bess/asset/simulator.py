"""Replay a schedule through the battery and work out what actually happened.

The only place in the project that turns a schedule into state of charge and
into money. Strategies propose; the simulator disposes. Every strategy is
therefore scored by identical arithmetic, and the optimiser's own objective can
be checked against an independent recomputation -- they agree only if the LP's
constraint matrix is right.

Infeasible requests are clipped to what the battery could really do, and the
removed energy is recorded. A broken strategy shows up as non-zero
``clipped_mwh``, and its revenue stays physically achievable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from bess.asset.battery import BatterySpec

#: Flows below this are treated as zero. Guards against float dust in the LP
#: solution being reported as a trade.
TOL_MWH = 1e-9


class Action(StrEnum):
    """What the battery did in an interval.

    A ``StrEnum`` so the column holds ordinary strings while code refers to
    ``Action.CHARGE``, so a typo raises ``AttributeError`` at the point it is
    written.
    """

    CHARGE = "charge"
    DISCHARGE = "discharge"
    BOTH = "both"
    IDLE = "idle"


class BindingConstraint(StrEnum):
    """Which limit, if any, stopped the battery doing more in an interval.

    This is the backbone of explainability: "it discharged 2.6 MWh of a possible 4.2
    because the tank ran dry" is a complete answer, and it comes from here.
    """

    NONE = "none"
    CONNECTION = "connection"
    SOC_CEILING = "soc_ceiling"
    SOC_FLOOR = "soc_floor"


#: Columns every ``SimulationResult`` is guaranteed to carry. Reporting, the
#: dashboard and the tests all read against this, so it is checked on
#: construction, where a missing column can still be named.
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
        """Check the frame carries what everything downstream expects.

        A DataFrame has no schema of its own, so without this
        :data:`REQUIRED_COLUMNS` is only a comment. Failing here names the
        missing column, close to where the mistake was made.
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
        profitable only below
        :attr:`BatterySpec.simultaneous_flow_price_threshold`. Reported rather
        than forbidden -- see the README.
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

    # A single grid connection, so one shared rating governs flow in either
    # direction. If the charge and discharge ratings differ, the connection
    # must be sized for the larger of the two.
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

        # 2. Shared connection rating. Scale both legs proportionally so the
        #    intended balance between them is preserved.
        total = c + d
        if total > connection_limit + TOL_MWH:
            scale = connection_limit / total
            c, d, reason = c * scale, d * scale, BindingConstraint.CONNECTION

        # 3. The tank must not overflow. Storing takes eta_c of what is bought.
        headroom = spec.capacity_mwh - soc
        max_c = headroom / spec.charge_efficiency
        if c > max_c:
            c, reason = max(0.0, max_c), BindingConstraint.SOC_CEILING

        # 4. The tank must not go negative. Delivering drains 1/eta_d of what
        #    is sold, so the sellable energy is less than the energy stored.
        max_d = soc * spec.discharge_efficiency
        if d > max_d:
            d, reason = max(0.0, max_d), BindingConstraint.SOC_FLOOR

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
