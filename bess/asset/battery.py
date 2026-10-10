"""Physical description of the battery.

Everything downstream -- the optimiser, the simulator, the dashboard -- reads
its limits from a single ``BatterySpec``. Nothing else hard-codes a number.

Sign and unit conventions used throughout the project:

* Energy flows (``charge_mwh``, ``discharge_mwh``) are **grid-side**: what the
  meter at the connection point sees, and therefore what is paid for.
* State of charge is **inside the battery**.
* The efficiencies are the exchange rate between those two places, and they
  always cost you energy: you store less than you buy, and you must drain more
  than you deliver.
"""

from __future__ import annotations

from dataclasses import dataclass

MINUTES_PER_HOUR = 60


@dataclass(frozen=True)
class BatterySpec:
    """Physical and commercial limits of the battery.

    Defaults describe the asset in the brief: a 200 MWh / 50 MW system, i.e. a
    four-hour battery, starting empty.
    """

    capacity_mwh: float = 200.0
    max_charge_mw: float = 50.0
    max_discharge_mw: float = 50.0
    charge_efficiency: float = 0.90
    discharge_efficiency: float = 0.90
    initial_soc_mwh: float = 0.0
    degradation_cost_per_mwh: float = 5.0
    interval_minutes: int = 5

    def __post_init__(self) -> None:
        if self.capacity_mwh <= 0:
            raise ValueError("capacity_mwh must be positive")
        if self.max_charge_mw <= 0 or self.max_discharge_mw <= 0:
            raise ValueError("power ratings must be positive")
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("charge_efficiency must be in (0, 1]")
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError("discharge_efficiency must be in (0, 1]")
        if not 0 <= self.initial_soc_mwh <= self.capacity_mwh:
            raise ValueError("initial_soc_mwh must lie within [0, capacity_mwh]")
        if self.degradation_cost_per_mwh < 0:
            raise ValueError("degradation_cost_per_mwh must not be negative")
        if self.interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive")

    # -- derived quantities -------------------------------------------------

    @property
    def interval_hours(self) -> float:
        """Length of one dispatch interval, in hours (5 min -> 1/12 h)."""
        return self.interval_minutes / MINUTES_PER_HOUR

    @property
    def max_charge_mwh(self) -> float:
        """Most energy that can be bought in a single interval (grid-side)."""
        return self.max_charge_mw * self.interval_hours

    @property
    def max_discharge_mwh(self) -> float:
        """Most energy that can be sold in a single interval (grid-side)."""
        return self.max_discharge_mw * self.interval_hours

    @property
    def round_trip_efficiency(self) -> float:
        """Fraction of bought energy that comes back out. 0.9 * 0.9 = 0.81."""
        return self.charge_efficiency * self.discharge_efficiency

    @property
    def duration_hours(self) -> float:
        """Hours of output at full power from full. 200 / 50 = a 4-hour battery."""
        return self.capacity_mwh / self.max_discharge_mw

    @property
    def breakeven_price_ratio(self) -> float:
        """Sell price must exceed buy price by this factor to break even.

        At 81% round-trip you buy 1/0.81 = 1.235 MWh for every MWh you sell, so
        a spread narrower than ~23.5% destroys value before degradation is even
        considered.
        """
        return 1.0 / self.round_trip_efficiency

    @property
    def simultaneous_flow_price_threshold(self) -> float:
        """How negative a price must be before burning energy for cash pays.

        The battery may charge and discharge within the same interval (300
        seconds is ample time to switch), which consumes grid energy without
        storing it. Holding SoC flat costs ``(1 + eta_c * eta_d)`` MWh of
        throughput for every ``(1 - eta_c * eta_d)`` MWh of net consumption, so
        it only pays when the price is below this threshold. See the README.
        """
        rte = self.round_trip_efficiency
        return -self.degradation_cost_per_mwh * (1.0 + rte) / (1.0 - rte)
