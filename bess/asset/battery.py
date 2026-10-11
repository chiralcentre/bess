"""Physical description of the battery.

Values come from :mod:`bess.config`; this module describes the shape.

Unit convention, relied on everywhere: ``charge_mwh`` and ``discharge_mwh`` are
**grid-side**, while state of charge is measured inside the battery. The
efficiencies are the exchange rate between the two.
"""

from __future__ import annotations

from dataclasses import dataclass

MINUTES_PER_HOUR = 60


@dataclass(frozen=True)
class BatterySpec:
    """Physical and commercial limits of the battery."""

    capacity_mwh: float
    max_charge_mw: float
    max_discharge_mw: float
    charge_efficiency: float
    discharge_efficiency: float
    initial_soc_mwh: float
    degradation_cost_per_mwh: float
    interval_minutes: int

    def __post_init__(self) -> None:
        """Reject a battery that could not exist.

        These values arrive from config, CLI arguments and dashboard sliders,
        so the interesting case is not the value we chose but the one someone
        else supplies. An efficiency above 1 is a perpetual motion machine, and
        the LP will find the infinite profit it implies.
        """
        if self.capacity_mwh <= 0:
            raise ValueError(f"capacity_mwh must be positive, got {self.capacity_mwh}")
        if self.max_charge_mw <= 0 or self.max_discharge_mw <= 0:
            raise ValueError("power ratings must be positive")
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError(
                f"charge_efficiency must be in (0, 1], got {self.charge_efficiency}"
            )
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError(
                f"discharge_efficiency must be in (0, 1], got {self.discharge_efficiency}"
            )
        if not 0 <= self.initial_soc_mwh <= self.capacity_mwh:
            raise ValueError(
                f"initial_soc_mwh must lie within [0, {self.capacity_mwh}], "
                f"got {self.initial_soc_mwh}"
            )
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
        """Most energy that can be bought in one interval, grid-side.

        50 MW for 1/12 of an hour is 4.167 MWh. This limit is what forces the
        battery to average into a position over many intervals, even when one
        price is far better than the rest.
        """
        return self.max_charge_mw * self.interval_hours

    @property
    def max_discharge_mwh(self) -> float:
        """Most energy that can be sold in one interval, grid-side."""
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
        """Spread needed to break even: 1/0.81, so about 23.5%."""
        return 1.0 / self.round_trip_efficiency

    @property
    def simultaneous_flow_price_threshold(self) -> float:
        """Price below which buying and selling at once becomes profitable.

        A battery can reverse direction inside a 300-second interval, so at a
        negative price it can be paid for the energy its own round-trip losses
        consume. Derivation and consequences are in the README under
        "Simultaneous charge and discharge"; at 81% round-trip and $5/MWh of
        wear the threshold is about -$47.63/MWh.

        Returns:
            A negative price in AUD/MWh, or ``-inf`` for a lossless battery,
            where consuming energy is never free.
        """
        rte = self.round_trip_efficiency
        if rte >= 1.0:
            return float("-inf")
        return -self.degradation_cost_per_mwh * (1.0 + rte) / (1.0 - rte)
