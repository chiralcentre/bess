"""What every strategy produces, and the contract it has to meet.

A strategy decides *when* to buy and sell, never what that is worth: it returns
a schedule, and :mod:`bess.asset.simulator` replays it to produce state of
charge and money. Keeping those apart is what makes a hindsight optimiser and a
live one comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from bess.asset.battery import BatterySpec


@dataclass(frozen=True)
class Schedule:
    """A proposed plan: how much to buy and sell in each interval, grid-side."""

    name: str
    charge_mwh: np.ndarray
    discharge_mwh: np.ndarray

    #: Marginal value of a stored MWh in each interval, in AUD/MWh, where the
    #: strategy can say. This is the battery's internal "water value": it buys
    #: when energy is cheaper than this and sells when it is dearer. Strategies
    #: that have no such notion leave it as None.
    water_value_aud_mwh: np.ndarray | None = None

    #: The strategy's own claim about what this schedule is worth. Checked
    #: against the simulator's independent recomputation; disagreement means a
    #: bug in the strategy, not a better answer.
    claimed_net_revenue_aud: float | None = None

    def __post_init__(self) -> None:
        if len(self.charge_mwh) != len(self.discharge_mwh):
            raise ValueError("charge and discharge arrays must be the same length")


class Strategy(Protocol):
    """Anything that can turn prices into a schedule."""

    name: str

    def plan(self, spec: BatterySpec, prices: pd.Series) -> Schedule:
        """Decide what the battery should do across the given price series."""
        ...
