"""Running a strategy across a window, by trading day or in one pass.

Two horizons. ``"daily"`` optimises each trading day independently, opening and
closing empty, so every day is a self-contained round trip. ``"continuous"``
optimises the whole window at once, letting charge carry over; it earns more,
but a day's revenue becomes a cashflow. See the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd

from bess.asset.battery import BatterySpec
from bess.asset.simulator import SimulationResult, simulate
from bess.marketdata.prices import DataQualityReport, PriceSeries
from bess.strategies.base import Schedule, Strategy

# Dispatch intervals in a full trading day.
INTERVALS_PER_DAY = 288

Horizon = Literal["daily", "continuous"]


@dataclass(frozen=True)
class BacktestResult:
    """Every interval of the window, with the day it belongs to."""

    intervals: pd.DataFrame
    spec: BatterySpec
    strategy_name: str
    quality: DataQualityReport
    partial_days_dropped: int
    horizon: Horizon = "daily"

    @property
    def trading_days(self) -> int:
        return int(self.intervals["trading_day"].nunique())

    @property
    def net_revenue_aud(self) -> float:
        return float(self.intervals["net_revenue_aud"].sum())

    @property
    def days_are_round_trips(self) -> bool:
        """Whether each day's revenue stands on its own.

        True under a daily horizon, where every day opens and closes empty.
        False under a continuous horizon, where a day may sell energy bought
        earlier, making its revenue a cashflow.
        """
        return self.horizon == "daily"

    def day(self, day: pd.Timestamp) -> pd.DataFrame:
        """Just one trading day's intervals."""
        return self.intervals[self.intervals["trading_day"] == day]


def assign_trading_days(
    index: pd.DatetimeIndex, start_hour: int, interval: timedelta
) -> pd.Series:
    """Label each interval with the trading day it falls in.

    Timestamps are interval-*ending*, so the interval stamped 04:00 covers
    03:55 to 04:00 and belongs to the day that is finishing. Stepping back one
    interval before taking the date handles that off-by-one, which otherwise
    puts one interval of every day in the wrong bucket.
    """
    shifted = index - pd.Timedelta(hours=start_hour) - pd.Timedelta(interval)
    return pd.Series(shifted.normalize(), index=index)


def run_backtest(
    spec: BatterySpec,
    prices: PriceSeries,
    strategy: Strategy,
    trading_day_start_hour: int = 4,
    drop_partial_days: bool = True,
    horizon: Horizon = "daily",
) -> BacktestResult:
    """Optimise and simulate the window, by day or in one pass.

    Args:
        spec: the battery being operated.
        prices: cleaned price series, including the tradable mask.
        strategy: anything satisfying the :class:`Strategy` protocol.
        trading_day_start_hour: hour in market time at which a day rolls over.
            Defaults to 4: the NEM trading day runs 04:00-03:59 AEST.
        drop_partial_days: skip days with fewer than 288 intervals. A part-day
            produces a revenue figure that is not comparable with the others.
            Applied under both horizons, so the two are measured over exactly
            the same intervals.
        horizon: ``"daily"`` optimises each trading day independently;
            ``"continuous"`` optimises the whole window in one pass.

    Returns:
        A :class:`BacktestResult` whose frame carries every simulator column
        plus ``trading_day`` and ``water_value_aud_mwh``.
    """
    if horizon not in ("daily", "continuous"):
        raise ValueError(f"horizon must be 'daily' or 'continuous', got {horizon!r}")

    interval = timedelta(minutes=spec.interval_minutes)
    days = assign_trading_days(prices.prices.index, trading_day_start_hour, interval)

    complete = days.groupby(days).transform("size") == INTERVALS_PER_DAY
    dropped = int(days[~complete].nunique()) if drop_partial_days else 0
    keep = complete.to_numpy() if drop_partial_days else np.ones(len(days), dtype=bool)

    if not keep.any():
        raise ValueError(
            f"no complete trading days in {prices.quality.start:%Y-%m-%d} to "
            f"{prices.quality.end:%Y-%m-%d}; {dropped} partial day(s) were dropped"
        )

    window = prices.prices[keep]
    tradable = prices.tradable[keep]
    labels = days[keep]

    runner = _run_daily if horizon == "daily" else _run_continuous
    intervals = runner(spec, window, tradable, labels, strategy)

    # The simulator restarts its running total per call, so restate it across
    # the whole window and the P&L curve stays continuous.
    intervals["cumulative_net_revenue_aud"] = intervals["net_revenue_aud"].cumsum()

    return BacktestResult(
        intervals=intervals,
        spec=spec,
        strategy_name=getattr(strategy, "name", type(strategy).__name__),
        quality=prices.quality,
        partial_days_dropped=dropped,
        horizon=horizon,
    )


def _run_daily(
    spec: BatterySpec,
    window: pd.Series,
    tradable: np.ndarray,
    labels: pd.Series,
    strategy: Strategy,
) -> pd.DataFrame:
    """One optimisation per trading day, each opening and closing empty."""
    mask = pd.Series(tradable, index=window.index)
    frames = []
    for day, stamps in labels.groupby(labels):
        prices = window.loc[stamps.index]
        schedule = _plan(strategy, spec, prices, mask.loc[stamps.index].to_numpy())
        frames.append(_frame(spec, prices, schedule, day))
    return pd.concat(frames).sort_index()


def _run_continuous(
    spec: BatterySpec,
    window: pd.Series,
    tradable: np.ndarray,
    labels: pd.Series,
    strategy: Strategy,
) -> pd.DataFrame:
    """One optimisation across the window, state of charge crossing days."""
    schedule = _plan(strategy, spec, window, tradable)
    frame = _frame(spec, window, schedule, day=None)
    frame["trading_day"] = labels.to_numpy()
    return frame


def _frame(
    spec: BatterySpec,
    prices: pd.Series,
    schedule: Schedule,
    day: pd.Timestamp | None,
) -> pd.DataFrame:
    """Simulate one schedule and attach its day label and water value."""
    result = simulate(spec, prices, schedule.charge_mwh, schedule.discharge_mwh)
    frame = result.intervals.copy()
    if day is not None:
        frame["trading_day"] = day
    frame["water_value_aud_mwh"] = (
        schedule.water_value_aud_mwh
        if schedule.water_value_aud_mwh is not None
        else pd.NA
    )
    return frame


def _plan(
    strategy: Strategy, spec: BatterySpec, prices: pd.Series, mask: np.ndarray
) -> Schedule:
    """Hand the strategy a tradable mask where it can accept one.

    The protocol does not require a mask, so this sets it where the attribute
    exists and leaves other strategies alone.
    """
    if not hasattr(strategy, "tradable"):
        return strategy.plan(spec, prices)
    original = strategy.tradable
    try:
        strategy.tradable = mask
        return strategy.plan(spec, prices)
    finally:
        strategy.tradable = original


def as_simulation_result(backtest: BacktestResult) -> SimulationResult:
    """View the whole window as a single simulation, for the headline totals."""
    return SimulationResult(intervals=backtest.intervals, spec=backtest.spec)
