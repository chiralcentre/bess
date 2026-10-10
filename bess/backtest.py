"""Running a strategy across a window, one trading day at a time.

Wires the packages together: takes a price series and a strategy, optimises
each trading day independently, replays every schedule through the simulator,
and returns one frame covering the whole window.

Days are optimised separately rather than as one long horizon. That keeps the
battery on a daily cycle, which is what a real operator runs and what warranty
cycle limits assume, and it makes daily revenue the natural reporting unit. The
cost is that charge cannot be carried across a day boundary -- a stated
simplification that slightly understates revenue.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from bess.asset.battery import BatterySpec
from bess.asset.simulator import SimulationResult, simulate
from bess.marketdata.prices import DataQualityReport, PriceSeries
from bess.strategies.base import Strategy

#: Dispatch intervals in a full trading day.
INTERVALS_PER_DAY = 288


@dataclass(frozen=True)
class BacktestResult:
    """Every interval of the window, with the day it belongs to."""

    intervals: pd.DataFrame
    spec: BatterySpec
    strategy_name: str
    quality: DataQualityReport
    partial_days_dropped: int

    @property
    def trading_days(self) -> int:
        return int(self.intervals["trading_day"].nunique())

    @property
    def net_revenue_aud(self) -> float:
        return float(self.intervals["net_revenue_aud"].sum())

    def day(self, day: pd.Timestamp) -> pd.DataFrame:
        """Just one trading day's intervals."""
        return self.intervals[self.intervals["trading_day"] == day]


def assign_trading_days(
    index: pd.DatetimeIndex, start_hour: int, interval: timedelta
) -> pd.Series:
    """Label each interval with the trading day it falls in.

    Timestamps are interval-*ending*, so the interval stamped 04:00 covers
    03:55 to 04:00 and belongs to the day that is finishing, not the one
    starting. Stepping back one interval before taking the date handles that
    off-by-one, which would otherwise put one interval of every day in the
    wrong bucket.
    """
    shifted = index - pd.Timedelta(hours=start_hour) - pd.Timedelta(interval)
    return pd.Series(shifted.normalize(), index=index)


def run_backtest(
    spec: BatterySpec,
    prices: PriceSeries,
    strategy: Strategy,
    trading_day_start_hour: int = 4,
    drop_partial_days: bool = True,
) -> BacktestResult:
    """Optimise and simulate every complete trading day in the window.

    Args:
        spec: the battery being operated.
        prices: cleaned price series, including the tradable mask.
        strategy: anything satisfying the :class:`Strategy` protocol.
        trading_day_start_hour: hour in market time at which a day rolls over.
        drop_partial_days: skip days with fewer than 288 intervals. A part-day
            produces a revenue figure that is not comparable with the others,
            which would distort the daily distribution.

    Returns:
        A :class:`BacktestResult` whose frame carries every simulator column
        plus ``trading_day`` and ``water_value_aud_mwh``.
    """
    interval = timedelta(minutes=spec.interval_minutes)
    days = assign_trading_days(prices.prices.index, trading_day_start_hour, interval)

    frames: list[pd.DataFrame] = []
    dropped = 0

    for day, stamps in days.groupby(days):
        window = prices.prices.loc[stamps.index]
        if drop_partial_days and len(window) < INTERVALS_PER_DAY:
            dropped += 1
            continue

        mask = prices.tradable[prices.prices.index.isin(stamps.index)]
        schedule = _plan_with_mask(strategy, spec, window, mask)
        result = simulate(spec, window, schedule.charge_mwh, schedule.discharge_mwh)

        frame = result.intervals.copy()
        frame["trading_day"] = day
        frame["water_value_aud_mwh"] = (
            schedule.water_value_aud_mwh
            if schedule.water_value_aud_mwh is not None
            else pd.NA
        )
        frames.append(frame)

    if not frames:
        raise ValueError(
            f"no complete trading days in {prices.quality.start:%Y-%m-%d} to "
            f"{prices.quality.end:%Y-%m-%d}; {dropped} partial day(s) were dropped"
        )

    intervals = pd.concat(frames).sort_index()
    # Cumulative revenue is per-day from the simulator; restate it across the
    # whole window so the P&L curve is continuous.
    intervals["cumulative_net_revenue_aud"] = intervals["net_revenue_aud"].cumsum()

    return BacktestResult(
        intervals=intervals,
        spec=spec,
        strategy_name=getattr(strategy, "name", type(strategy).__name__),
        quality=prices.quality,
        partial_days_dropped=dropped,
    )


def _plan_with_mask(strategy: Strategy, spec: BatterySpec, window, mask):
    """Hand the strategy a per-day tradable mask where it can accept one.

    Only strategies that take a mask can honour it; the protocol does not
    require one, so this sets it where the attribute exists and leaves other
    strategies alone.
    """
    if hasattr(strategy, "tradable"):
        original = strategy.tradable
        try:
            strategy.tradable = mask
            return strategy.plan(spec, window)
        finally:
            strategy.tradable = original
    return strategy.plan(spec, window)


def as_simulation_result(backtest: BacktestResult) -> SimulationResult:
    """View the whole window as a single simulation, for the headline totals."""
    return SimulationResult(intervals=backtest.intervals, spec=backtest.spec)
