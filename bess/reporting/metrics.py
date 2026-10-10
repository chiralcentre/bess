"""Headline numbers, in the terms a trader would ask for them.

Two things are reported that a naive summary would leave out.

**Gross and net revenue separately.** Gross is market cashflow -- what
settlement would pay. Net subtracts modelled degradation, which is a real cost
but not a cashflow. Conflating them either overstates the P&L or hides what the
wear assumption is doing.

**The daily distribution, not just the total.** NEM revenue is concentrated:
the market is energy-only, so price spikes are the mechanism that funds peaking
capacity, and a handful of days can carry a large share of the value. A total
alone hides that, and it is the first thing to check before trusting a number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bess.asset.battery import BatterySpec
from bess.asset.simulator import Action


@dataclass(frozen=True)
class Metrics:
    """Summary of a backtest."""

    strategy_name: str
    trading_days: int
    gross_revenue_aud: float
    degradation_cost_aud: float
    net_revenue_aud: float
    revenue_per_mwh_capacity: float
    energy_bought_mwh: float
    energy_sold_mwh: float
    equivalent_full_cycles: float
    cycles_per_day: float
    average_buy_price: float
    average_sell_price: float
    captured_spread: float
    spread_ratio: float
    breakeven_ratio: float
    active_interval_share: float
    best_day_aud: float
    worst_day_aud: float
    median_day_aud: float
    top_three_day_share: float
    simultaneous_flow_intervals: int
    clipped_mwh: float

    def summary(self) -> str:
        return "\n".join(
            [
                f"{self.strategy_name} over {self.trading_days} trading days",
                "",
                f"  net revenue         ${self.net_revenue_aud:>14,.2f}",
                f"    gross             ${self.gross_revenue_aud:>14,.2f}",
                f"    degradation       ${-self.degradation_cost_aud:>14,.2f}",
                f"  per MWh capacity    ${self.revenue_per_mwh_capacity:>14,.2f}",
                "",
                f"  bought / sold        {self.energy_bought_mwh:>8,.0f} /"
                f" {self.energy_sold_mwh:,.0f} MWh",
                f"  full cycles          {self.equivalent_full_cycles:>8.1f}"
                f"  ({self.cycles_per_day:.2f}/day)",
                f"  active intervals     {self.active_interval_share:>8.1%}",
                "",
                f"  average buy         ${self.average_buy_price:>14,.2f}",
                f"  average sell        ${self.average_sell_price:>14,.2f}",
                f"  captured spread     ${self.captured_spread:>14,.2f}"
                f"   ({self.spread_ratio:.2f}x vs {self.breakeven_ratio:.2f}x needed)",
                "",
                f"  best day            ${self.best_day_aud:>14,.2f}",
                f"  median day          ${self.median_day_aud:>14,.2f}",
                f"  worst day           ${self.worst_day_aud:>14,.2f}",
                f"  top 3 days           {self.top_three_day_share:>8.1%} of total",
                "",
                f"  both directions      {self.simultaneous_flow_intervals:>8,} intervals",
                f"  clipped              {self.clipped_mwh:>8.1e} MWh",
            ]
        )


def summarise(
    intervals: pd.DataFrame, spec: BatterySpec, strategy_name: str = "strategy"
) -> Metrics:
    """Reduce a backtest frame to headline numbers."""
    if intervals.empty:
        raise ValueError("cannot summarise an empty backtest")

    bought = float(intervals["charge_mwh"].sum())
    sold = float(intervals["discharge_mwh"].sum())
    gross = float(intervals["gross_revenue_aud"].sum())
    degradation = float(intervals["degradation_aud"].sum())
    net = float(intervals["net_revenue_aud"].sum())

    buy_price = _vwap(intervals, "charge_mwh")
    sell_price = _vwap(intervals, "discharge_mwh")

    daily = _daily_revenue(intervals)
    top_three = daily.nlargest(min(3, len(daily))).sum()

    drained = sold / spec.discharge_efficiency
    cycles = drained / spec.capacity_mwh
    days = max(len(daily), 1)

    return Metrics(
        strategy_name=strategy_name,
        trading_days=len(daily),
        gross_revenue_aud=gross,
        degradation_cost_aud=degradation,
        net_revenue_aud=net,
        revenue_per_mwh_capacity=net / spec.capacity_mwh,
        energy_bought_mwh=bought,
        energy_sold_mwh=sold,
        equivalent_full_cycles=cycles,
        cycles_per_day=cycles / days,
        average_buy_price=buy_price,
        average_sell_price=sell_price,
        captured_spread=sell_price - buy_price,
        # Guard the ratio: a negative or near-zero average buy price is normal
        # in NSW1 and would otherwise produce a meaningless number.
        spread_ratio=sell_price / buy_price if buy_price > 1e-6 else float("inf"),
        breakeven_ratio=spec.breakeven_price_ratio,
        active_interval_share=float((intervals["action"] != Action.IDLE).mean()),
        best_day_aud=float(daily.max()),
        worst_day_aud=float(daily.min()),
        median_day_aud=float(daily.median()),
        top_three_day_share=float(top_three / net) if net else 0.0,
        simultaneous_flow_intervals=int((intervals["action"] == Action.BOTH).sum()),
        clipped_mwh=float(intervals["clipped_mwh"].sum()),
    )


def daily_summary(intervals: pd.DataFrame) -> pd.DataFrame:
    """One row per trading day, for the distribution chart and the day picker."""
    grouped = intervals.groupby(_day_key(intervals))
    frame = pd.DataFrame(
        {
            "net_revenue_aud": grouped["net_revenue_aud"].sum(),
            "gross_revenue_aud": grouped["gross_revenue_aud"].sum(),
            "energy_bought_mwh": grouped["charge_mwh"].sum(),
            "energy_sold_mwh": grouped["discharge_mwh"].sum(),
            "min_price": grouped["price_aud_mwh"].min(),
            "max_price": grouped["price_aud_mwh"].max(),
            "mean_price": grouped["price_aud_mwh"].mean(),
            "active_intervals": grouped["action"].apply(lambda s: (s != Action.IDLE).sum()),
        }
    )
    frame["spread"] = frame["max_price"] - frame["min_price"]
    return frame.sort_index()


def _vwap(intervals: pd.DataFrame, volume_column: str) -> float:
    """Volume-weighted average price, which is the only honest average here.

    A plain mean over intervals would weight a 0.1 MWh interval the same as a
    full one.
    """
    volume = intervals[volume_column]
    total = float(volume.sum())
    if total <= 0:
        return float("nan")
    return float((intervals["price_aud_mwh"] * volume).sum() / total)


def _daily_revenue(intervals: pd.DataFrame) -> pd.Series:
    return intervals.groupby(_day_key(intervals))["net_revenue_aud"].sum()


def _day_key(intervals: pd.DataFrame):
    """Group by trading day where the backtest labelled one, else calendar day."""
    if "trading_day" in intervals.columns:
        return intervals["trading_day"]
    return pd.Series(intervals.index.date, index=intervals.index, name="trading_day")
