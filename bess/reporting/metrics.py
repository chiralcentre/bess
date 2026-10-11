"""Reports headline numbers, in the terms a trader would ask for them."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bess.asset.battery import BatterySpec
from bess.asset.simulator import Action

#: Clipping below this is solver float dust, not a real infeasibility.
CLIPPING_TOLERANCE_MWH = 1e-6


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
                # Cycles rather than raw MWh: this is the wear budget.
                # Warranties are written in equivalent full cycles, and the
                # rate is comparable across batteries of different sizes.
                f"  cycles per day       {self.cycles_per_day:>8.2f}",
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
            ]
            + self._diagnostics()
        )

    def _diagnostics(self) -> list[str]:
        """Lines worth printing only when they have something to report.

        Both are zero on every run where the optimiser and the simulator agree,
        which is every run with a correct LP. A permanent zero teaches a reader
        to skip the line, so these appear only when they fire.
        """
        lines: list[str] = []
        if self.clipped_mwh > CLIPPING_TOLERANCE_MWH:
            lines.extend([
                "",
                f"  WARNING: {self.clipped_mwh:,.1f} MWh clipped. The strategy asked for",
                "  more than the battery could physically deliver.",
            ])
        if self.simultaneous_flow_intervals:
            lines.extend([
                "",
                f"  {self.simultaneous_flow_intervals:,} intervals bought and sold at "
                "once, which pays only at",
                "  deeply negative prices. See the README.",
            ])
        return lines


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
    """Volume-weighted average price"""
    volume = intervals[volume_column]
    total = float(volume.sum())
    if total <= 0:
        return float("nan")
    return float((intervals["price_aud_mwh"] * volume).sum() / total)


def _daily_revenue(intervals: pd.DataFrame) -> pd.Series:
    return intervals.groupby(_day_key(intervals))["net_revenue_aud"].sum()


def _day_key(intervals: pd.DataFrame):
    """The column to group per-day figures by.

    ``run_backtest`` writes a ``trading_day`` column holding the 04:00 boundary,
    already corrected for interval-ending timestamps. Use it when it is there.

    A frame straight from :func:`simulate` has no such column, since nothing has
    told it where a day begins; those fall back to the calendar date. That is a
    plain midnight cut and would be the wrong boundary for real NEM data, so it
    only serves the short fixtures in the tests, which span no day boundary.
    """
    if "trading_day" in intervals.columns:
        return intervals["trading_day"]
    return pd.Series(intervals.index.date, index=intervals.index, name="trading_day")
