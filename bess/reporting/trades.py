"""Turning 8,640 five-minute rows into something a trader would read.

Nobody reads a schedule interval by interval. A trader reads *trades*: a block
of time, a direction, a volume, an average price, and a reason. This module
collapses runs of consecutive intervals with the same action into those blocks
and writes the reason in words.

The reason is assembled here, deliberately. The simulator records facts --
which constraint bound, what the state of charge was -- and this module turns
those facts into prose. That split keeps the physics testable without parsing
English.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bess.asset.simulator import Action, BindingConstraint

#: Blocks smaller than this are float dust from the solver.
MIN_TRADE_MWH = 1e-6


def build_trade_log(intervals: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive same-action intervals into trades.

    Args:
        intervals: a simulator frame, optionally carrying ``trading_day`` and
            ``water_value_aud_mwh``.

    Returns:
        One row per trade, newest decision order preserved, with a plain
        English ``reason`` column.
    """
    frame = intervals[intervals["action"] != Action.IDLE].copy()
    if frame.empty:
        return _empty_log()

    has_day = "trading_day" in frame.columns
    # A block breaks when the direction changes, when there is a gap in time,
    # or at a trading day boundary -- a trade cannot span two days when each
    # day is optimised independently.
    direction_change = frame["action"].astype(str).ne(
        frame["action"].astype(str).shift()
    )
    time_gap = frame.index.to_series().diff() != pd.Series(
        intervals.index
    ).diff().mode().iloc[0]
    breaks = direction_change | time_gap.fillna(True)
    if has_day:
        breaks |= frame["trading_day"].ne(frame["trading_day"].shift())

    frame["_block"] = breaks.cumsum()

    rows = [_summarise_block(block) for _, block in frame.groupby("_block")]
    log = pd.DataFrame([r for r in rows if r is not None])
    return log if not log.empty else _empty_log()


def _summarise_block(block: pd.DataFrame) -> dict | None:
    """Reduce one run of same-direction intervals to a single trade row."""
    action = str(block["action"].iloc[0])
    volume_column = (
        "charge_mwh" if action in (Action.CHARGE, Action.BOTH) else "discharge_mwh"
    )
    energy = float(block[volume_column].sum())
    if energy < MIN_TRADE_MWH:
        return None

    vwap = float((block["price_aud_mwh"] * block[volume_column]).sum() / energy)
    water = (
        float(block["water_value_aud_mwh"].mean())
        if "water_value_aud_mwh" in block and block["water_value_aud_mwh"].notna().any()
        else np.nan
    )
    constraints = [
        c
        for c in block["binding_constraint"].astype(str).unique()
        if c != BindingConstraint.NONE
    ]

    row = {
        "start": block.index[0],
        "end": block.index[-1],
        "intervals": len(block),
        "action": action,
        "energy_mwh": energy,
        "vwap_aud_mwh": vwap,
        "gross_revenue_aud": float(block["gross_revenue_aud"].sum()),
        "degradation_aud": float(block["degradation_aud"].sum()),
        "net_revenue_aud": float(block["net_revenue_aud"].sum()),
        "soc_start_mwh": float(block["soc_before_mwh"].iloc[0]),
        "soc_end_mwh": float(block["soc_after_mwh"].iloc[-1]),
        "water_value_aud_mwh": water,
        "binding_constraint": constraints[0] if constraints else BindingConstraint.NONE,
    }
    if "trading_day" in block:
        row["trading_day"] = block["trading_day"].iloc[0]
    row["reason"] = _explain_trade(row)
    return row


def _explain_trade(row: dict) -> str:
    """Why this trade happened, in a sentence a trader would accept."""
    action = row["action"]
    minutes = row["intervals"] * 5
    verb = {
        Action.CHARGE: "Bought",
        Action.DISCHARGE: "Sold",
        Action.BOTH: "Bought and sold",
    }.get(action, "Moved")

    sentence = (
        f"{verb} {row['energy_mwh']:,.1f} MWh over {minutes} minutes at a "
        f"volume-weighted ${row['vwap_aud_mwh']:,.2f}/MWh"
    )

    water = row["water_value_aud_mwh"]
    if not np.isnan(water):
        if action == Action.CHARGE:
            sentence += f", below the ${water:,.2f} the battery valued stored energy at"
        elif action == Action.DISCHARGE:
            sentence += f", above the ${water:,.2f} the battery valued stored energy at"

    sentence += "."

    limit = str(row["binding_constraint"])
    if limit == BindingConstraint.SOC_CEILING:
        sentence += " Stopped by the battery filling up."
    elif limit == BindingConstraint.SOC_FLOOR:
        sentence += " Stopped by the battery emptying."
    elif limit == BindingConstraint.CONNECTION:
        sentence += " Held at the connection limit throughout."

    if action == Action.BOTH:
        sentence += (
            " Charging and discharging within the same interval: at this price the "
            "battery is paid for the energy its own round-trip losses consume."
        )
    return sentence


def explain_interval(row: pd.Series) -> str:
    """One interval's decision, for a chart tooltip."""
    action = str(row["action"])
    price = row["price_aud_mwh"]
    soc = row["soc_after_mwh"]

    if action == Action.IDLE:
        head = f"Idle at ${price:,.2f}/MWh"
    elif action == Action.CHARGE:
        head = f"Bought {row['charge_mwh']:.2f} MWh at ${price:,.2f}/MWh"
    elif action == Action.DISCHARGE:
        head = f"Sold {row['discharge_mwh']:.2f} MWh at ${price:,.2f}/MWh"
    else:
        head = (
            f"Bought {row['charge_mwh']:.2f} and sold {row['discharge_mwh']:.2f} MWh "
            f"at ${price:,.2f}/MWh"
        )

    water = row.get("water_value_aud_mwh", np.nan)
    tail = f" | SoC {soc:,.1f} MWh"
    if water is not None and not pd.isna(water):
        tail += f" | internal value ${float(water):,.2f}/MWh"
    limit = str(row["binding_constraint"])
    if limit != BindingConstraint.NONE:
        tail += f" | limited by {limit.replace('_', ' ')}"
    return head + tail


def _empty_log() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "start", "end", "intervals", "action", "energy_mwh", "vwap_aud_mwh",
            "gross_revenue_aud", "degradation_aud", "net_revenue_aud",
            "soc_start_mwh", "soc_end_mwh", "water_value_aud_mwh",
            "binding_constraint", "trading_day", "reason",
        ]
    )
