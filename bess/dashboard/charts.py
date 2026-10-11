"""Plotly figures for the dashboard.

Kept apart from the Streamlit layout so the figures can be built and checked
without a browser, and so the layout file stays about arrangement.

The day chart is the one that matters. It puts the water value on the same axis
as the price, because that single pair explains every decision the battery
makes: above the line it sells, below it buys. Everything else -- the shading,
the state of charge, the running P&L -- is there to confirm what those two
lines already say.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bess.asset.simulator import Action
from bess.reporting.trades import explain_interval

PRICE = "#1f2d3d"
WATER = "#7b68ee"
CHARGE = "#2a7de1"
DISCHARGE = "#e8833a"
SOC = "#4a9d7f"
PNL = "#2f6f4e"
GRID = "#e6e9ed"


def day_figure(frame: pd.DataFrame, title: str = "") -> go.Figure:
    """Price, water value, schedule, state of charge and P&L for one day.

    Three panels sharing a time axis: what the market did, what the battery
    held, and what it earned.
    """
    # No subplot titles: each panel's y-axis already names it, and centred
    # titles overlap the legend and the panel above them at this row spacing.
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.54, 0.23, 0.23],
    )

    hover = [explain_interval(row) for _, row in frame.iterrows()]

    figure.add_trace(
        go.Scatter(
            x=frame.index, y=frame["price_aud_mwh"], name="RRP",
            line=dict(color=PRICE, width=1.6, shape="hv"),
            customdata=hover,
            hovertemplate="<b>%{x|%H:%M}</b><br>%{customdata}<extra></extra>",
        ),
        row=1, col=1,
    )

    if "water_value_aud_mwh" in frame and frame["water_value_aud_mwh"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=frame.index, y=frame["water_value_aud_mwh"],
                name="Internal value of stored energy",
                line=dict(color=WATER, width=1.6, dash="dot", shape="hv"),
                hovertemplate="internal value $%{y:,.2f}/MWh<extra></extra>",
            ),
            row=1, col=1,
        )

    # Must run after the panel's traces: add_vrect defaults to
    # exclude_empty_subplots=True and silently drops every shape aimed at a
    # subplot with no traces yet. layer="below" keeps the bands behind the
    # lines, so drawing order has no effect on appearance.
    _add_action_bands(figure, frame)

    figure.add_trace(
        go.Scatter(
            x=frame.index, y=frame["soc_after_mwh"], name="State of charge",
            line=dict(color=SOC, width=1.4), fill="tozeroy",
            fillcolor="rgba(74,157,127,0.14)",
            hovertemplate="SoC %{y:,.1f} MWh<extra></extra>",
        ),
        row=2, col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=frame.index, y=frame["net_revenue_aud"].cumsum(), name="Cumulative revenue",
            line=dict(color=PNL, width=1.6), fill="tozeroy",
            fillcolor="rgba(47,111,78,0.12)",
            hovertemplate="$%{y:,.0f}<extra></extra>",
        ),
        row=3, col=1,
    )

    figure.update_yaxes(title_text="Price (AUD/MWh)", row=1, col=1)
    figure.update_yaxes(title_text="Stored (MWh)", row=2, col=1)
    figure.update_yaxes(title_text="Revenue (AUD)", row=3, col=1)
    figure.update_xaxes(title_text="AEST", row=3, col=1)

    _style(figure, title, height=720)
    figure.update_layout(hovermode="x unified")
    return figure


def _add_action_bands(figure: go.Figure, frame: pd.DataFrame) -> None:
    """Shade the periods the battery was buying or selling.

    One rectangle per contiguous block. A day has perhaps a dozen blocks
    against 288 intervals, and 288 overlapping shapes would look wrong and
    render slowly.
    """
    action = frame["action"].astype(str)
    blocks = (action != action.shift()).cumsum()

    for _, block in frame.groupby(blocks):
        label = str(block["action"].iloc[0])
        if label == Action.IDLE:
            continue
        colour = CHARGE if label == Action.CHARGE else DISCHARGE
        figure.add_vrect(
            x0=block.index[0], x1=block.index[-1],
            fillcolor=colour, opacity=0.13, line_width=0,
            layer="below", row=1, col=1,
        )


def daily_revenue_figure(daily: pd.DataFrame) -> go.Figure:
    """Revenue per trading day, to show how evenly it is spread.

    A total hides concentration. Seeing the spread of days is the first check
    on whether a headline number is worth trusting.
    """
    median = daily["net_revenue_aud"].median()
    figure = go.Figure(
        go.Bar(
            x=daily.index, y=daily["net_revenue_aud"],
            marker_color=PNL, marker_line_width=0,
            customdata=daily[["energy_sold_mwh", "min_price", "max_price"]].to_numpy(),
            hovertemplate=(
                "<b>%{x|%a %d %b}</b><br>$%{y:,.0f}<br>"
                "sold %{customdata[0]:,.0f} MWh<br>"
                "RRP $%{customdata[1]:,.2f} to $%{customdata[2]:,.2f}<extra></extra>"
            ),
        )
    )
    figure.add_hline(
        y=median, line=dict(color=DISCHARGE, width=1.4, dash="dash"),
        annotation_text=f"median ${median:,.0f}", annotation_position="top left",
    )
    figure.update_yaxes(title_text="Net revenue (AUD)")
    _style(figure, "Revenue by trading day", height=320)
    figure.update_layout(showlegend=False)  # one series; the title names it
    return figure


def price_duration_figure(prices: pd.Series) -> go.Figure:
    """Prices sorted high to low: how much of the window is worth trading.

    A battery only cares about the two ends of this curve. The steepness at
    each end is the arbitrage opportunity, and the flat middle is the time it
    sits idle.
    """
    ordered = prices.sort_values(ascending=False).reset_index(drop=True)
    share = ordered.index / max(len(ordered) - 1, 1) * 100

    figure = go.Figure(
        go.Scatter(
            x=share, y=ordered, line=dict(color=PRICE, width=1.6),
            hovertemplate="top %{x:.1f}% of intervals<br>$%{y:,.2f}/MWh<extra></extra>",
        )
    )
    figure.add_hline(y=0, line=dict(color=GRID, width=1))
    figure.update_xaxes(title_text="Share of intervals at or above this price (%)")
    figure.update_yaxes(title_text="AUD/MWh")
    _style(figure, "Price duration curve", height=320)
    figure.update_layout(showlegend=False)
    return figure


def _style(figure: go.Figure, title: str, height: int) -> None:
    figure.update_layout(
        title=dict(text=title, font=dict(size=15)) if title else None,
        height=height,
        margin=dict(l=72, r=24, t=64 if title else 46, b=44),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.015, x=0,
            bgcolor="rgba(255,255,255,0.85)", borderwidth=0,
        ),
        showlegend=True,
    )
    figure.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
