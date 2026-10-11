"""Streamlit dashboard.

    python -m streamlit run app.py

Reads the committed price cache, so it starts with no network access. The
sidebar exists so the assumptions can be argued with rather than taken on
trust.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from bess.backtest import run_backtest
from bess.config import CONFIG
from bess.dashboard.charts import (
    daily_revenue_figure,
    day_figure,
    price_duration_figure,
)
from bess.marketdata.nemweb import NemwebClient
from bess.marketdata.prices import PriceSeries, load_prices
from bess.reporting.metrics import daily_summary, summarise
from bess.reporting.trades import build_trade_log
from bess.strategies.foresight_lp import ForesightLP

st.set_page_config(page_title="BESS arbitrage - NSW1", layout="wide")


@st.cache_data(show_spinner="Loading NSW1 dispatch prices...")
def _prices() -> PriceSeries:
    return load_prices(
        NemwebClient(cache_dir=CONFIG.paths.raw_dir),
        CONFIG.backtest.start_date,
        CONFIG.backtest.end_date,
        CONFIG.market.region,
        cache_path=CONFIG.paths.price_cache,
    )


@st.cache_data(show_spinner="Optimising...")
def _backtest(
    power_mw: float, degradation: float, efficiency: float, start_hour: int,
    horizon: str = "daily",
):
    """Re-run the whole window for one set of assumptions.

    Cached on the arguments, so moving a slider back to a value already seen
    is instant. A full 29-day run takes about 0.2 seconds anyway.
    """
    spec = replace(
        CONFIG.battery,
        max_charge_mw=power_mw,
        max_discharge_mw=power_mw,
        degradation_cost_per_mwh=degradation,
        charge_efficiency=efficiency,
        discharge_efficiency=efficiency,
    )
    result = run_backtest(spec, _prices(), ForesightLP(), start_hour,
                          horizon=horizon)
    return result, summarise(result.intervals, spec, result.strategy_name)


def sidebar() -> tuple[float, float, float, int, str]:
    st.sidebar.title("Assumptions")
    st.sidebar.caption(
        "Defaults are the configured case. Change them to test how much the "
        "result depends on a choice."
    )

    power = st.sidebar.slider(
        "Power rating (MW)", 25.0, 200.0, float(CONFIG.battery.max_charge_mw), 5.0,
        help="At a fixed 200 MWh, this sets the duration. 50 MW is a four-hour "
             "battery; 100 MW a two-hour one.",
    )
    st.sidebar.caption(f"= a {200.0 / power:.1f}-hour battery")

    efficiency = st.sidebar.slider(
        "One-way efficiency", 0.80, 1.00, float(CONFIG.battery.charge_efficiency), 0.01,
        help="Applied on each leg, so the round trip is this squared.",
    )
    st.sidebar.caption(
        f"round trip {efficiency ** 2:.1%} - needs a "
        f"{1 / efficiency ** 2 - 1:.1%} spread to break even"
    )

    degradation = st.sidebar.slider(
        "Degradation ($/MWh moved)", 0.0, 25.0,
        float(CONFIG.battery.degradation_cost_per_mwh), 0.5,
        help="Wear cost per MWh in either direction. Filters out trades too "
             "small to be worth the cycle.",
    )

    start_hour = st.sidebar.selectbox(
        "Trading day starts at", list(range(24)),
        index=CONFIG.backtest.trading_day_start_hour,
        format_func=lambda h: f"{h:02d}:00 AEST",
        help="Not midnight by default: the 00:00-04:00 window is expensive "
             "here, so a midnight split would cut the evening discharge in two.",
    )
    horizon = st.sidebar.radio(
        "Horizon", ["daily", "continuous"],
        format_func=lambda h: {
            "daily": "Each day on its own",
            "continuous": "One pass, charge carries over",
        }[h],
        help="Resetting to empty at 04:00 is bookkeeping, not physics. "
             "Letting charge carry across days earns more, but a day's "
             "revenue then becomes a cashflow instead of a round trip.",
    )
    return power, efficiency, degradation, start_hour, horizon


def main() -> None:
    power, efficiency, degradation, start_hour, horizon = sidebar()
    prices = _prices()
    result, metrics = _backtest(power, degradation, efficiency, start_hour,
                                horizon)
    daily = daily_summary(result.intervals)

    st.title("Battery arbitrage - NSW1")
    st.caption(
        f"{result.trading_days} trading days, "
        f"{prices.quality.start:%d %b} to {prices.quality.end:%d %b %Y} AEST  -  "
        f"perfect foresight, so this is a ceiling rather than a forecast"
    )

    # -- what did it earn ---------------------------------------------------
    row = st.columns(5)
    row[0].metric("Net revenue", f"${metrics.net_revenue_aud:,.0f}",
                  help="Gross market cashflow less modelled degradation.")
    row[1].metric("Per day (median)", f"${metrics.median_day_aud:,.0f}",
                  help="Median is used rather than mean: revenue is skewed.")
    row[2].metric("Average buy", f"${metrics.average_buy_price:,.2f}")
    row[3].metric("Average sell", f"${metrics.average_sell_price:,.2f}")
    row[4].metric("Cycles per day", f"{metrics.cycles_per_day:.2f}")

    second = st.columns(5)
    second[0].metric("Gross revenue", f"${metrics.gross_revenue_aud:,.0f}")
    second[1].metric("Degradation", f"-${metrics.degradation_cost_aud:,.0f}")
    second[2].metric("Captured spread", f"${metrics.captured_spread:,.2f}",
                     help=f"Needs {metrics.breakeven_ratio:.2f}x to break even "
                          f"at this efficiency.")
    second[3].metric("Energy sold", f"{metrics.energy_sold_mwh:,.0f} MWh")
    second[4].metric("Active intervals", f"{metrics.active_interval_share:.0%}",
                     help="The rest of the time it sits idle, because no "
                          "spread cleared the hurdle.")

    # -- why did it trade when it did ---------------------------------------
    st.subheader("A single trading day")
    st.caption(
        "The dotted line is what the battery reckons a stored MWh is worth at "
        "that moment. It buys below the line and sells above it - that one "
        "comparison explains every decision on the chart."
    )

    days = sorted(result.intervals["trading_day"].dt.date.unique())
    default = int(daily["net_revenue_aud"].to_numpy().argmax())
    chosen = st.selectbox(
        "Trading day", days, index=default,
        format_func=lambda d: (
            f"{d:%a %d %b}  -  ${daily['net_revenue_aud'].iloc[days.index(d)]:,.0f}"
        ),
    )
    frame = result.intervals[result.intervals["trading_day"].dt.date == chosen]
    st.plotly_chart(day_figure(frame), use_container_width=True)

    st.markdown("**Trades on this day**")
    log = build_trade_log(frame)
    if log.empty:
        st.info("No spread cleared the hurdle on this day, so the battery stayed idle.")
    else:
        st.dataframe(
            log[["start", "end", "action", "energy_mwh", "vwap_aud_mwh",
                 "net_revenue_aud", "reason"]],
            use_container_width=True, hide_index=True,
            column_config={
                "start": st.column_config.DatetimeColumn("Start", format="HH:mm"),
                "end": st.column_config.DatetimeColumn("End", format="HH:mm"),
                "action": st.column_config.TextColumn("Direction"),
                "energy_mwh": st.column_config.NumberColumn("MWh", format="%.1f"),
                "vwap_aud_mwh": st.column_config.NumberColumn("Avg price", format="$%.2f"),
                "net_revenue_aud": st.column_config.NumberColumn(
                    "Cashflow", format="$%.0f",
                    help="This leg only. A buy is negative; day profit is both legs.",
                ),
                "reason": st.column_config.TextColumn("Why", width="large"),
            },
        )

    # -- should I believe it ------------------------------------------------
    st.subheader("Across the window")
    left, right = st.columns(2)
    left.plotly_chart(daily_revenue_figure(daily), use_container_width=True)
    right.plotly_chart(price_duration_figure(prices.prices), use_container_width=True)

    # Dollar signs are escaped throughout: Streamlit's markdown reads a pair of
    # them as LaTeX delimiters and silently swallows everything between.
    st.caption(
        f"The best three days are {metrics.top_three_day_share:.0%} of total "
        f"revenue. Worst day \\${metrics.worst_day_aud:,.0f}, best "
        f"\\${metrics.best_day_aud:,.0f}."
    )

    with st.expander("Data quality and what this model does not include"):
        quality, caveats = st.columns(2)
        quality.code(prices.quality.summary(), language=None)
        caveats.markdown(
            f"""
**Checks**

- Simulator clipped `{metrics.clipped_mwh:.1e}` MWh, so the schedule was
  always physically possible.
- `{metrics.simultaneous_flow_intervals}` intervals bought and sold at once,
  which pays only at deeply negative prices. See the README.
- `{result.partial_days_dropped}` partial day(s) dropped at the window edges.

**Known biases**

- **Perfect foresight.** The whole price series is known in advance, so this
  is an upper bound, not a forecast.
- **Price taking.** 50 MW discharging into a tight peak would push the price
  down, which overstates revenue exactly where it matters most.
- **Perfect fills.** A real battery offers into dispatch and may not be taken.
- **Energy only.** No FCAS or capacity contracts, which understates the real
  revenue stack and partly offsets the two biases above.
"""
        )


if __name__ == "__main__":
    main()
