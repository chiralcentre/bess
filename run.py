"""Command line entry point for the backtest.

    python run.py              summary report
    python run.py --help       every option

Reads the committed price cache by default, so nothing needs the network.
Options are documented in the README under "Running a backtest".
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from bess.backtest import BacktestResult, run_backtest
from bess.config import CONFIG, Config
from bess.marketdata.nemweb import NemwebClient
from bess.marketdata.prices import PriceSeries, load_prices
from bess.reporting.metrics import daily_summary, summarise
from bess.reporting.trades import build_trade_log, explain_interval
from bess.strategies.foresight_lp import ForesightLP

RULE = "-" * 72


def main() -> None:
    args = _parse_args()
    config = _apply_overrides(CONFIG, args)

    prices = _load(config, refresh=args.refresh)
    print(RULE)
    print(prices.quality.summary())

    backtest = run_backtest(
        config.battery,
        prices,
        ForesightLP(),
        config.backtest.trading_day_start_hour,
        horizon=args.horizon,
    )
    metrics = summarise(backtest.intervals, config.battery, backtest.strategy_name)

    print(RULE)
    print(metrics.summary())
    if not backtest.days_are_round_trips:
        print(
            "\n  Continuous horizon: charge carries across day boundaries, so a"
            "\n  day's revenue is a cashflow and may sell energy bought earlier."
        )
    if backtest.partial_days_dropped:
        print(
            f"\n  {backtest.partial_days_dropped} partial day(s) dropped at the "
            f"window edges; a part-day's revenue is not comparable."
        )

    if args.trades:
        _print_trades(backtest, args.trades)
    if args.daily:
        _print_daily(backtest)
    if args.day:
        _print_day(backtest, args.day)
    if args.export:
        _export(backtest, Path(args.export))

    print(RULE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trades", type=int, metavar="N",
                        help="show the N largest trades by cashflow")
    parser.add_argument("--daily", action="store_true",
                        help="show revenue for every trading day")
    parser.add_argument("--day", type=date.fromisoformat, metavar="YYYY-MM-DD",
                        help="show one trading day interval by interval")
    parser.add_argument("--export", metavar="DIR",
                        help="write intervals, trades and daily summary to CSV")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch from NEMWEB instead of using the cache")
    parser.add_argument("--horizon", choices=["daily", "continuous"],
                        default="daily",
                        help="optimise each trading day on its own (default), "
                             "or the whole window in one pass with charge "
                             "carrying across day boundaries")
    parser.add_argument("--start", type=date.fromisoformat, metavar="YYYY-MM-DD",
                        help="first calendar day of the window")
    parser.add_argument("--end", type=date.fromisoformat, metavar="YYYY-MM-DD",
                        help="last calendar day of the window")
    parser.add_argument("--power-mw", type=float, metavar="MW",
                        help="override the power rating, for a sensitivity run")
    parser.add_argument("--degradation", type=float, metavar="AUD_PER_MWH",
                        help="override the degradation cost")
    return parser.parse_args()


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Let the flags override the configured assumptions, for sweeps."""
    battery = config.battery
    if args.power_mw is not None:
        battery = replace(
            battery, max_charge_mw=args.power_mw, max_discharge_mw=args.power_mw
        )
    if args.degradation is not None:
        battery = replace(battery, degradation_cost_per_mwh=args.degradation)

    backtest = config.backtest
    if args.start or args.end:
        backtest = replace(
            backtest,
            start_date=args.start or backtest.start_date,
            end_date=args.end or backtest.end_date,
        )
    return replace(config, battery=battery, backtest=backtest)


def _load(config: Config, refresh: bool) -> PriceSeries:
    client = NemwebClient(cache_dir=config.paths.raw_dir)
    cache = None if refresh else config.paths.price_cache
    return load_prices(
        client,
        config.backtest.start_date,
        config.backtest.end_date,
        config.market.region,
        cache_path=cache,
    )


def _print_trades(backtest: BacktestResult, count: int) -> None:
    log = build_trade_log(backtest.intervals)
    print(RULE)
    print(f"{len(log):,} trades. Largest {count} by cashflow:\n")
    for _, trade in log.nlargest(count, "net_revenue_aud").iterrows():
        print(
            f"  {trade['start']:%Y-%m-%d %H:%M} -> {trade['end']:%H:%M}"
            f"   {trade['action']:>9}"
            f"   {trade['energy_mwh']:>7,.1f} MWh"
            f"   ${trade['net_revenue_aud']:>10,.0f}"
        )
        print(f"      {trade['reason']}")


def _print_daily(backtest: BacktestResult) -> None:
    daily = daily_summary(backtest.intervals)
    print(RULE)
    print("trading day      net revenue    bought     sold   min RRP   max RRP")
    for day, row in daily.iterrows():
        print(
            f"  {day:%Y-%m-%d}  ${row['net_revenue_aud']:>12,.2f}"
            f"  {row['energy_bought_mwh']:>7,.0f}  {row['energy_sold_mwh']:>7,.0f}"
            f"  ${row['min_price']:>7,.2f}  ${row['max_price']:>7,.2f}"
        )


def _print_day(backtest: BacktestResult, day: date) -> None:
    stamp = pd.Timestamp(day)
    frame = backtest.intervals[
        backtest.intervals["trading_day"].dt.date == day
    ]
    if frame.empty:
        available = backtest.intervals["trading_day"].dt.date
        raise SystemExit(
            f"no trading day {day}. Available: {available.min()} to {available.max()}"
        )

    print(RULE)
    print(f"trading day {stamp:%Y-%m-%d}, active intervals only\n")
    for timestamp, row in frame[frame["action"] != "idle"].iterrows():
        print(f"  {timestamp:%H:%M}  {explain_interval(row)}")
    print(f"\n  day total ${frame['net_revenue_aud'].sum():,.2f}")


def _export(backtest: BacktestResult, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    written = {
        "intervals.csv": backtest.intervals,
        "trades.csv": build_trade_log(backtest.intervals),
        "daily.csv": daily_summary(backtest.intervals),
    }
    print(RULE)
    for name, frame in written.items():
        path = directory / name
        frame.to_csv(path)
        print(f"  wrote {path}  ({len(frame):,} rows)")


if __name__ == "__main__":
    main()
