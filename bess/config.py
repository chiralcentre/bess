"""Every assumption in the model, in one place.

Changing a number here changes it everywhere. The reasoning behind each choice
is in the README under "Assumptions"; vary one for a sensitivity run with
:func:`dataclasses.replace`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from bess.asset.battery import BatterySpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class MarketConfig:
    """Which market to operate in."""

    region: str


@dataclass(frozen=True)
class BacktestConfig:
    """The window to optimise over, and where one trading day ends."""

    start_date: date
    end_date: date
    trading_day_start_hour: int

    def __post_init__(self) -> None:
        if self.end_date < self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) is before start_date ({self.start_date})"
            )
        if not 0 <= self.trading_day_start_hour <= 23:
            raise ValueError("trading_day_start_hour must be an hour of the day")


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    price_cache: Path


@dataclass(frozen=True)
class Config:
    battery: BatterySpec
    market: MarketConfig
    backtest: BacktestConfig
    paths: Paths


BATTERY = BatterySpec(
    capacity_mwh=200.0,          # given by the brief
    max_charge_mw=50.0,          # 200/50 = a four-hour battery
    max_discharge_mw=50.0,
    charge_efficiency=0.90,      # applied per leg, so 81% round-trip
    discharge_efficiency=0.90,
    initial_soc_mwh=0.0,         # starts empty: nothing to sell that was not bought
    degradation_cost_per_mwh=5.0,
    interval_minutes=5,          # five-minute settlement since October 2021
)

MARKET = MarketConfig(region="NSW1")

BACKTEST = BacktestConfig(
    start_date=date(2026, 9, 9),
    end_date=date(2026, 10, 8),
    trading_day_start_hour=4,    # the NEM trading day runs 04:00-03:59 AEST
)

PATHS = Paths(
    raw_dir=PROJECT_ROOT / "data" / "raw",
    price_cache=PROJECT_ROOT
    / "data"
    / "sample"
    / "nsw1_dispatch_price_20260909_20261008.parquet",
)

# The configuration everything reads from.
CONFIG = Config(battery=BATTERY, market=MARKET, backtest=BACKTEST, paths=PATHS)
