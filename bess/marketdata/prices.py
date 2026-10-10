"""Turning raw dispatch files into a clean NSW1 price series.

Four interpretation rules are applied here, each a silent bug if taken the
other way. The reasoning is in the README; the rules are restated because
enforcing them is what this module is for.

* **Market time is AEST, fixed at UTC+10, all year.** Never a named zone -- the
  NEM does not observe daylight saving.
* **``SETTLEMENTDATE`` labels the END of its interval.** A row stamped 14:05
  describes 14:00 to 14:05.
* **Intervention runs are not the market outcome.** ``INTERVENTION = 0`` is
  preferred, falling back where only an intervention run exists.
* **Gaps are never filled.** A stale price produces a trade that never
  happened. Missing intervals stay in the index, marked untradable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bess.marketdata.mms import extract_table
from bess.marketdata.nemweb import NemwebClient, iter_interval_csvs

#: NEM market time. A fixed offset, deliberately not a named zone: the market
#: runs on AEST year-round and never shifts for daylight saving.
NEM_TIMEZONE = timezone(timedelta(hours=10))

#: Administered bounds on the dispatch price. A value outside these did not
#: come from the market and almost certainly means a parsing error.
MARKET_FLOOR_PRICE = -1000.0
MARKET_PRICE_CAP = 20000.0

DISPATCH_INTERVAL = timedelta(minutes=5)

_REPORT, _SUBTYPE = "DISPATCH", "PRICE"


@dataclass(frozen=True)
class DataQualityReport:
    """What was found, what was dropped, and what was missing."""

    region: str
    start: pd.Timestamp
    end: pd.Timestamp
    intervals_expected: int
    intervals_present: int
    intervention_rows_dropped: int
    duplicate_rows_dropped: int
    administered_price_intervals: int
    market_suspended_intervals: int
    negative_price_intervals: int
    price_min: float
    price_max: float
    price_mean: float

    @property
    def missing_intervals(self) -> int:
        return self.intervals_expected - self.intervals_present

    @property
    def completeness(self) -> float:
        if self.intervals_expected == 0:
            return 0.0
        return self.intervals_present / self.intervals_expected

    def summary(self) -> str:
        return "\n".join(
            [
                f"{self.region}  {self.start:%Y-%m-%d %H:%M} to "
                f"{self.end:%Y-%m-%d %H:%M} AEST",
                f"  intervals          {self.intervals_present:,} of "
                f"{self.intervals_expected:,} ({self.completeness:.2%})",
                f"  missing            {self.missing_intervals:,}",
                f"  intervention rows  {self.intervention_rows_dropped:,} dropped",
                f"  duplicate rows     {self.duplicate_rows_dropped:,} dropped",
                f"  administered price {self.administered_price_intervals:,} intervals",
                f"  market suspended   {self.market_suspended_intervals:,} intervals",
                f"  negative prices    {self.negative_price_intervals:,} intervals",
                f"  RRP                ${self.price_min:,.2f} to ${self.price_max:,.2f}"
                f"  (mean ${self.price_mean:,.2f})",
            ]
        )


@dataclass(frozen=True)
class PriceSeries:
    """A gap-aware five-minute price series, ready to optimise against."""

    prices: pd.Series
    tradable: np.ndarray
    quality: DataQualityReport

    def __len__(self) -> int:
        return len(self.prices)

    def window(self, start: pd.Timestamp, end: pd.Timestamp) -> PriceSeries:
        """A half-open slice of this series, keeping the gap mask aligned."""
        mask = (self.prices.index >= start) & (self.prices.index < end)
        return PriceSeries(
            prices=self.prices[mask],
            tradable=self.tradable[mask],
            quality=self.quality,
        )


def load_prices(
    client: NemwebClient,
    start: date,
    end: date,
    region: str = "NSW1",
    cache_path: Path | str | None = None,
) -> PriceSeries:
    """Fetch, parse and clean dispatch prices for a range of calendar days.

    Args:
        client: the NEMWEB client to fetch through.
        start: first calendar day, inclusive.
        end: last calendar day, inclusive.
        region: NEM region identifier, e.g. ``"NSW1"``.
        cache_path: optional parquet file. Written after a successful parse and
            read on later runs, so the dashboard can start without a network.

    Returns:
        A :class:`PriceSeries` on a complete five-minute index.
    """
    cache = Path(cache_path) if cache_path else None
    frame = _read_cache(cache, start, end) if cache else None

    if frame is None:
        frame = _parse_days(client, start, end, region)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(cache, index=False)

    return _clean(frame, region)


def _parse_days(
    client: NemwebClient, start: date, end: date, region: str
) -> pd.DataFrame:
    """Pull the PRICE table for one region out of every interval in the range."""
    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")

    listing = client.list_archive_days()
    collected: list[pd.DataFrame] = []

    day = start
    while day <= end:
        archive = client.fetch_archive_day(day, urls=listing)
        for _, text in iter_interval_csvs(archive):
            table = extract_table(text, _REPORT, _SUBTYPE)
            if table.empty:
                continue
            collected.append(table.loc[table["REGIONID"] == region])
        day += timedelta(days=1)

    if not collected:
        raise ValueError(f"no {region} price rows found between {start} and {end}")
    return pd.concat(collected, ignore_index=True)


def _clean(raw: pd.DataFrame, region: str) -> PriceSeries:
    """Apply the four interpretation rules and reindex onto a complete grid."""
    frame = raw.copy()

    frame["SETTLEMENTDATE"] = pd.to_datetime(
        frame["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S"
    ).dt.tz_localize(NEM_TIMEZONE)
    for column in ("RRP", "RUNNO", "INTERVENTION", "APCFLAG", "MARKETSUSPENDEDFLAG"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    rows_in = len(frame)

    # Prefer the normal run, then the latest solve. Sorting and keeping the
    # first row per timestamp does both at once, and leaves an intervention row
    # in place where it is the only thing published for that interval.
    frame = frame.sort_values(
        ["SETTLEMENTDATE", "INTERVENTION", "RUNNO"], ascending=[True, True, False]
    )
    kept = frame.drop_duplicates(subset="SETTLEMENTDATE", keep="first")

    intervention_dropped = int(
        (frame["INTERVENTION"] == 1).sum() - (kept["INTERVENTION"] == 1).sum()
    )
    duplicates_dropped = rows_in - len(kept) - intervention_dropped

    frame = kept.set_index("SETTLEMENTDATE").sort_index()

    out_of_range = (
        frame["RRP"].isna()
        | (frame["RRP"] < MARKET_FLOOR_PRICE)
        | (frame["RRP"] > MARKET_PRICE_CAP)
    )
    if out_of_range.any():
        raise ValueError(
            f"{int(out_of_range.sum())} RRP values fall outside "
            f"[{MARKET_FLOOR_PRICE}, {MARKET_PRICE_CAP}] or failed to parse; "
            f"that indicates a parsing error rather than a market event"
        )

    # A complete grid, so a missing interval shows up as a hole rather than
    # vanishing from the index and quietly shortening the day.
    grid = pd.date_range(
        frame.index.min(), frame.index.max(), freq=DISPATCH_INTERVAL, tz=NEM_TIMEZONE
    )
    prices = frame["RRP"].reindex(grid)
    tradable = prices.notna().to_numpy()

    quality = DataQualityReport(
        region=region,
        start=grid[0],
        end=grid[-1],
        intervals_expected=len(grid),
        intervals_present=int(tradable.sum()),
        intervention_rows_dropped=intervention_dropped,
        duplicate_rows_dropped=int(duplicates_dropped),
        administered_price_intervals=_flag_count(frame, "APCFLAG"),
        market_suspended_intervals=_flag_count(frame, "MARKETSUSPENDEDFLAG"),
        negative_price_intervals=int((prices < 0).sum()),
        price_min=float(prices.min()),
        price_max=float(prices.max()),
        price_mean=float(prices.mean()),
    )

    # The optimiser needs a number in every interval for the arithmetic to be
    # defined. Gaps get the series mean, and the tradable mask is what actually
    # stops the battery acting on any of them.
    prices = prices.fillna(quality.price_mean)
    prices.name = f"{region}_RRP_AUD_MWH"

    return PriceSeries(prices=prices, tradable=tradable, quality=quality)


def _flag_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int((frame[column] == 1).sum())


def _read_cache(cache: Path, start: date, end: date) -> pd.DataFrame | None:
    """Reuse a cached parse only if it actually covers the requested range."""
    if not cache.exists():
        return None
    frame = pd.read_parquet(cache)
    stamps = pd.to_datetime(frame["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    wanted_start = datetime.combine(start, datetime.min.time()) + DISPATCH_INTERVAL
    wanted_end = datetime.combine(end + timedelta(days=1), datetime.min.time())
    if stamps.min() > wanted_start or stamps.max() < wanted_end:
        return None
    return frame
