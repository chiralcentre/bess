"""Fetching raw dispatch files from AEMO's public NEMWEB server.

Targets the daily ``ARCHIVE`` files. A thirty-day backtest is thirty downloads
there, against roughly 8,640 from ``Current``. Each daily file is a
**zip of 288 zips**, one per dispatch interval, each holding a single CSV.

Everything is cached to disk by filename, so later runs are offline and
instant. The README covers the trade-offs between the two directories.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

import requests

NEMWEB_BASE = "https://nemweb.com.au"
ARCHIVE_PATH = "/Reports/ARCHIVE/DispatchIS_Reports/"

#: The directory listing is IIS-generated and uses uppercase ``<A HREF=`` --
#: a lowercase-only pattern matches nothing at all, silently.
_LINK_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)
_ARCHIVE_FILE_RE = re.compile(r"PUBLIC_DISPATCHIS_(\d{8})\.zip$", re.IGNORECASE)

DEFAULT_USER_AGENT = "bess-arbitrage-study/1.0 (educational backtest)"


class NemwebError(RuntimeError):
    """A file could not be listed or retrieved."""


class NemwebClient:
    """Lists and downloads archive files, caching them on disk."""

    def __init__(
        self,
        cache_dir: Path | str = Path("data/raw"),
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    # -- listing ------------------------------------------------------------

    def list_archive_days(self) -> dict[date, str]:
        """Map each available calendar day to the URL of its daily file.

        The archive is listed, then read. Days do go missing, and a
        constructed URL turns that into a 404 mid-run, where a listing says
        "this day is not available" up front.
        """
        html = self._get(NEMWEB_BASE + ARCHIVE_PATH).decode("utf-8", "replace")
        days: dict[date, str] = {}
        for href in _LINK_RE.findall(html):
            match = _ARCHIVE_FILE_RE.search(href)
            if match:
                day = datetime.strptime(match.group(1), "%Y%m%d").date()
                days[day] = href if href.startswith("http") else NEMWEB_BASE + href
        if not days:
            raise NemwebError(
                f"no dispatch files found in the archive listing at "
                f"{NEMWEB_BASE + ARCHIVE_PATH}; the page layout may have changed"
            )
        return days

    # -- fetching -----------------------------------------------------------

    def fetch_archive_day(self, day: date, urls: dict[date, str] | None = None) -> Path:
        """Return a local path to one day's file, downloading it if needed.

        Args:
            day: the calendar day to fetch.
            urls: a listing from :meth:`list_archive_days`, to avoid re-listing
                the directory once per day in a loop.
        """
        filename = f"PUBLIC_DISPATCHIS_{day:%Y%m%d}.zip"
        destination = self.cache_dir / filename
        if destination.exists() and destination.stat().st_size > 0:
            return destination

        listing = urls if urls is not None else self.list_archive_days()
        if day not in listing:
            raise NemwebError(
                f"{day:%Y-%m-%d} is not in the NEMWEB archive. It may be too "
                f"recent (the archive runs about two days behind) or too old."
            )

        payload = self._get(listing[day])
        # Write via a temporary name so an interrupted download can never be
        # mistaken for a complete cache entry on the next run.
        partial = destination.with_suffix(".zip.part")
        partial.write_bytes(payload)
        partial.replace(destination)
        return destination

    def _get(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_seconds * (2**attempt))
        raise NemwebError(f"could not retrieve {url}: {last_error}") from last_error


def iter_interval_csvs(archive_zip: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(name, text)`` for every dispatch interval in a daily archive.

    Handles the nesting: the daily file contains one zip per interval, and each
    of those contains one CSV. Older or hand-assembled files occasionally hold
    the CSV directly, so both shapes are accepted.
    """
    with zipfile.ZipFile(archive_zip) as outer:
        for name in sorted(outer.namelist()):
            payload = outer.read(name)
            if payload[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                    for inner_name in inner.namelist():
                        yield inner_name, inner.read(inner_name).decode("utf-8", "replace")
            else:
                yield name, payload.decode("utf-8", "replace")
