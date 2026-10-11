"""Parser for AEMO's MMS flat-file CSV format.

A file holds several tables, each introduced by its own ``I`` header row::

    I,DISPATCH,PRICE,5,SETTLEMENTDATE,RUNNO,REGIONID,...
    D,DISPATCH,PRICE,5,"2026/10/08 00:05:00",1,NSW1,...

**Columns must be read from the ``I`` row, and not fixed indices.** Versions
differ within one file, so a hard-coded index works until AEMO adds a column
and then silently reads the wrong one.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import pandas as pd

#: The four fields every row begins with: type, report, subtype, version.
_PREFIX_FIELDS = 4


class MMSParseError(ValueError):
    """The file was not in the shape an MMS CSV is supposed to be."""


@dataclass
class TableAccumulator:
    """Collects ``D`` rows for one table, tolerating a mid-file version change."""

    columns: list[str] | None = None
    rows: list[list[str]] = field(default_factory=list)
    frames: list[pd.DataFrame] = field(default_factory=list)

    def start_version(self, columns: list[str]) -> None:
        """A new ``I`` row arrived. Bank what we have and switch schema."""
        self._bank()
        self.columns = columns

    def add(self, values: list[str]) -> None:
        self.rows.append(values)

    def _bank(self) -> None:
        if self.columns is not None and self.rows:
            self.frames.append(pd.DataFrame(self.rows, columns=self.columns))
        self.rows = []

    def to_frame(self) -> pd.DataFrame:
        self._bank()
        if not self.frames:
            return pd.DataFrame()
        # Versions differ only by added columns, so aligning on name and
        # letting the older rows carry NaN is the correct reconciliation.
        return pd.concat(self.frames, ignore_index=True)


def extract_table(text: str, report: str, subtype: str) -> pd.DataFrame:
    """Pull a single named table out of an MMS file.

    Args:
        text: the decoded contents of one MMS CSV.
        report: the report name, e.g. ``"DISPATCH"``.
        subtype: the table name within it, e.g. ``"PRICE"``.

    Returns:
        A DataFrame of string values with the table's own column names. Empty
        if the file contains no such table. Type conversion is left to the
        caller, which knows what the columns mean.
    """
    info_prefix = f"I,{report},{subtype},"
    data_prefix = f"D,{report},{subtype},"
    acc = TableAccumulator()

    for line in text.splitlines():
        # A cheap prefix test first. A DispatchIS file is mostly CONSTRAINT
        # rows we do not want, and CSV-parsing all of them is wasted work when
        # this runs over 8,640 files.
        if line.startswith(data_prefix):
            if acc.columns is None:
                raise MMSParseError(
                    f"data row for {report},{subtype} appeared before its header row"
                )
            values = _split(line)[_PREFIX_FIELDS:]
            # Trailing empty fields are sometimes omitted; pad to the schema.
            if len(values) < len(acc.columns):
                values = values + [""] * (len(acc.columns) - len(values))
            acc.add(values[: len(acc.columns)])
        elif line.startswith(info_prefix):
            acc.start_version(_split(line)[_PREFIX_FIELDS:])

    return acc.to_frame()


def _split(line: str) -> list[str]:
    """Split one CSV line, honouring the quoting AEMO uses around timestamps."""
    return next(csv.reader(io.StringIO(line)))
