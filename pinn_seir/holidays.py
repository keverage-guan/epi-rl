"""Load the fixed historical school-holiday calendar from a data file.

The per-nation holiday ranges used to be hard-coded in :mod:`pinn_seir.config`.
They now live in a CSV alongside the other Great Britain inputs
(``data/great_brittain/school_holidays.csv``) so the calendar can be swapped,
extended to other nations, or versioned as data rather than as code.

File format (header required)::

    nation,holiday,start_date,end_date
    Scotland,summer,2009-07-03,2009-08-16
    England,autumn half-term,2009-10-26,2009-11-01

* ``nation``      must match the nation names in ``crosswalk.tsv``.
* ``holiday``     free-text label; ignored by the model, kept for readability.
* ``start_date``  / ``end_date``: ISO ``YYYY-MM-DD``, **inclusive** on both ends
  (the same convention ``schedules.DailyCalendar`` and
  ``schedules.holiday_week_spans`` already assume).

Ranges are returned sorted by start date within each nation.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

REQUIRED_COLUMNS = ("nation", "start_date", "end_date")

HolidayRanges = Dict[str, List[Tuple[str, str]]]


def _parse_date(s: str, *, column: str, row: int) -> date:
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover - message clarity only
        raise ValueError(
            f"Bad {column} '{s}' on data row {row} of the holiday file: "
            "expected ISO format YYYY-MM-DD."
        ) from exc


def load_holiday_ranges(path: Path | str) -> HolidayRanges:
    """Read a school-holiday CSV into ``{nation: [(start, end), ...]}``.

    Dates are returned as ISO strings so the result is a drop-in replacement for
    the dict that used to be built in ``ModelConfig``; downstream code
    (``schedules.DailyCalendar``, ``schedules.holiday_week_spans``) parses them.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"School-holiday file not found: {path}. Expected a CSV with columns "
            f"{', '.join(REQUIRED_COLUMNS)}."
        )

    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}
    missing = [c for c in REQUIRED_COLUMNS if c not in lower]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {missing}. "
            f"Found: {list(df.columns)}"
        )

    by_nation: Dict[str, List[Tuple[date, date]]] = {}
    for i, row in df.iterrows():
        nation = str(row[lower["nation"]]).strip()
        if not nation or nation.lower() == "nan":
            raise ValueError(f"Empty nation on data row {i} of {path}.")
        a = _parse_date(row[lower["start_date"]], column="start_date", row=int(i))
        b = _parse_date(row[lower["end_date"]], column="end_date", row=int(i))
        if b < a:
            raise ValueError(
                f"Holiday on data row {i} of {path} ends ({b}) before it starts ({a})."
            )
        by_nation.setdefault(nation, []).append((a, b))

    if not by_nation:
        raise ValueError(f"{path} contains no holiday rows.")

    out: HolidayRanges = {}
    for nation, ranges in by_nation.items():
        ranges.sort(key=lambda ab: ab[0])
        _warn_on_overlap(nation, ranges, path)
        out[nation] = [(a.isoformat(), b.isoformat()) for a, b in ranges]
    return out


def _warn_on_overlap(
    nation: str, ranges: Sequence[Tuple[date, date]], path: Path
) -> None:
    """Overlapping ranges are harmless (the calendar ORs them) but usually a typo."""
    for (a0, b0), (a1, b1) in zip(ranges, ranges[1:]):
        if a1 <= b0:
            import warnings

            warnings.warn(
                f"Overlapping holiday ranges for {nation} in {path}: "
                f"({a0}, {b0}) and ({a1}, {b1}).",
                stacklevel=3,
            )


def holiday_days(ranges: Sequence[Tuple[str, str]]) -> set:
    """Expand ISO ``(start, end)`` pairs into the inclusive set of dates covered."""
    from datetime import timedelta

    days: set = set()
    for a_str, b_str in ranges:
        a = datetime.strptime(a_str, "%Y-%m-%d").date()
        b = datetime.strptime(b_str, "%Y-%m-%d").date()
        d = a
        while d <= b:
            days.add(d)
            d += timedelta(days=1)
    return days