"""Closure-schedule sampling for physics/junction generalisation.

The data loss is tied to the single true 2009 calendar, but physics and junction
losses are enforced over a *distribution* of weekly closure schedules so the network
generalises to interventions it never saw in data -- the space a downstream RL agent
will search.

A schedule is a (n_weeks, P) array in {0,1}: 1 = schools OPEN, 0 = CLOSED.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def holiday_calendar(
    n_weeks: int,
    n_patches: int,
    epidemic_start: str,
    holiday_ranges: Sequence[Tuple[str, str]],
    days_per_week: int = 7,
    overlap_threshold: float = 0.5,
) -> np.ndarray:
    """Build the realised school calendar from real holiday date ranges.

    Model week ``k`` spans ``[start + 7k, start + 7(k+1))``. A week is marked CLOSED
    (schools on holiday, value 0) when the fraction of its days that fall inside any
    holiday range is at least ``overlap_threshold``; otherwise it is OPEN (value 1).
    The same calendar is applied uniformly across all patches.

    Using fractional overlap (rather than "touches at all") keeps boundary weeks that
    are only a day or two into a holiday classified as term-time, which matches how the
    weekly term/holiday contact matrix should switch.
    """
    start = _parse_date(epidemic_start)
    ranges = [(_parse_date(a), _parse_date(b)) for a, b in holiday_ranges]

    cal = np.ones((n_weeks, n_patches), dtype=np.float32)
    for k in range(n_weeks):
        week_start = start + timedelta(days=days_per_week * k)
        # count holiday days within this week
        holiday_days = 0
        for d in range(days_per_week):
            day = week_start + timedelta(days=d)
            if any(a <= day <= b for a, b in ranges):
                holiday_days += 1
        if holiday_days / days_per_week >= overlap_threshold:
            cal[k, :] = 0.0
    return cal


def all_open(n_weeks: int, n_patches: int) -> np.ndarray:
    return np.ones((n_weeks, n_patches), dtype=np.float32)


def holiday_week_spans(
    epidemic_start: str,
    holiday_ranges: Sequence[Tuple[str, str]],
    days_per_week: int = 7,
) -> List[Tuple[float, float]]:
    """Return each holiday range as a (start_week, end_week) span in model-week units.

    These are *fractional* week coordinates measured from ``epidemic_start`` (week 0),
    intended for shading holiday periods on plots so the band matches the true calendar
    dates exactly -- independent of the term/holiday switch threshold used for the
    contact matrices. A holiday running from date a to date b (inclusive) spans
    ``[(a - start)/7, (b + 1 - start)/7]`` so the band covers the full inclusive range.
    """
    start = _parse_date(epidemic_start)
    spans: List[Tuple[float, float]] = []
    for a_str, b_str in holiday_ranges:
        a = _parse_date(a_str)
        b = _parse_date(b_str)
        w0 = (a - start).days / days_per_week
        w1 = (b + timedelta(days=1) - start).days / days_per_week
        spans.append((w0, w1))
    return spans


def all_closed(n_weeks: int, n_patches: int) -> np.ndarray:
    return np.zeros((n_weeks, n_patches), dtype=np.float32)


def random_budgeted(
    n_weeks: int,
    n_patches: int,
    budget_weeks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """A random schedule closing exactly ``budget_weeks`` weeks per patch."""
    cal = np.ones((n_weeks, n_patches), dtype=np.float32)
    k = min(budget_weeks, n_weeks)
    for p in range(n_patches):
        closed_weeks = rng.choice(n_weeks, size=k, replace=False)
        cal[closed_weeks, p] = 0.0
    return cal


class ScheduleSampler:
    """Draws a batch of schedules, always including the true calendar first."""

    def __init__(
        self,
        n_weeks: int,
        n_patches: int,
        budget_weeks: int,
        epidemic_start: str,
        holiday_ranges: Sequence[Tuple[str, str]],
        include_all_open: bool = True,
        include_all_closed: bool = True,
        seed: int = 0,
    ) -> None:
        self.n_weeks = n_weeks
        self.n_patches = n_patches
        self.budget_weeks = budget_weeks
        self.include_all_open = include_all_open
        self.include_all_closed = include_all_closed
        self.rng = np.random.default_rng(seed)
        self.true = holiday_calendar(
            n_weeks, n_patches, epidemic_start, holiday_ranges
        )

    def sample(self, n_schedules: int) -> np.ndarray:
        """Return (n_schedules, n_weeks, P). Index 0 is always the true calendar."""
        out: List[np.ndarray] = [self.true]
        if self.include_all_open and len(out) < n_schedules:
            out.append(all_open(self.n_weeks, self.n_patches))
        if self.include_all_closed and len(out) < n_schedules:
            out.append(all_closed(self.n_weeks, self.n_patches))
        while len(out) < n_schedules:
            out.append(
                random_budgeted(
                    self.n_weeks, self.n_patches, self.budget_weeks, self.rng
                )
            )
        return np.stack(out[:n_schedules], axis=0)