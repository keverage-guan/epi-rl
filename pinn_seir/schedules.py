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


# --------------------------------------------------------------------------- #
# Fixed historical calendar at DAILY resolution
# --------------------------------------------------------------------------- #
class DailyCalendar:
    """Fixed historical school calendar, per patch, queried at day resolution.

    Each patch (district) inherits its nation's holiday ranges. ``term_time`` returns
    1.0 if the day at local time ``tau`` within week ``k`` is term-time for that patch,
    else 0.0. The day within a week is ``floor(tau * days_per_week)`` (tau in [0, 1)).

    This is the *historical* calendar only. The controllable POLICY closure is a
    separate weekly, per-patch quantity; schools are open in a patch on a day iff
    term-time AND not policy-closed (combined in the physics residual, not here).
    """

    def __init__(
        self,
        epidemic_start: str,
        holiday_ranges_by_nation: Dict[str, Sequence[Tuple[str, str]]],
        district_nations: Sequence[str],
        n_weeks: int,
        days_per_week: int = 7,
    ) -> None:
        self.start = _parse_date(epidemic_start)
        self.n_weeks = n_weeks
        self.days_per_week = days_per_week
        self.district_nations = list(district_nations)
        P = len(self.district_nations)

        # Parse each nation's ranges once.
        parsed: Dict[str, List[Tuple[date, date]]] = {}
        for nation, ranges in holiday_ranges_by_nation.items():
            parsed[nation] = [(_parse_date(a), _parse_date(b)) for a, b in ranges]

        # Precompute a per-patch (P, n_weeks, days_per_week) term-time table.
        table = np.ones((P, n_weeks, days_per_week), dtype=np.float32)
        for p, nation in enumerate(self.district_nations):
            ranges = parsed.get(nation)
            if ranges is None:
                raise ValueError(
                    f"No holiday ranges configured for nation '{nation}' "
                    f"(patch {p}). Configured nations: {sorted(parsed)}."
                )
            for k in range(n_weeks):
                for d in range(days_per_week):
                    day = self.start + timedelta(days=days_per_week * k + d)
                    if any(a <= day <= b for a, b in ranges):
                        table[p, k, d] = 0.0
        self.table = table  # (P, n_weeks, days_per_week)

    def day_of(self, tau: np.ndarray) -> np.ndarray:
        """Map local time tau in [0,1) to a day index 0..days_per_week-1."""
        d = np.floor(np.asarray(tau) * self.days_per_week).astype(np.int64)
        return np.clip(d, 0, self.days_per_week - 1)

    def term_time(self, week: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Vectorised term-time lookup, returning (len, P).

        week: int array (len,), tau: float array (len,) in [0,1). Returns term-time
        for every patch at each queried (week, tau).
        """
        week = np.asarray(week, dtype=np.int64)
        d = self.day_of(tau)
        return self.table[:, week, d].T  # (len, P)

    def transition_taus(self, week_k: int, patch: int = None) -> List[float]:
        """Local-time points in (0,1) where term-time flips within week ``week_k``.

        If ``patch`` is given, transitions for that patch; otherwise the union across
        all patches (any patch that switches). Used for reference/inspection only --
        mid-week switches are handled by per-collocation-point effective_open, not by
        junctions.
        """
        if patch is not None:
            rows = [self.table[patch, week_k]]
        else:
            rows = [self.table[p, week_k] for p in range(self.table.shape[0])]
        taus = set()
        for row in rows:
            for d in range(1, self.days_per_week):
                if row[d] != row[d - 1]:
                    taus.add(d / self.days_per_week)
        return sorted(taus)


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
    """Draws a batch of weekly POLICY-closure schedules.

    A schedule is (n_weeks, P) in {0,1}: 1 = schools NOT policy-closed (open, subject
    to the calendar), 0 = policy-closed. Index 0 of a sampled batch is always the true
    2009 policy, which is all-open (the historical school holidays are handled by the
    fixed daily calendar, not by policy). Physics/junction losses train over this
    distribution so the surrogate generalises across the RL action space; the data loss
    uses only the true (all-open) policy.
    """

    def __init__(
        self,
        n_weeks: int,
        n_patches: int,
        budget_weeks: int,
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
        # True historical policy: no mandated closures -> all open.
        self.true = all_open(n_weeks, n_patches)

    def sample(self, n_schedules: int) -> np.ndarray:
        """Return (n_schedules, n_weeks, P). Index 0 is always the true (all-open) policy."""
        out: List[np.ndarray] = [self.true]
        if self.include_all_closed and len(out) < n_schedules:
            out.append(all_closed(self.n_weeks, self.n_patches))
        while len(out) < n_schedules:
            out.append(
                random_budgeted(
                    self.n_weeks, self.n_patches, self.budget_weeks, self.rng
                )
            )
        return np.stack(out[:n_schedules], axis=0)