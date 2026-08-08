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
    start = _parse_date(epidemic_start)
    ranges = [(_parse_date(a), _parse_date(b)) for a, b in holiday_ranges]

    cal = np.ones((n_weeks, n_patches), dtype=np.float32)
    for k in range(n_weeks):
        week_start = start + timedelta(days=days_per_week * k)
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

class DailyCalendar:

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

        parsed: Dict[str, List[Tuple[date, date]]] = {}
        for nation, ranges in holiday_ranges_by_nation.items():
            parsed[nation] = [(_parse_date(a), _parse_date(b)) for a, b in ranges]

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
        self.table = table

    def day_of(self, tau: np.ndarray) -> np.ndarray:
        d = np.floor(np.asarray(tau) * self.days_per_week).astype(np.int64)
        return np.clip(d, 0, self.days_per_week - 1)

    def term_time(self, week: np.ndarray, tau: np.ndarray) -> np.ndarray:
        week = np.asarray(week, dtype=np.int64)
        d = self.day_of(tau)
        return self.table[:, week, d].T

    def transition_taus(self, week_k: int, patch: int = None) -> List[float]:
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
    cal = np.ones((n_weeks, n_patches), dtype=np.float32)
    k = min(budget_weeks, n_weeks)
    for p in range(n_patches):
        closed_weeks = rng.choice(n_weeks, size=k, replace=False)
        cal[closed_weeks, p] = 0.0
    return cal

def random_fraction_closed(
    n_weeks: int,
    n_patches: int,
    rng: np.random.Generator,
    min_frac: float = 0.0,
    max_frac: float = 1.0,
) -> np.ndarray:
    frac = rng.uniform(min_frac, max_frac)
    n_closed = int(round(frac * n_patches))
    cal = np.ones((n_weeks, n_patches), dtype=np.float32)
    if n_closed > 0:
        closed_patches = rng.choice(n_patches, size=n_closed, replace=False)
        block_len = int(rng.integers(1, n_weeks + 1))
        start = int(rng.integers(0, n_weeks - block_len + 1))
        cal[start:start + block_len, closed_patches] = 0.0
    return cal

class ScheduleSampler:

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
        self.true = all_open(n_weeks, n_patches)

    def sample(self, n_schedules: int) -> np.ndarray:
        out: List[np.ndarray] = [self.true]
        if self.include_all_closed and len(out) < n_schedules:
            out.append(all_closed(self.n_weeks, self.n_patches))
        while len(out) < n_schedules:
            out.append(
                random_fraction_closed(self.n_weeks, self.n_patches, self.rng)
            )
        return np.stack(out[:n_schedules], axis=0)
