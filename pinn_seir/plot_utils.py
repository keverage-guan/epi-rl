"""Shared plotting helpers: regime-switch markers on a model-week or date axis."""

from __future__ import annotations

from typing import List

from .schedules import _parse_date


def switch_dates(mcfg) -> List[str]:
    """Configured change-point date(s) as a list, tolerating None / str / sequence."""
    raw = getattr(mcfg, "regime_switch_date", None)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [d for d in raw if d is not None]


def switch_weeks(mcfg) -> List[float]:
    """Change points in fractional model-week coordinates from epidemic_start."""
    start = _parse_date(mcfg.epidemic_start)
    dpw = float(mcfg.days_per_week)
    return [(_parse_date(d) - start).days / dpw for d in switch_dates(mcfg)]


def mark_switches(ax, mcfg, use_dates: bool = False, label: bool = True) -> None:
    """Draw a dashed red vertical line at each regime change point.

    Call BEFORE ax.legend() or the label will not appear. Only the first line is
    labelled, so multiple change points produce one legend entry.
    """
    horizon_weeks = float(mcfg.n_weeks)
    for i, (d, w) in enumerate(zip(switch_dates(mcfg), switch_weeks(mcfg))):
        if not (0.0 <= w <= horizon_weeks):
            continue                      # outside the plotted horizon
        x = _parse_date(d) if use_dates else w
        ax.axvline(
            x, color="red", ls="--", lw=1.2, alpha=0.9, zorder=4,
            label=f"regime switch ({d})" if (label and i == 0) else None,
        )