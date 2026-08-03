"""Split the weekly ILI series into train/dev/test, stratified by school-holiday status.

Each row of ``uk_flu_per_100000.csv`` is one epidemiological week, identified by its
``week_end_date`` (M/D/Y). The week is taken to span the seven inclusive days
``[week_end_date - 6, week_end_date]``.

For every modelled nation we compute the fraction of those seven days that fall inside
one of that nation's school-holiday ranges (from
``data/great_brittain/school_holidays.csv``). A nation is "on holiday" that week when
that fraction is at least ``--nation-threshold``. The row's binary stratum label is then
formed across nations by ``--rule``:

    any        holiday if at least one nation is on holiday   (default)
    majority   holiday if more than half of the nations are
    all        holiday only if every nation is

Rows are shuffled with a seeded RNG and allocated within each stratum by largest
remainder, so the holiday / term-time proportion in each of train, dev and test matches
the full series as closely as integer counts allow. Where a stratum is big enough, every
requested split is guaranteed at least one row of it.

Usage
-----
    python -m pinn_seir.split_epidemic_data \
        --flu      data/epidemic/uk_flu_per_100000.csv \
        --holidays data/great_brittain/school_holidays.csv \
        --split 0.70 0.15 0.15 --seed 0 \
        --out-dir  data/epidemic/splits

Writes ``<stem>_train.csv``, ``<stem>_dev.csv``, ``<stem>_test.csv`` (original columns,
original row order preserved within each split) and ``<stem>_split_labels.csv`` (every
row plus the ``is_holiday``, ``n_nations_on_holiday`` and ``split`` columns, for
auditing). Pass ``--split 0.75 0 0.25`` for a plain two-way split; the dev file is then
not written.

CAVEAT: this is a random split of a time series. It is the right thing for assessing
interpolation across the observed period -- and stratifying on holiday status stops a
lucky draw from putting the entire summer-holiday trough on one side -- but it leaks
temporal neighbours between splits. For forecasting claims, hold out contiguous blocks
at the end instead (``--holdout-tail DEV TEST``).

With ~43 weeks of data a 15% slice is only 6-7 rows, of which roughly 2 are holiday
weeks. Model selection on a dev set that small is noisy; treat differences of one or two
rows' worth of error as unresolvable.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# Allow both `python -m scripts.split_epidemic_data` and `python scripts/...py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinn_seir.holidays import holiday_days, load_holiday_ranges  # noqa: E402

DEFAULT_NATIONS = ("england", "scotland", "wales")
DAYS_PER_WEEK = 7
SPLIT_NAMES = ("train", "dev", "test")


# --------------------------------------------------------------------------- #
# Labelling
# --------------------------------------------------------------------------- #
def label_rows(
    df: pd.DataFrame,
    holiday_ranges: Dict[str, List],
    nations: Sequence[str],
    week_end_col: str,
    nation_threshold: float,
    rule: str,
) -> pd.DataFrame:
    """Return `df` with `n_nations_on_holiday` and boolean `is_holiday` columns."""
    # Map the lower-cased nation keys the user passed onto the holiday file's casing.
    ranges_lower = {k.lower(): v for k, v in holiday_ranges.items()}
    missing = [n for n in nations if n.lower() not in ranges_lower]
    if missing:
        raise ValueError(
            f"No holiday ranges for nation(s) {missing}. "
            f"The holiday file defines: {sorted(holiday_ranges)}."
        )
    days_by_nation = {n: holiday_days(ranges_lower[n.lower()]) for n in nations}

    end_dates = pd.to_datetime(df[week_end_col], format="%m/%d/%Y").dt.date

    n_on_holiday = []
    for end in end_dates:
        week = [end - timedelta(days=d) for d in range(DAYS_PER_WEEK)]
        count = 0
        for n in nations:
            hol_days = sum(1 for day in week if day in days_by_nation[n])
            if hol_days / DAYS_PER_WEEK >= nation_threshold:
                count += 1
        n_on_holiday.append(count)

    n_on_holiday = np.asarray(n_on_holiday, dtype=int)
    n_nations = len(nations)

    if rule == "any":
        is_holiday = n_on_holiday >= 1
    elif rule == "majority":
        is_holiday = n_on_holiday > n_nations / 2
    elif rule == "all":
        is_holiday = n_on_holiday == n_nations
    else:  # pragma: no cover - argparse restricts this
        raise ValueError(f"Unknown rule '{rule}'.")

    out = df.copy()
    out["n_nations_on_holiday"] = n_on_holiday
    out["is_holiday"] = is_holiday
    return out


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def allocate(n: int, fractions: Sequence[float]) -> List[int]:
    """Split `n` items across `fractions` by largest remainder; counts sum to `n`.

    Splits with a nonzero fraction are then guaranteed at least one item, provided
    `n` is at least the number of such splits; the item is taken from whichever split
    currently holds the most.
    """
    raw = [f * n for f in fractions]
    counts = [int(np.floor(r)) for r in raw]
    remainder = n - sum(counts)
    if remainder:
        # Largest fractional part first; ties broken by split order (train, dev, test).
        order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))
        for i in order[:remainder]:
            counts[i] += 1

    active = [i for i, f in enumerate(fractions) if f > 0]
    if n >= len(active):
        for i in active:
            if counts[i] == 0:
                donor = max(active, key=lambda j: counts[j])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[i] += 1
    return counts


def stratified_split(
    labels: np.ndarray, fractions: Sequence[float], rng: np.random.Generator
) -> np.ndarray:
    """Return an array of split names ('train'/'dev'/'test'), one per row."""
    assignment = np.empty(len(labels), dtype=object)
    n_active = sum(1 for f in fractions if f > 0)

    for value in np.unique(labels):
        idx = np.flatnonzero(labels == value)
        rng.shuffle(idx)
        counts = allocate(len(idx), fractions)
        if len(idx) < n_active:
            print(
                f"  WARNING: stratum is_holiday={bool(value)} has only {len(idx)} "
                f"row(s); it cannot appear in all {n_active} splits.",
                file=sys.stderr,
            )
        pos = 0
        for name, k in zip(SPLIT_NAMES, counts):
            assignment[idx[pos:pos + k]] = name
            pos += k
    return assignment


def tail_split(n_rows: int, n_dev: int, n_test: int) -> np.ndarray:
    """Contiguous holdout in file order: ... | dev block | test block (the last rows)."""
    if n_dev + n_test >= n_rows:
        raise ValueError(
            f"--holdout-tail {n_dev} {n_test} leaves no training rows out of {n_rows}."
        )
    assignment = np.array(["train"] * n_rows, dtype=object)
    if n_test:
        assignment[n_rows - n_test:] = "test"
    if n_dev:
        assignment[n_rows - n_test - n_dev: n_rows - n_test] = "dev"
    return assignment


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flu", type=Path,
                   default=Path("data/epidemic/uk_flu_per_100000.csv"))
    p.add_argument("--holidays", type=Path,
                   default=Path("data/great_brittain/school_holidays.csv"))
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: same directory as --flu")
    p.add_argument("--nations", nargs="+", default=list(DEFAULT_NATIONS),
                   help="nations whose calendars define the stratum label")
    p.add_argument("--week-end-col", type=str, default="week_end_date")
    p.add_argument("--nation-threshold", type=float, default=0.5,
                   help="fraction of a week's days inside a holiday for that nation "
                        "to count as on holiday (default 0.5)")
    p.add_argument("--rule", choices=("any", "majority", "all"), default="any",
                   help="how per-nation holiday flags combine into one label")
    p.add_argument("--split", nargs=3, type=float, metavar=("TRAIN", "DEV", "TEST"),
                   default=[0.70, 0.15, 0.15],
                   help="split proportions; must be non-negative and sum to 1. "
                        "Set DEV to 0 for a two-way split.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout-tail", nargs=2, type=int, metavar=("DEV", "TEST"),
                   default=None,
                   help="ignore the stratified shuffle; hold out the last TEST rows "
                        "as test and the DEV rows before them as dev "
                        "(forecasting-style split)")
    p.add_argument("--keep-labels", action="store_true",
                   help="also write is_holiday / n_nations_on_holiday into the split "
                        "CSVs (off by default so they stay drop-in replacements for "
                        "the original file)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    fractions = list(args.split)
    if any(f < 0 for f in fractions):
        raise ValueError(f"--split proportions must be non-negative; got {fractions}.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"--split proportions must sum to 1; got {sum(fractions)}.")
    if fractions[0] <= 0:
        raise ValueError("The train proportion must be greater than zero.")

    df = pd.read_csv(args.flu)
    if args.week_end_col not in df.columns:
        raise ValueError(
            f"'{args.week_end_col}' not in {args.flu}. Columns: {list(df.columns)}"
        )

    holiday_ranges = load_holiday_ranges(args.holidays)
    labelled = label_rows(
        df,
        holiday_ranges=holiday_ranges,
        nations=args.nations,
        week_end_col=args.week_end_col,
        nation_threshold=args.nation_threshold,
        rule=args.rule,
    )
    labels = labelled["is_holiday"].to_numpy()

    print(f"Loaded {len(df)} weeks from {args.flu}")
    print(f"Holiday calendar: {args.holidays} ({', '.join(sorted(holiday_ranges))})")
    print(f"Label rule: {args.rule} of {list(args.nations)} "
          f"at >={args.nation_threshold:.2f} of the week")
    print(f"  holiday weeks:   {int(labels.sum())}")
    print(f"  term-time weeks: {int((~labels).sum())}")

    if args.holdout_tail is not None:
        n_dev, n_test = args.holdout_tail
        print(f"Contiguous tail holdout: last {n_test} weeks -> test, "
              f"{n_dev} weeks before that -> dev")
        assignment = tail_split(len(labelled), n_dev, n_test)
    else:
        print(f"Stratified shuffle: train/dev/test = "
              f"{fractions[0]:.0%}/{fractions[1]:.0%}/{fractions[2]:.0%}, "
              f"seed {args.seed}")
        rng = np.random.default_rng(args.seed)
        assignment = stratified_split(labels, fractions, rng)

    labelled["split"] = assignment

    out_cols = list(df.columns)
    if args.keep_labels:
        out_cols += ["is_holiday", "n_nations_on_holiday"]

    out_dir = args.out_dir or args.flu.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.flu.stem

    _report(labels, assignment)
    print()
    for name in SPLIT_NAMES:
        mask = assignment == name
        if not mask.any():
            print(f"Skipped {name}: no rows allocated")
            continue
        path = out_dir / f"{stem}_{name}.csv"
        labelled.loc[mask, out_cols].to_csv(path, index=False)
        print(f"Wrote {int(mask.sum()):3d} rows -> {path}")

    labels_path = out_dir / f"{stem}_split_labels.csv"
    labelled.to_csv(labels_path, index=False)
    print(f"Wrote audit table   -> {labels_path}")


def _report(labels: np.ndarray, assignment: np.ndarray) -> None:
    print("\n           holiday  term  total   holiday share")
    rows = [(name, assignment == name) for name in SPLIT_NAMES]
    rows.append(("all", np.ones(len(labels), dtype=bool)))
    for name, mask in rows:
        h = int((labels & mask).sum())
        t = int((~labels & mask).sum())
        n = h + t
        share = f"{h / n:6.1%}" if n else "     -"
        print(f"  {name:<7} {h:5d} {t:5d}  {n:5d}   {share}")


if __name__ == "__main__":
    main()