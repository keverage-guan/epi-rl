from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ModelConfig, TrainConfig
from .data import load_epi_data
from .model import PINNTrainer
from .schedules import holiday_week_spans

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--census", type=Path, default=ModelConfig.census_path)
    p.add_argument("--commute", type=Path, default=ModelConfig.commute_path)
    p.add_argument("--crosswalk", type=Path, default=ModelConfig.crosswalk_path)
    p.add_argument("--contacts", type=Path, default=ModelConfig.contacts_dir)
    p.add_argument("--flu", type=Path, default=ModelConfig.flu_path)
    p.add_argument("--n-weeks", type=int, default=ModelConfig.n_weeks)
    p.add_argument("--seed-district", type=str, default=ModelConfig.seed_district)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=Path, default=Path("outputs/seir_pinn"))
    p.add_argument("--dates", action="store_true", help="label x-axis with calendar dates")
    p.add_argument("--no-holidays", action="store_true", help="do not shade holidays")
    p.add_argument("--samples-per-day", type=int, default=2,
                   help="network evaluations per day for the daily prediction curve")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mcfg = ModelConfig(
        census_path=args.census,
        commute_path=args.commute,
        crosswalk_path=args.crosswalk,
        contacts_dir=args.contacts,
        flu_path=args.flu,
        n_weeks=args.n_weeks,
        seed_district=args.seed_district,
    )
    tcfg = TrainConfig(device=args.device)

    print("Rebuilding model from data ...")
    data = load_epi_data(mcfg)
    trainer = PINNTrainer(data, mcfg, tcfg)
    trainer.load_checkpoint(str(args.checkpoint))
    print(f"  loaded checkpoint: R0={trainer.r0.item():.4f} mu={trainer.mu.item():.4f} "
          f"kappa={trainer.kappa.item():.4f} alpha={trainer.alpha.item():.4f}")

    t_days, pred_daily = trainer.predict_nation_incidence_daily(
        samples_per_day=args.samples_per_day
    )
    obs = data.y_obs
    obs_weeks = data.obs_week_index

    _plot(data, t_days, pred_daily, obs, obs_weeks, mcfg, args)

def _week_to_date(week_idx, epidemic_start: str):
    start = datetime.strptime(epidemic_start, "%Y-%m-%d").date()
    return [start + timedelta(days=7 * float(k)) for k in week_idx]

def _shade_holidays(ax, mcfg, use_dates: bool, nation: str) -> None:
    ranges = mcfg.holiday_ranges_by_nation.get(nation, [])
    if not ranges:
        return
    spans = holiday_week_spans(mcfg.epidemic_start, ranges)
    start = datetime.strptime(mcfg.epidemic_start, "%Y-%m-%d").date()
    for i, (w0, w1) in enumerate(spans):
        if use_dates:
            x0 = start + timedelta(days=7 * w0)
            x1 = start + timedelta(days=7 * w1)
        else:
            x0, x1 = w0, w1
        ax.axvspan(
            x0, x1, color="0.75", alpha=0.35, lw=0, zorder=0,
            label="school holiday" if i == 0 else None,
        )

def _plot(data, t_days, pred_daily, obs, obs_weeks, mcfg, args) -> None:
    R = data.n_nations
    dpw = float(mcfg.days_per_week)

    pred_weekly_equiv = pred_daily * dpw

    if args.dates:
        x_pred = _week_to_date(t_days, mcfg.epidemic_start)
        x_obs = _week_to_date(obs_weeks, mcfg.epidemic_start)
        xlabel = "date"
    else:
        x_pred = t_days
        x_obs = obs_weeks
        xlabel = "model week"

    fig, axes = plt.subplots(1, R, figsize=(6 * R, 4.5), sharex=True)
    if R == 1:
        axes = [axes]

    for r, ax in enumerate(axes):
        if not args.no_holidays:
            _shade_holidays(ax, mcfg, args.dates, data.nation_names[r])
        ax.plot(x_pred, pred_weekly_equiv[r], lw=1.5, color="C0",
                label="PINN (daily)")
        ax.scatter(x_obs, obs[r], s=22, color="k", zorder=3, label="observed ILI (weekly)")
        ax.set_title(data.nation_names[r])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ILI per 100k / week")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if args.dates:
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(45)
                lbl.set_ha("right")

    fig.suptitle("PINN fit vs. observed 2009 H1N1 ILI", y=1.02)
    fig.tight_layout()
    out = args.out / "pinn_vs_data.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved prediction plot -> {out}")

if __name__ == "__main__":
    main()
