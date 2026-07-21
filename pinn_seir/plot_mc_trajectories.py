"""Reload a checkpoint and plot the distribution of MC-dropout trajectories.

Draws N stochastic forward passes with dropout active (see
``PINNTrainer.sample_nation_incidence`` / ``sample_state_trajectories``) and plots
them as a fan chart (median + percentile bands) with an optional spaghetti overlay of
individual draws, alongside the observed weekly ILI points. Requires a checkpoint that
was trained with ``dropout_rate > 0`` (see ``fit_pinn.py --dropout-rate``); the rate is
read back from the checkpoint automatically.

Usage
-----
Weekly incidence fan chart (default):
    python -m pinn_seir.plot_mc_trajectories \
        --checkpoint outputs/seir_pinn/11426009/checkpoint.pt \
        --census     data/great_brittain/census.csv \
        --commute    data/great_brittain/commute.csv \
        --crosswalk  data/great_brittain/crosswalk.tsv \
        --contacts   data/contacts \
        --flu        data/epidemic/uk_flu_per_100000.csv \
        --n-samples  20 \
        --out        outputs/seir_pinn/11426009 --dates

Daily incidence instead of weekly:
    ... --daily --samples-per-day 2

Full SEIAR compartment trajectories (one figure per nation, all 5 compartments):
    ... --compartments
"""

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

_COMPARTMENT_NAMES = ["S", "E", "I (symptomatic)", "A (asymptomatic)", "R"]


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

    p.add_argument("--n-samples", type=int, default=200,
                   help="number of MC-dropout forward passes to draw")
    p.add_argument("--daily", action="store_true",
                   help="sample the daily incidence curve instead of weekly")
    p.add_argument("--samples-per-day", type=int, default=2,
                   help="network evaluations per day (only used with --daily)")
    p.add_argument("--compartments", action="store_true",
                   help="plot full S/E/I/A/R nation trajectories instead of incidence")
    p.add_argument("--n-spaghetti", type=int, default=20,
                   help="number of individual sampled trajectories to overlay as thin "
                        "lines (0 disables the spaghetti overlay)")
    p.add_argument("--percentiles", type=str, default="2.5,25,75,97.5",
                   help="comma-separated percentiles for the fan-chart bands, given as pairs from the outside in (e.g. 2.5,25,75,97.5 draws a 95%% band and a 50%% band)")
    p.add_argument("--dates", action="store_true", help="label x-axis with calendar dates")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the MC draws")
    return p.parse_args()


def _week_to_date(epidemic_start: str, week_float: np.ndarray) -> list:
    start = datetime.strptime(epidemic_start, "%Y-%m-%d")
    return [start + timedelta(days=7.0 * float(w)) for w in week_float]


def _plot_fan(
    ax, x, samples_r, obs_x=None, obs_y=None, percentile_pairs=None,
    n_spaghetti=0, ylabel="", title="", rng=None,
):
    """samples_r: (n_samples, n_x) array for one nation."""
    median = np.median(samples_r, axis=0)
    ax.plot(x, median, color="C0", lw=2, label="MC-dropout median", zorder=3)

    if percentile_pairs:
        n_bands = len(percentile_pairs)
        for i, (lo_q, hi_q) in enumerate(percentile_pairs):
            lo = np.percentile(samples_r, lo_q, axis=0)
            hi = np.percentile(samples_r, hi_q, axis=0)
            alpha = 0.15 + 0.15 * (i + 1) / n_bands
            ax.fill_between(
                x, lo, hi, color="C0", alpha=alpha, zorder=1,
                label=f"{hi_q - lo_q:.0f}% band" if hi_q - lo_q < 100 else None,
            )

    if n_spaghetti > 0:
        rng = rng or np.random.default_rng(0)
        n_samples = samples_r.shape[0]
        idx = rng.choice(n_samples, size=min(n_spaghetti, n_samples), replace=False)
        for j in idx:
            ax.plot(x, samples_r[j], color="C0", lw=0.5, alpha=0.35, zorder=2)

    if obs_x is not None:
        ax.scatter(obs_x, obs_y, s=18, color="k", label="observed", zorder=4)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    perc = [float(v) for v in args.percentiles.split(",")]
    if len(perc) % 2 != 0:
        raise ValueError("--percentiles must contain an even number of values (pairs).")
    perc = sorted(perc)
    n_bands = len(perc) // 2
    percentile_pairs = [(perc[i], perc[-(i + 1)]) for i in range(n_bands)]

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
          f"kappa={trainer.kappa.item():.4f} alpha={trainer.alpha.item():.4f} "
          f"dropout_rate={trainer.net.dropout_rate}")

    if trainer.net.dropout_rate <= 0.0:
        print(
            "WARNING: this checkpoint has dropout_rate=0, so all MC samples will be "
            "identical (a single deterministic trajectory). Retrain with "
            "--dropout-rate > 0 to get a real distribution."
        )

    torch_rng_seed = args.seed
    import torch
    torch.manual_seed(torch_rng_seed)
    rng = np.random.default_rng(torch_rng_seed)

    with torch.no_grad():
        if args.compartments:
            t_days, traj = trainer.sample_state_trajectories(
                n_samples=args.n_samples,
                samples_per_day=args.samples_per_day,
            )  # traj: (n_samples, R, n_days, 5)
            x = t_days
            if args.dates:
                x = _week_to_date(mcfg.epidemic_start, t_days)

            for r, nation in enumerate(data.nation_names):
                fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharex=True)
                for c in range(5):
                    _plot_fan(
                        axes[c], x, traj[:, r, :, c],
                        percentile_pairs=percentile_pairs,
                        n_spaghetti=args.n_spaghetti,
                        ylabel="count", title=_COMPARTMENT_NAMES[c], rng=rng,
                    )
                fig.suptitle(f"MC-dropout SEIAR trajectories — {nation}")
                fig.tight_layout()
                out_path = args.out / f"mc_compartments_{nation.lower()}.png"
                fig.savefig(out_path, dpi=130)
                print(f"Saved -> {out_path}")

        else:
            # Incidence (weekly or daily), the same quantity fit against the ILI data.
            if args.daily:
                t_x, samples = None, None
                samples = trainer.sample_nation_incidence(
                    n_samples=args.n_samples, daily=True, samples_per_day=args.samples_per_day,
                )  # (n_samples, R, n_days)
                t_x, _ = trainer.predict_nation_incidence_daily(samples_per_day=args.samples_per_day)
                x = t_x
                xlabel = "week"
            else:
                samples = trainer.sample_nation_incidence(n_samples=args.n_samples)  # (n_samples, R, n_weeks)
                x = np.arange(samples.shape[-1])
                xlabel = "week"

            if args.dates:
                x = _week_to_date(mcfg.epidemic_start, x)
                xlabel = "date"

            obs = data.y_obs
            obs_x = data.obs_week_index.astype(float)
            if args.dates:
                obs_x = _week_to_date(mcfg.epidemic_start, obs_x)

            n_nations = data.n_nations
            fig, axes = plt.subplots(1, n_nations, figsize=(5.5 * n_nations, 4.5), sharex=True)
            if n_nations == 1:
                axes = [axes]
            for r, ax in enumerate(axes):
                _plot_fan(
                    ax, x, samples[:, r, :],
                    obs_x=obs_x, obs_y=obs[r, : len(data.obs_week_index)],
                    percentile_pairs=percentile_pairs,
                    n_spaghetti=args.n_spaghetti,
                    ylabel="ILI per 100k" + (" / day" if args.daily else " / week"),
                    title=data.nation_names[r], rng=rng,
                )
                ax.set_xlabel(xlabel)
            fig.suptitle(
                f"MC-dropout {'daily' if args.daily else 'weekly'} incidence "
                f"({args.n_samples} samples, dropout_rate={trainer.net.dropout_rate})"
            )
            fig.tight_layout()
            suffix = "daily" if args.daily else "weekly"
            out_path = args.out / f"mc_incidence_{suffix}.png"
            fig.savefig(out_path, dpi=130)
            print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()