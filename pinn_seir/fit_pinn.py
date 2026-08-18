"""Fit the policy-conditioned SEIR PINN to the 2009 H1N1 ILI data and export results.

Example
-------
    python -m pinn_seir.fit_pinn \
        --census        data/great_brittain/census.csv \
        --commute       data/great_brittain/commute.csv \
        --crosswalk     data/great_brittain/crosswalk.tsv \
        --contacts      data/contacts \
        --flu           data/epidemic/splits/uk_flu_per_100000_train.csv \
        --flu-dev       data/epidemic/splits/uk_flu_per_100000_dev.csv \
        --holidays      data/great_brittain/school_holidays.csv \
        --adam-iters    5000 \
        --out           /tmp/seir_pinn

Pass ``--no-historical-holidays`` to ablate the fixed school calendar entirely (every
day term-time). Note this is NOT the same as `plot_fit`'s ``--no-holidays``, which
only suppresses the shaded bands on the plot.

For Monte-Carlo dropout (epistemic-uncertainty sampling later), train with a nonzero
dropout rate, e.g. ``--dropout-rate 0.05``. The saved checkpoint records the rate so
`plot_fit`/reload rebuild an identical dropout network; MC samples are then drawn with
``PINNTrainer.sample_nation_incidence`` / ``sample_state_trajectories``.

Outputs written to --out:
    checkpoint.pt   trained network + fitted parameters
    fit.png         predicted vs observed weekly ILI per 100k, per nation
    params.json     final fitted parameter values and loss breakdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import ModelConfig, TrainConfig
from .data import load_epi_data
from .model import PINNTrainer

from .plot_utils import mark_switches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    # data paths
    p.add_argument("--census", type=Path, default=ModelConfig.census_path)
    p.add_argument("--commute", type=Path, default=ModelConfig.commute_path)
    p.add_argument("--crosswalk", type=Path, default=ModelConfig.crosswalk_path)
    p.add_argument("--contacts", type=Path, default=ModelConfig.contacts_dir)
    p.add_argument("--flu", type=Path, default=ModelConfig.flu_path)
    p.add_argument("--holidays", type=Path, default=ModelConfig.holidays_path,
                   help="per-nation school-holiday CSV (see pinn_seir/holidays.py)")
    p.add_argument("--no-historical-holidays", action="store_true",
                   help="ablation: ignore the school calendar, treat every day as "
                        "term-time (POLICY closures are unaffected)")
    # structure
    p.add_argument("--n-weeks", type=int, default=ModelConfig.n_weeks)
    p.add_argument("--seed-district", type=str, default=ModelConfig.seed_district)
    p.add_argument("--seed-exposed", type=float, default=ModelConfig.seed_exposed_count)
    # parameter freedom
    p.add_argument("--train-kappa", action="store_true", help="free the coupling scale kappa")
    p.add_argument("--train-alpha", action="store_true", help="free ascertainment alpha")
    p.add_argument("--r0-init", type=float, default=ModelConfig.r0_init)
    # optimisation
    p.add_argument("--adam-iters", type=int, default=TrainConfig.adam_iters)
    p.add_argument("--adam-lr", type=float, default=TrainConfig.adam_lr)
    p.add_argument("--lr-decay-step", type=int, default=TrainConfig.lr_decay_step,
                   help="StepLR interval in Adam iterations")
    p.add_argument("--lr-decay-gamma", type=float, default=TrainConfig.lr_decay_gamma,
                   help="StepLR multiplicative factor at each decay step")
    p.add_argument("--lbfgs-iters", type=int, default=TrainConfig.lbfgs_iters)
    p.add_argument("--hidden-layers", type=int, default=TrainConfig.hidden_layers)
    p.add_argument("--hidden-width", type=int, default=TrainConfig.hidden_width)
    p.add_argument("--dropout-rate", type=float, default=TrainConfig.dropout_rate,
                   help="trunk dropout probability; >0 enables MC-dropout sampling")
    p.add_argument("--n-schedules", type=int, default=TrainConfig.n_schedules)
    p.add_argument("--n-collocation", type=int, default=TrainConfig.n_collocation)
    p.add_argument("--device", type=str, default=TrainConfig.device)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    # output
    p.add_argument("--out", type=Path, default=Path("/tmp/seir_pinn"))
    p.add_argument("--epidemic-start", default=ModelConfig.epidemic_start)
    p.add_argument("--regime-switch-date", default=ModelConfig.regime_switch_date,
                help="ISO date of the containment->treatment switch; "
                        "pass --no-regime-switch for a single-regime fit")
    p.add_argument("--no-regime-switch", dest="regime_switch_date",
                action="store_const", const=None)
    p.add_argument("--r0-post-init",    type=float, default=ModelConfig.r0_post_init)
    p.add_argument("--gamma-init",      type=float, default=ModelConfig.gamma)
    p.add_argument("--gamma-post-init", type=float, default=ModelConfig.gamma_post_init)
    p.add_argument("--train-r0-post",    action="store_true")
    p.add_argument("--train-gamma",      action="store_true")
    p.add_argument("--train-gamma-post", action="store_true")
    p.add_argument("--flu-dev", type=Path, default=None,
                   help="held-out observation CSV for dev evaluation; never enters "
                        "any loss term. Same schema as --flu.")
    p.add_argument("--data-scale", type=float, default=None,
                   help="pin the data-loss normaliser mean(y_obs**2). Leave unset "
                        "within a sweep; pass the sweep's value when refitting on a "
                        "different observation file (Phase 5).")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="stop after N dev evaluations without improvement (0 = off, "
                        "recommended: save-best is on regardless)")
    # ---- Phase 3: shared unsupervised pretraining ----
    p.add_argument("--pretrain", action="store_true",
                   help="Stage A: train with w_data = 0 (no ILI values are read) and "
                        "skip dev evaluation. Produces a reusable warm start.")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Stage B: warm-start from a checkpoint. Architecture must "
                        "match exactly.")
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
        flu_dev_path=None if args.pretrain else args.flu_dev,
        holidays_path=args.holidays,
        no_holidays=args.no_historical_holidays,
        n_weeks=args.n_weeks,
        seed_district=args.seed_district,
        seed_exposed_count=args.seed_exposed,
        r0_init=args.r0_init,
        train_kappa=args.train_kappa,
        train_alpha=args.train_alpha,
        # ---- regime switch ----
        epidemic_start=args.epidemic_start,
        regime_switch_date=args.regime_switch_date,
        r0_post_init=args.r0_post_init,
        gamma=args.gamma_init,
        gamma_post_init=args.gamma_post_init,
        train_r0_post=args.train_r0_post,
        train_gamma=args.train_gamma,
        train_gamma_post=args.train_gamma_post,
    )
    tcfg = TrainConfig(
        adam_iters=args.adam_iters,
        adam_lr=args.adam_lr,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma,
        lbfgs_iters=args.lbfgs_iters,
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
        dropout_rate=args.dropout_rate,
        data_scale=args.data_scale,
        early_stop_patience=args.early_stop_patience,
        n_schedules=args.n_schedules,
        n_collocation=args.n_collocation,
        device=args.device,
        seed=args.seed,
    )
    if args.pretrain:
        tcfg.w_data = 0.0
        if args.flu_dev is not None:
            print("  --pretrain: ignoring --flu-dev (no dev evaluation in Stage A)")
    print("Loading data ...")
    data = load_epi_data(mcfg)
    print(
        f"  {data.n_patches} districts -> {data.n_nations} nations "
        f"({', '.join(data.nation_names)}); seed = "
        f"{data.district_names[data.seed_district_index]}"
    )
    if mcfg.no_holidays:
        print("  ABLATION: historical school calendar disabled (all days term-time)")
    else:
        print(f"  school calendar: {mcfg.holidays_path} "
              f"({', '.join(sorted(mcfg.holiday_ranges_by_nation))})")

    print("Building and training PINN ...")
    if tcfg.dropout_rate > 0.0:
        print(f"  dropout_rate = {tcfg.dropout_rate} (MC-dropout sampling enabled)")
    trainer = PINNTrainer(data, mcfg, tcfg)
    print(f"  data_scale = {trainer._data_scale:.6g}"
          f"{' (pinned)' if tcfg.data_scale is not None else ' (from --flu)'}")
    if args.pretrain:
        print("  STAGE A pretraining: w_data = 0, no observations enter the loss")
    if trainer.has_dev:
        print(f"  dev: {data.y_obs_dev.shape[1]} weeks from {args.flu_dev}")

    if args.init_from is not None:
        trainer.load_checkpoint(str(args.init_from), strict_arch=True)
        trainer.net.train()
        print(f"  warm start from {args.init_from}")

    best_path = args.out / "checkpoint_best.pt" if trainer.has_dev else None
    logs = trainer.train(best_checkpoint_path=str(best_path) if best_path else None)

    ckpt = args.out / "checkpoint.pt"
    trainer.save(str(ckpt))
    print(f"Saved checkpoint -> {ckpt}")

    # Record the calendar setting alongside the fitted parameters: a no-holidays fit
    # is not comparable to a normal one, and the checkpoint alone does not say which
    # it was.
    logs = dict(logs)
    logs["no_holidays"] = bool(mcfg.no_holidays)
    logs["holidays_path"] = None if mcfg.no_holidays else str(mcfg.holidays_path)
    logs["data_scale"] = trainer._data_scale
    logs["flu_path"] = str(mcfg.flu_path)
    logs["flu_dev_path"] = None if mcfg.flu_dev_path is None else str(mcfg.flu_dev_path)
    logs["w_data"] = tcfg.w_data
    logs["w_junction"] = tcfg.w_junction
    logs["w_ic"] = tcfg.w_ic
    logs["dropout_rate"] = tcfg.dropout_rate
    logs["seed"] = tcfg.seed
    logs["pretrain"] = bool(args.pretrain)
    logs["init_from"] = None if args.init_from is None else str(args.init_from)

    if trainer.has_dev:
        print(f"Best dev: {trainer.best_dev:.4e} at iteration {trainer.best_iter}")
        for k in ("dev_rmse", "dev_rmse_holiday", "dev_rmse_term"):
            print(f"  {k} = {trainer.best_dev_logs.get(k, float('nan')):.4f}")

    with open(args.out / "params.json", "w") as fh:
        json.dump(logs, fh, indent=2)
    print("Fitted parameters:")
    for key in ("R0", "R0_post", "gamma", "gamma_post", "mu", "kappa", "alpha"):
        print(f"  {key} = {logs[key]:.4f}")

    _export_plot(trainer, data, args.out / "fit.png")
    print(f"Saved fit plot -> {args.out / 'fit.png'}")


def _export_plot(trainer: PINNTrainer, data, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    pred = trainer.predict_nation_incidence()          # (R, n_weeks)
    obs = data.y_obs                                    # (R, T)
    weeks = np.arange(pred.shape[1])

    # If trained with dropout, overlay an MC-dropout uncertainty band (mean +/- 2 std).
    band = None
    if getattr(trainer.net, "dropout_rate", 0.0) > 0.0:
        samples = trainer.sample_nation_incidence(n_samples=100)  # (S, R, n_weeks)
        band = (samples.mean(axis=0), samples.std(axis=0))

    fig, axes = plt.subplots(
        1, data.n_nations, figsize=(5 * data.n_nations, 4), sharex=True
    )
    if data.n_nations == 1:
        axes = [axes]
    for r, ax in enumerate(axes):
        ax.plot(weeks, pred[r], label="PINN", lw=2)
        if band is not None:
            mean_r, std_r = band[0][r], band[1][r]
            ax.fill_between(
                weeks, mean_r - 2 * std_r, mean_r + 2 * std_r,
                alpha=0.25, color="C0", label="MC-dropout ±2σ",
            )
        ax.scatter(
            data.obs_week_index, obs[r, : len(data.obs_week_index)],
            s=18, color="k", label="observed", zorder=3,
        )
        mark_switches(ax, trainer.mcfg, use_dates=False)
        ax.set_title(data.nation_names[r])
        ax.set_xlabel("week")
        ax.set_ylabel("ILI per 100k / week")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()