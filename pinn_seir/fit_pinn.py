from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import ModelConfig, TrainConfig
from .data import load_epi_data
from .model import PINNTrainer

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    p.add_argument("--census", type=Path, default=ModelConfig.census_path)
    p.add_argument("--commute", type=Path, default=ModelConfig.commute_path)
    p.add_argument("--crosswalk", type=Path, default=ModelConfig.crosswalk_path)
    p.add_argument("--contacts", type=Path, default=ModelConfig.contacts_dir)
    p.add_argument("--flu", type=Path, default=ModelConfig.flu_path)
    p.add_argument("--n-weeks", type=int, default=ModelConfig.n_weeks)
    p.add_argument("--seed-district", type=str, default=ModelConfig.seed_district)
    p.add_argument("--seed-exposed", type=float, default=ModelConfig.seed_exposed_count)
    p.add_argument("--train-kappa", action="store_true", help="free the coupling scale kappa")
    p.add_argument("--train-alpha", action="store_true", help="free ascertainment alpha")
    p.add_argument("--r0-init", type=float, default=ModelConfig.r0_init)
    p.add_argument("--adam-iters", type=int, default=TrainConfig.adam_iters)
    p.add_argument("--adam-lr", type=float, default=TrainConfig.adam_lr)
    p.add_argument("--lbfgs-iters", type=int, default=TrainConfig.lbfgs_iters)
    p.add_argument("--hidden-layers", type=int, default=TrainConfig.hidden_layers)
    p.add_argument("--hidden-width", type=int, default=TrainConfig.hidden_width)
    p.add_argument("--dropout-rate", type=float, default=TrainConfig.dropout_rate,
                   help="trunk dropout probability; >0 enables MC-dropout sampling")
    p.add_argument("--n-schedules", type=int, default=TrainConfig.n_schedules)
    p.add_argument("--n-collocation", type=int, default=TrainConfig.n_collocation)
    p.add_argument("--w-phys", type=float, default=TrainConfig.w_phys,
                   help="physics-loss weight; raise this if the fitted surrogate is "
                        "insensitive to closure schedules it never saw in the data loss")
    p.add_argument("--phys-activity-floor", type=float,
                   default=TrainConfig.phys_activity_floor,
                   help="floor (as a fraction of peak I+A activity) for the physics-"
                        "residual importance weighting; raise if the network freezes "
                        "state within a week and lets the week-embedding memorise "
                        "weekly incidence instead of evolving genuine SEIR dynamics")
    p.add_argument("--w-monotonic", type=float, default=TrainConfig.w_monotonic,
                   help="weight on the dS/dtau<=0, dR/dtau>=0 hinge penalty; raise "
                        "if R decreases or S increases within a week (unphysical)")
    p.add_argument("--w-cumulative", type=float, default=TrainConfig.w_cumulative,
                   help="weight anchoring cumulative R at the last observed week to "
                        "the data-implied total attack rate; raise if the rollout "
                        "attack rate stays ~0 despite healthy per-week diagnostics")
    p.add_argument("--checkpoint-every", type=int, default=None,
                   help="also save checkpoint_iterNNNNNN.pt every N iterations, so a "
                        "long run can be inspected/used before it fully completes")
    p.add_argument("--device", type=str, default=TrainConfig.device)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--out", type=Path, default=Path("/tmp/seir_pinn"))
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
        seed_exposed_count=args.seed_exposed,
        r0_init=args.r0_init,
        train_kappa=args.train_kappa,
        train_alpha=args.train_alpha,
    )
    tcfg = TrainConfig(
        adam_iters=args.adam_iters,
        adam_lr=args.adam_lr,
        lbfgs_iters=args.lbfgs_iters,
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
        dropout_rate=args.dropout_rate,
        n_schedules=args.n_schedules,
        n_collocation=args.n_collocation,
        w_phys=args.w_phys,
        phys_activity_floor=args.phys_activity_floor,
        w_monotonic=args.w_monotonic,
        w_cumulative=args.w_cumulative,
        device=args.device,
        seed=args.seed,
    )

    print("Loading data ...")
    data = load_epi_data(mcfg)
    print(
        f"  {data.n_patches} districts -> {data.n_nations} nations "
        f"({', '.join(data.nation_names)}); seed = "
        f"{data.district_names[data.seed_district_index]}"
    )

    print("Building and training PINN ...")
    if tcfg.dropout_rate > 0.0:
        print(f"  dropout_rate = {tcfg.dropout_rate} (MC-dropout sampling enabled)")
    trainer = PINNTrainer(data, mcfg, tcfg)
    logs = trainer.train(checkpoint_dir=str(args.out), checkpoint_every=args.checkpoint_every)

    ckpt = args.out / "checkpoint.pt"
    trainer.save(str(ckpt))
    print(f"Saved checkpoint -> {ckpt}")

    with open(args.out / "params.json", "w") as fh:
        json.dump(logs, fh, indent=2)
    print("Fitted parameters:")
    for key in ("R0", "mu", "kappa", "alpha"):
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

    pred = trainer.predict_nation_incidence()
    obs = data.y_obs
    weeks = np.arange(pred.shape[1])

    band = None
    if getattr(trainer.net, "dropout_rate", 0.0) > 0.0:
        samples = trainer.sample_nation_incidence(n_samples=100)
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
        ax.set_title(data.nation_names[r])
        ax.set_xlabel("week")
        ax.set_ylabel("ILI per 100k / week")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)

if __name__ == "__main__":
    main()
