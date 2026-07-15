"""Fit the policy-conditioned SEIR PINN to the 2009 H1N1 ILI data and export results.

Example
-------
    python -m pinn_seir.fit_pinn \
        --census        data/great_brittain/census.csv \
        --commute       data/great_brittain/commute.csv \
        --crosswalk     data/great_brittain/crosswalk.tsv \
        --contacts      data/contacts \
        --flu           data/epidemic/uk_flu_per_100000.csv \
        --adam-iters    40000 \
        --out           /tmp/seir_pinn

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    # data paths
    p.add_argument("--census", type=Path, default=ModelConfig.census_path)
    p.add_argument("--commute", type=Path, default=ModelConfig.commute_path)
    p.add_argument("--crosswalk", type=Path, default=ModelConfig.crosswalk_path)
    p.add_argument("--contacts", type=Path, default=ModelConfig.contacts_dir)
    p.add_argument("--flu", type=Path, default=ModelConfig.flu_path)
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
    p.add_argument("--lbfgs-iters", type=int, default=TrainConfig.lbfgs_iters)
    p.add_argument("--hidden-layers", type=int, default=TrainConfig.hidden_layers)
    p.add_argument("--hidden-width", type=int, default=TrainConfig.hidden_width)
    p.add_argument("--n-schedules", type=int, default=TrainConfig.n_schedules)
    p.add_argument("--n-collocation", type=int, default=TrainConfig.n_collocation)
    p.add_argument("--device", type=str, default=TrainConfig.device)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    # output
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
        n_schedules=args.n_schedules,
        n_collocation=args.n_collocation,
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
    trainer = PINNTrainer(data, mcfg, tcfg)
    logs = trainer.train()

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

    pred = trainer.predict_nation_incidence()          # (R, n_weeks)
    obs = data.y_obs                                    # (R, T)
    weeks = np.arange(pred.shape[1])

    fig, axes = plt.subplots(
        1, data.n_nations, figsize=(5 * data.n_nations, 4), sharex=True
    )
    if data.n_nations == 1:
        axes = [axes]
    for r, ax in enumerate(axes):
        ax.plot(weeks, pred[r], label="PINN", lw=2)
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