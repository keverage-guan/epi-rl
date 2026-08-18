"""Score a trained checkpoint on a held-out split. Run ONCE, on test, at the end.

Rebuilds the model exactly as plot_fit does -- the calendar and regime switch are
fixed physics inputs, not carried by the weights, so they must match how the
checkpoint was trained (params.json records both).

Usage
-----
    python -m pinn_seir.eval_pinn \
        --checkpoint outputs/seir_pinn/11911711/checkpoint_best.pt \
        --flu        data/epidemic/splits/uk_flu_per_100000_train.csv \
        --flu-eval   data/epidemic/splits/uk_flu_per_100000_test.csv \
        --data-scale 1234.5 \
        --out        outputs/seir_pinn/11911711
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ModelConfig, TrainConfig
from .data import load_epi_data, load_flu_series, assert_disjoint_weeks
from .model import PINNTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--census", type=Path, default=ModelConfig.census_path)
    p.add_argument("--commute", type=Path, default=ModelConfig.commute_path)
    p.add_argument("--crosswalk", type=Path, default=ModelConfig.crosswalk_path)
    p.add_argument("--contacts", type=Path, default=ModelConfig.contacts_dir)
    p.add_argument("--flu", type=Path, required=True,
                   help="the TRAIN split the checkpoint was fit on")
    p.add_argument("--flu-eval", type=Path, required=True,
                   help="the split to score (normally test)")
    p.add_argument("--split-name", default="test")
    p.add_argument("--holidays", type=Path, default=ModelConfig.holidays_path)
    p.add_argument("--no-historical-holidays", action="store_true")
    p.add_argument("--epidemic-start", default=ModelConfig.epidemic_start)
    p.add_argument("--regime-switch-date", default=ModelConfig.regime_switch_date)
    p.add_argument("--no-regime-switch", dest="regime_switch_date",
                   action="store_const", const=None)
    p.add_argument("--n-weeks", type=int, default=ModelConfig.n_weeks)
    p.add_argument("--seed-district", type=str, default=ModelConfig.seed_district)
    p.add_argument("--data-scale", type=float, default=None,
                   help="pass the value from params.json so <split>_data is "
                        "comparable with the training curve")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mcfg = ModelConfig(
        census_path=args.census, commute_path=args.commute,
        crosswalk_path=args.crosswalk, contacts_dir=args.contacts,
        flu_path=args.flu, holidays_path=args.holidays,
        no_holidays=args.no_historical_holidays,
        n_weeks=args.n_weeks, seed_district=args.seed_district,
        epidemic_start=args.epidemic_start,
        regime_switch_date=args.regime_switch_date,
    )
    tcfg = TrainConfig(device=args.device, data_scale=args.data_scale)

    data = load_epi_data(mcfg)
    trainer = PINNTrainer(data, mcfg, tcfg)
    trainer.load_checkpoint(str(args.checkpoint))

    y_eval, weeks_eval = load_flu_series(mcfg, data.nation_names, path=args.flu_eval)
    assert_disjoint_weeks({"train": data.obs_week_index, args.split_name: weeks_eval})

    metrics = trainer.evaluate_on(
        torch.as_tensor(y_eval, dtype=trainer.dtype, device=trainer.device),
        torch.as_tensor(weeks_eval, dtype=torch.int64, device=trainer.device),
        prefix=args.split_name,
    )
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["flu_eval_path"] = str(args.flu_eval)
    metrics["data_scale"] = trainer._data_scale
    metrics["n_obs"] = int(len(weeks_eval))

    for k, v in metrics.items():
        print(f"  {k} = {v}")

    out_dir = args.out or args.checkpoint.parent
    out_path = out_dir / f"{args.split_name}_metrics.json"
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()