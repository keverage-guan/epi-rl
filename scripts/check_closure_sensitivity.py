import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pinn_seir.config import ModelConfig, TrainConfig
from pinn_seir.data import load_epi_data
from pinn_seir.model import PINNTrainer

FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--flu", type=Path, default=Path("data/epidemic/uk_flu_per_100000.csv"))
    p.add_argument("--dropout-rate", type=float, default=0.1)
    p.add_argument("--week", type=int, default=20)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="force 'cpu' to avoid contending with a training run still "
                        "using the GPU; defaults to cuda if available")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    mcfg = ModelConfig(flu_path=args.flu)
    tcfg = TrainConfig(dropout_rate=args.dropout_rate, device=device)
    data = load_epi_data(mcfg)
    trainer = PINNTrainer(data, mcfg, tcfg)
    trainer.load_checkpoint(str(args.checkpoint))
    net = trainer.net
    net.eval()
    print(f"Loaded {args.checkpoint}")
    print(f"  R0={trainer.r0.item():.4f} mu={trainer.mu.item():.4f} "
          f"kappa={trainer.kappa.item():.4f} alpha={trainer.alpha.item():.4f}")

    P = data.n_patches
    week_t = torch.tensor([args.week], dtype=torch.int64, device=device)
    tau = torch.full((1, 1), 0.5, device=device)

    results = []
    for f in FRACTIONS:
        n_closed = int(round(f * P))
        closure = torch.ones((1, P), device=device)
        closure[0, :n_closed] = 0.0
        with torch.no_grad():
            s = net(tau, week_t, closure)
        S = s[..., 0].sum().item()
        I = s[..., 2].sum().item()
        results.append({"frac_closed": f, "S": S, "I": I})
        print(f"  frac_closed={f:.2f} (n={n_closed:3d}/{P})  S={S:14,.0f}  I={I:12,.0f}")

    S_vals = np.array([r["S"] for r in results])
    S_range_0_90 = S_vals[:-1].max() - S_vals[:-1].min()
    S_jump_at_100 = abs(S_vals[-1] - S_vals[-2])
    S_total_span = S_vals.max() - S_vals.min()
    frac_flat_variation = S_range_0_90 / max(S_total_span, 1e-9)

    print("\n=== SUMMARY ===")
    print(f"  S range across 0-90% closed : {S_range_0_90:,.0f}")
    print(f"  |S(100%) - S(90%)| jump     : {S_jump_at_100:,.0f}")
    print(f"  total S span (0% to 100%)   : {S_total_span:,.0f}")
    if S_total_span < 1000:
        verdict = "FAIL -- essentially no response to closures at all (flat everywhere)."
    elif frac_flat_variation < 0.05:
        verdict = ("FAIL -- still a step function: flat across 0-90%, all the movement "
                   "is a discontinuous jump only at/near 100% closed.")
    else:
        verdict = "PASS-ish -- response varies smoothly across the closure range."
    print(f"  verdict: {verdict}")

    if args.out:
        json.dump({"results": results, "R0": trainer.r0.item(), "mu": trainer.mu.item(),
                   "verdict": verdict}, open(args.out, "w"), indent=2)
        print(f"\nSaved -> {args.out}")

if __name__ == "__main__":
    main()
