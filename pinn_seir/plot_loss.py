"""Parse a training .out log and plot the loss curves.

The trainer emits periodic lines of the form:

    [   500] loss=1.4667e-01 phys=3.778e-03 junc=5.518e-03 data=1.349e-01 ic=2.445e-04 | R0=1.743 mu=0.483 kappa=1.000 alpha=0.250 (229s)

This script extracts iteration, the four loss components, the total, and the fitted
parameters, then plots (a) the loss components on a log scale and (b) the parameter
trajectories.

Usage
-----
    python -m pinn_seir.plot_loss --log logs/seir_pinn_11426009.out --out outputs/seir_pinn/11426009
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# One regex for the whole logged line; all fields optional-safe via named groups.
_LINE = re.compile(
    r"\[\s*(?P<it>\d+)\]\s*"
    r"loss=(?P<loss>[-\d.eE+]+)\s+"
    r"phys=(?P<phys>[-\d.eE+]+)\s+"
    r"junc=(?P<junc>[-\d.eE+]+)\s+"
    r"data=(?P<data>[-\d.eE+]+)\s+"
    r"ic=(?P<ic>[-\d.eE+]+)\s*\|\s*"
    r"R0=(?P<R0>[-\d.eE+]+)\s+"
    r"mu=(?P<mu>[-\d.eE+]+)\s+"
    r"kappa=(?P<kappa>[-\d.eE+]+)\s+"
    r"alpha=(?P<alpha>[-\d.eE+]+)"
)

_FIELDS = ["it", "loss", "phys", "junc", "data", "ic", "R0", "mu", "kappa", "alpha"]


def parse_log(path: Path) -> dict:
    """Return a dict of field -> np.ndarray parsed from the log file."""
    rows = {f: [] for f in _FIELDS}
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            m = _LINE.search(line)
            if not m:
                continue
            for f in _FIELDS:
                rows[f].append(float(m.group(f)))
    if not rows["it"]:
        raise ValueError(
            f"No parseable log lines found in {path}. Expected lines like "
            "'[   500] loss=... phys=... junc=... data=... ic=... | R0=... mu=...'."
        )
    return {f: np.asarray(v) for f, v in rows.items()}


def plot_losses(data: dict, out_path: Path) -> None:
    it = data["it"]
    fig, (ax_loss, ax_par) = plt.subplots(1, 2, figsize=(14, 5))

    for key, label in [
        ("loss", "total"),
        ("phys", "physics"),
        ("junc", "junction"),
        ("data", "data"),
        ("ic", "IC"),
    ]:
        ax_loss.plot(it, data[key], label=label, lw=1.5)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Adam iteration")
    ax_loss.set_ylabel("loss (log scale)")
    ax_loss.set_title("Loss components")
    ax_loss.legend()
    ax_loss.grid(True, which="both", alpha=0.3)

    ax_par.plot(it, data["R0"], label="R0", lw=1.5)
    ax_par.plot(it, data["mu"], label="mu", lw=1.5)
    ax_par.plot(it, data["kappa"], label="kappa", lw=1.5)
    ax_par.plot(it, data["alpha"], label="alpha", lw=1.5)
    ax_par.set_xlabel("Adam iteration")
    ax_par.set_ylabel("parameter value")
    ax_par.set_title("Fitted parameters")
    ax_par.legend()
    ax_par.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved loss curves -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    p.add_argument("--log", type=Path, required=True, help="path to the .out log file")
    p.add_argument("--out", type=Path, default=Path("outputs/seir_pinn"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = parse_log(args.log)
    print(f"Parsed {len(data['it'])} logged iterations "
          f"(iters {int(data['it'][0])}..{int(data['it'][-1])}).")
    plot_losses(data, args.out / "loss_curves.png")


if __name__ == "__main__":
    main()