import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs/pinn_rl_two_stage_cvar_r18")
data = json.load(open(OUT_DIR / "closure_sensitivity_sweep.json"))
fracs = [d["frac_closed"] * 100 for d in data]
S = [d["S"] / 1e6 for d in data]

GRID = "#D9D9D9"
INK = "#1F2937"
C_LINE = "#CC3311"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

ymin, ymax = min(S) - 2, max(S) + 1
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(fracs, S, color=C_LINE, lw=2.4, marker="o", markersize=3.5, zorder=3)
ax.axvspan(0, 90, color="#94A3B8", alpha=0.12, zorder=0)
ax.set_ylim(ymin, ymax)
ax.text(45, ymin + (ymax - ymin) * 0.15, "flat: no learned response\nto partial closures (0-90%)",
        ha="center", fontsize=9.5, color="#475569")
ax.annotate("sudden jump only\nat 100% closed", xy=(100, S[-1]), xytext=(62, ymin + (ymax - ymin) * 0.35),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.2), fontsize=9.5, color=INK)
ax.set_xlabel("% of districts with schools closed (single query, week 20)")
ax.set_ylabel("Predicted total susceptibles (millions)")
ax.set_title("PINN Response to Closure Fraction — Reproduced Checkpoint (R0=1.71)",
              loc="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "closure_sensitivity_sweep.png", dpi=170)
print("saved", OUT_DIR / "closure_sensitivity_sweep.png")
