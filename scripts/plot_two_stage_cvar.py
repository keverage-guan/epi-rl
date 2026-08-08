import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = Path("outputs/pinn_rl_two_stage_cvar")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

C_PPO = "#0072B2"
C_HEUR = "#E69F00"
C_SELECTED = "#009E73"
GRID = "#D9D9D9"
INK = "#1F2937"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

res = json.load(open(OUT_DIR / "results.json"))
cands = res["candidates"]
selected = res["selected_by_cvar10"]
best_mean = res["best_by_mean"]
names = list(cands.keys())

def kind(name):
    return "PPO seed" if name.startswith("ppo_seed") else "Heuristic"

def color(name):
    return C_PPO if name.startswith("ppo_seed") else C_HEUR

def pretty(name):
    return name.replace("ppo_seed", "PPO seed ").replace("heur_", "").replace("_", " ")

sf = 1e5

sf2 = 1e11
fig, ax = plt.subplots(figsize=(7.8, 5.8))
for name in names:
    c = cands[name]
    mean_ar, cvar_ar = c["mean_attack_rate"] * sf2, c["cvar10_attack_rate"] * sf2
    marker = "o" if name.startswith("ppo_seed") else "s"
    ax.scatter(mean_ar, cvar_ar, s=110, color=color(name), marker=marker,
               edgecolor="white", linewidth=0.8, zorder=3, alpha=0.85)

sel = cands[selected]
ax.scatter(sel["mean_attack_rate"] * sf2, sel["cvar10_attack_rate"] * sf2,
           s=280, facecolors="none", edgecolors=C_SELECTED, linewidth=2.2, zorder=4)

lo = min(min(c["mean_attack_rate"] for c in cands.values()),
         min(c["cvar10_attack_rate"] for c in cands.values())) * sf2
hi = max(max(c["mean_attack_rate"] for c in cands.values()),
         max(c["cvar10_attack_rate"] for c in cands.values())) * sf2
pad = (hi - lo) * 0.25 + 1e-9
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GRID, lw=1.0, ls="--", zorder=1)
ax.set_xlim(lo - pad, hi + pad)
ax.set_ylim(lo - pad, hi + pad)

from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_PPO, markersize=9, label="PPO seed"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=C_HEUR, markersize=9, label="Heuristic"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
           markeredgecolor=C_SELECTED, markeredgewidth=2, markersize=13, label="Selected"),
]
ax.legend(handles=handles, frameon=False, loc="upper left")
ax.set_xlabel(r"Mean attack rate ($\times 10^{-11}$ per 100,000)")
ax.set_ylabel(r"CVaR 10% attack rate ($\times 10^{-11}$ per 100,000)")
ax.set_title("Candidate Landscape — Mean vs. Tail Risk", loc="left", fontsize=12, fontweight="bold")
ax.text(0.02, 0.02, "All 10 candidates cluster within float32 noise of zero\n"
                     "(note the axis scale: $10^{-11}$) — no candidate is\n"
                     "distinguishable from any other.",
        transform=ax.transAxes, fontsize=8.5, color="#6B7280", va="bottom")
fig.tight_layout()
fig.savefig(FIG_DIR / "1_candidate_landscape.png", dpi=170)
plt.close(fig)

show = sorted(set([selected, best_mean, "heur_no_closures"]), key=names.index)
fig, ax = plt.subplots(figsize=(8, 4.6))
data_ = [np.array(cands[n]["rollout_attack_rates"]) * sf for n in show]
bp = ax.boxplot(data_, vert=True, patch_artist=True, widths=0.5, showfliers=True,
                 medianprops=dict(color=INK, lw=1.6),
                 flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgewidth=0))
for patch, n in zip(bp["boxes"], show):
    patch.set_facecolor(C_SELECTED if n == selected else color(n))
    patch.set_alpha(0.75)
ax.set_xticks(range(1, len(show) + 1))
ax.set_xticklabels([pretty(n) for n in show], fontsize=9)
ax.set_ylabel("Attack rate (per 100,000)")
ax.set_title("Stochastic Rollout Spread (100 MC-dropout draws)",
             loc="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "2_rollout_distributions.png", dpi=170)
plt.close(fig)

order = sorted(names, key=lambda n: cands[n]["cvar10_attack_rate"])
x = np.arange(len(order))
width = 0.38
fig, ax = plt.subplots(figsize=(9.5, 4.8))
means = [cands[n]["mean_attack_rate"] * sf for n in order]
cvars = [cands[n]["cvar10_attack_rate"] * sf for n in order]
bars1 = ax.bar(x - width / 2, means, width, label="Mean", color="#94A3B8")
bars2 = ax.bar(x + width / 2, cvars, width, label="CVaR 10%", color=C_PPO)
for i, n in enumerate(order):
    if n == selected:
        ax.text(i, max(means[i], cvars[i]) + 0.03, "selected", ha="center",
                 fontsize=8, color=C_SELECTED, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([pretty(n) for n in order], rotation=30, ha="right", fontsize=8.5)
ax.set_ylabel("Attack rate (per 100,000)")
ax.set_title("All Candidates, Ranked by Tail Risk\n"
             "(bar heights $\\leq$ 0.002 per 100,000 — indistinguishable from zero)",
             loc="left", fontsize=12, fontweight="bold")
ax.legend(frameon=False)
fig.subplots_adjust(bottom=0.28, top=0.82)
fig.savefig(FIG_DIR / "3_all_candidates_ranked.png", dpi=170)
plt.close(fig)

sched = cands[selected]["closed_weeks_per_district"]
names_d = list(sched.keys())
vals = list(sched.values())
fig, ax = plt.subplots(figsize=(7.5, 4.2))
bar_colors = [C_SELECTED if v > 0 else "#D1D5DB" for v in vals]
ax.barh(names_d, vals, color=bar_colors)
ax.set_xlabel("Closed weeks (of 43, budget max 6)")
ax.set_title(f"Selected Candidate ({pretty(selected)}) — Closure Budget Used",
             loc="left", fontsize=11.5, fontweight="bold")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIG_DIR / "4_selected_closure_budget.png", dpi=170)
plt.close(fig)

print("Saved 4 figures ->", FIG_DIR)
for p in sorted(FIG_DIR.glob("*.png")):
    print(" ", p)
