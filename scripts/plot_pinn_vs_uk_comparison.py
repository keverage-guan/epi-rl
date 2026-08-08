import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

PINN_DIR = Path("outputs/pinn_rl_two_stage_cvar_r18")
UK_DIR = Path("outputs/uk_rl_two_stage_cvar")
FIG_DIR = Path("outputs/pinn_vs_uk_figures")
FIG_DIR.mkdir(exist_ok=True)

C_PINN = "#0072B2"
C_UK = "#D55E00"
C_SELECTED = "#009E73"
GRID = "#D9D9D9"
INK = "#1F2937"
MUTED = "#6B7280"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

pinn = json.load(open(PINN_DIR / "results.json"))
uk = json.load(open(UK_DIR / "results.json"))
pinn_c, uk_c = pinn["candidates"], uk["candidates"]
names = list(pinn_c.keys())

def pretty(name):
    return name.replace("ppo_seed", "PPO seed ").replace("heur_", "").replace("_", " ")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, res, color, label in [(axes[0], pinn_c, C_PINN, "PINN (R0=1.706)"),
                                (axes[1], uk_c, C_UK, "UK / Libin SDE (R0=1.8)")]:
    means = [res[n]["mean_attack_rate"] for n in names]
    cvars = [res[n]["cvar10_attack_rate"] for n in names]
    marker = lambda n: "o" if n.startswith("ppo_seed") else "s"
    for n, m, c in zip(names, means, cvars):
        ax.scatter(m, c, s=100, color=color, marker=marker(n), edgecolor="white",
                   linewidth=0.8, zorder=3, alpha=0.85)
    lo, hi = min(means + cvars), max(means + cvars)
    pad = (hi - lo) * 0.15 + 1e-12
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GRID, lw=1.0, ls="--", zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xlabel("Mean attack rate")
    ax.set_ylabel("CVaR 10% attack rate")
    ax.set_title(label, loc="left", fontsize=11.5, fontweight="bold")
handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=MUTED, markersize=9, label="PPO seed"),
           Line2D([0], [0], marker="s", color="none", markerfacecolor=MUTED, markersize=9, label="Heuristic")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.suptitle("Candidate Landscape: Same 10 Policies, Two Simulators", fontsize=13, fontweight="bold", y=1.1)
fig.tight_layout()
fig.savefig(FIG_DIR / "1_candidate_landscape_side_by_side.png", dpi=170, bbox_inches="tight")
plt.close(fig)

def effect_size(res):
    means = [res[n]["mean_attack_rate"] for n in names]
    return max(means) - min(means)

pinn_effect = abs(effect_size(pinn_c))
uk_effect = abs(effect_size(uk_c))

fig, ax = plt.subplots(figsize=(6.5, 5))
bars = ax.bar(["PINN", "UK / Libin"], [pinn_effect, uk_effect], color=[C_PINN, C_UK], width=0.55)
ax.set_yscale("log")
ax.set_ylabel("Spread across candidates\n(max - min mean attack rate, log scale)")
ax.set_title("Does the Closure Policy Even Matter?", loc="left", fontsize=12.5, fontweight="bold")
for bar, val in zip(bars, [pinn_effect, uk_effect]):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.3, f"{val:.1e}",
            ha="center", fontsize=10, color=INK, fontweight="bold")
ratio = uk_effect / max(pinn_effect, 1e-300)
ax.text(0.5, -0.16, f"UK/Libin's policy sensitivity is ~{ratio:.0e}x larger than the PINN's",
        transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig(FIG_DIR / "2_effect_size_log_comparison.png", dpi=170)
plt.close(fig)

order = sorted(names, key=lambda n: uk_c[n]["cvar10_attack_rate"])
x = np.arange(len(order))
width = 0.38
fig, ax = plt.subplots(figsize=(9.5, 5))
means = [uk_c[n]["mean_attack_rate"] * 100 for n in order]
cvars = [uk_c[n]["cvar10_attack_rate"] * 100 for n in order]
ax.bar(x - width / 2, means, width, label="Mean", color="#94A3B8")
ax.bar(x + width / 2, cvars, width, label="CVaR 10%", color=C_UK)
sel = uk["selected_by_cvar10"]
for i, n in enumerate(order):
    if n == sel:
        ax.text(i, max(means[i], cvars[i]) + 0.4, "selected", ha="center",
                fontsize=8, color=C_SELECTED, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([pretty(n) for n in order], rotation=30, ha="right", fontsize=8.5)
ax.set_ylabel("Attack rate (%)")
ax.set_ylim(min(means + cvars) - 0.15, max(means + cvars) + 0.35)
ax.set_title("UK / Libin Model — All Candidates, Ranked by Tail Risk\n"
             "(y-axis zoomed to show real differences — all candidates are near 45%)",
             loc="left", fontsize=12, fontweight="bold")
ax.legend(frameon=False)
fig.subplots_adjust(bottom=0.28, top=0.88)
fig.savefig(FIG_DIR / "3_uk_all_candidates_ranked.png", dpi=170)
plt.close(fig)

show = ["heur_no_closures", pinn["selected_by_cvar10"]]
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
for ax, res, color, label in [(axes[0], pinn_c, C_PINN, "PINN"), (axes[1], uk_c, C_UK, "UK / Libin")]:
    vals = [res[n]["rollout_attack_rates"] for n in show]
    bp = ax.boxplot(vals, patch_artist=True, widths=0.5, showfliers=True,
                     medianprops=dict(color=INK, lw=1.6),
                     flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgewidth=0))
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["No closures", "Selected\ncandidate"], fontsize=9.5)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Attack rate (100 stochastic rollouts)")
    ax.set_title(label, loc="left", fontsize=11.5, fontweight="bold")
fig.suptitle("Rollout-to-Rollout Spread: Flat vs. Real Variance", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG_DIR / "4_rollout_distribution_comparison.png", dpi=170)
plt.close(fig)

print("Saved figures ->", FIG_DIR)
for p in sorted(FIG_DIR.glob("*.png")):
    print(" ", p)
