import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

UK_DIR = Path("outputs/uk_rl_two_stage_cvar")
FIG_DIR = Path("outputs/libin_cvar_vs_no_cvar_figures")
FIG_DIR.mkdir(exist_ok=True)

C_BASELINE = "#94A3B8"
C_NOCVAR = "#0072B2"
C_CVAR = "#009E73"
C_PPO_OTHER = "#B9DAF0"
C_HEUR_OTHER = "#FBD8B0"
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

r = json.load(open(UK_DIR / "results.json"))
c = r["candidates"]
names = list(c.keys())
ppo_names = [n for n in names if n.startswith("ppo_seed")]
best_ppo = min(ppo_names, key=lambda k: c[k]["mean_attack_rate"])
cvar_pick = r["selected_by_cvar10"]

def pretty(n):
    return n.replace("ppo_seed", "PPO seed ").replace("heur_", "").replace("_", " ")

fig, ax = plt.subplots(figsize=(7, 5.5))
cats = ["No closures", f"No CVaR\n({pretty(best_ppo)})", f"With CVaR\n({pretty(cvar_pick)})"]
means = [c["heur_no_closures"]["mean_attack_rate"] * 100,
         c[best_ppo]["mean_attack_rate"] * 100,
         c[cvar_pick]["mean_attack_rate"] * 100]
cvars = [c["heur_no_closures"]["cvar10_attack_rate"] * 100,
         c[best_ppo]["cvar10_attack_rate"] * 100,
         c[cvar_pick]["cvar10_attack_rate"] * 100]
colors = [C_BASELINE, C_NOCVAR, C_CVAR]
x = np.arange(3)
width = 0.35
ax.bar(x - width / 2, means, width, color=colors, alpha=0.55, label="Mean")
ax.bar(x + width / 2, cvars, width, color=colors, label="CVaR 10%")
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=9.5)
ax.set_ylim(44.6, 45.15)
ax.set_ylabel("Attack rate (%)")
ax.set_title("UK / Libin Model: Standard RL vs. Risk-Averse (CVaR) RL",
              loc="left", fontsize=12.5, fontweight="bold")
ax.legend(frameon=False)
delta = (c[best_ppo]["mean_attack_rate"] - c[cvar_pick]["mean_attack_rate"]) * 100
ax.text(0.5, -0.17, f"CVaR selection beats standard-RL-alone by {delta:.2f} percentage "
                     f"points mean attack rate\n(it picked a heuristic PPO never found)",
        transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig(FIG_DIR / "1_headline_no_cvar_vs_cvar.png", dpi=170)
plt.close(fig)

order = sorted(names, key=lambda n: c[n]["mean_attack_rate"])
fig, ax = plt.subplots(figsize=(10, 5.2))
bar_colors = []
for n in order:
    if n == cvar_pick:
        bar_colors.append(C_CVAR)
    elif n == best_ppo:
        bar_colors.append(C_NOCVAR)
    elif n.startswith("ppo_seed"):
        bar_colors.append(C_PPO_OTHER)
    else:
        bar_colors.append(C_HEUR_OTHER)
means = [c[n]["mean_attack_rate"] * 100 for n in order]
bars = ax.bar(range(len(order)), means, color=bar_colors)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([pretty(n) for n in order], rotation=32, ha="right", fontsize=8.5)
ax.set_ylim(44.6, 45.1)
ax.set_ylabel("Mean attack rate (%)")
ax.set_title("Every Candidate, Best to Worst — PPO Seeds vs. Heuristics",
              loc="left", fontsize=12.5, fontweight="bold")
for i, n in enumerate(order):
    if n == cvar_pick:
        ax.text(i, means[i] - 0.03, "picked by\nCVaR", ha="center", va="top",
                fontsize=7.5, color="white", fontweight="bold")
    elif n == best_ppo:
        ax.text(i, means[i] - 0.03, "best PPO\n(no CVaR)", ha="center", va="top",
                fontsize=7.5, color="white", fontweight="bold")
from matplotlib.patches import Patch
handles = [Patch(color=C_CVAR, label="Selected by CVaR"), Patch(color=C_NOCVAR, label="Best PPO (no CVaR)"),
           Patch(color=C_PPO_OTHER, label="Other PPO seeds"), Patch(color=C_HEUR_OTHER, label="Other heuristics")]
ax.legend(handles=handles, frameon=False, fontsize=8.5)
fig.subplots_adjust(bottom=0.3, top=0.88)
fig.savefig(FIG_DIR / "2_all_candidates_ppo_vs_heuristic.png", dpi=170)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 5.2))
show = ["heur_no_closures", best_ppo, cvar_pick]
labels = ["No closures", f"No CVaR\n({pretty(best_ppo)})", f"With CVaR\n({pretty(cvar_pick)})"]
data = [np.array(c[n]["rollout_attack_rates"]) * 100 for n in show]
bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True,
                 medianprops=dict(color=INK, lw=1.6),
                 flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgewidth=0))
for patch, color in zip(bp["boxes"], [C_BASELINE, C_NOCVAR, C_CVAR]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Attack rate (%), 100 stochastic rollouts")
ax.set_title("Rollout Spread: Baseline vs. Standard RL vs. CVaR-Selected",
              loc="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "3_rollout_distribution.png", dpi=170)
plt.close(fig)

sched = np.load(UK_DIR / "schedules.npz", allow_pickle=True)
districts = list(sched["controlled_districts"])
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, name, color, title in [(axes[0], best_ppo, C_NOCVAR, f"No CVaR: {pretty(best_ppo)}"),
                                 (axes[1], cvar_pick, C_CVAR, f"With CVaR: {pretty(cvar_pick)}")]:
    s = sched[name]
    ax.imshow(1.0 - s.T, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(np.arange(len(districts)))
    ax.set_yticklabels(districts, fontsize=8)
    ax.set_xlabel("Week")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=color)
fig.suptitle("Closure Schedules (black = closed, white = open)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG_DIR / "4_closure_schedule_comparison.png", dpi=170)
plt.close(fig)

print("Saved ->", FIG_DIR)
for p in sorted(FIG_DIR.glob("*.png")):
    print(" ", p)
