import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

FIG_DIR = Path("outputs/cvar_vs_no_cvar_figures")
FIG_DIR.mkdir(exist_ok=True)

RUNS = [
    ("PINN R0=1.01\n(Kevin's original)", "outputs/pinn_rl_two_stage_cvar"),
    ("PINN R0=1.71\n(reproduced)", "outputs/pinn_rl_two_stage_cvar_r18"),
    ("UK / Libin R0=1.8", "outputs/uk_rl_two_stage_cvar"),
]

C_BASELINE = "#94A3B8"
C_NOCVAR = "#0072B2"
C_CVAR = "#009E73"
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

runs = []
for label, path in RUNS:
    r = json.load(open(Path(path) / "results.json"))
    c = r["candidates"]
    runs.append({
        "label": label,
        "no_closures": c["heur_no_closures"],
        "no_cvar": c[r["best_by_mean"]],
        "no_cvar_name": r["best_by_mean"],
        "with_cvar": c[r["selected_by_cvar10"]],
        "with_cvar_name": r["selected_by_cvar10"],
        "same": r["best_by_mean"] == r["selected_by_cvar10"],
    })

CATS = ["no_closures", "no_cvar", "with_cvar"]
CAT_LABELS = ["No closures", "No CVaR\n(best mean)", "With CVaR\n(selected)"]
CAT_COLORS = [C_BASELINE, C_NOCVAR, C_CVAR]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, run in zip(axes, runs):
    means = [run[k]["mean_attack_rate"] for k in CATS]
    cvars = [run[k]["cvar10_attack_rate"] for k in CATS]
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width / 2, means, width, color=CAT_COLORS, alpha=0.55, label="Mean")
    ax.bar(x + width / 2, cvars, width, color=CAT_COLORS, label="CVaR 10%")
    ax.set_xticks(x)
    ax.set_xticklabels(CAT_LABELS, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title(run["label"], loc="left", fontsize=11, fontweight="bold")
    if run["same"]:
        ax.text(0.98, 0.97, "CVaR agrees\nwith mean", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=MUTED, style="italic")
    else:
        ax.text(0.98, 0.97, "CVaR picked a\ndifferent candidate", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=C_CVAR, style="italic", fontweight="bold")
axes[0].set_ylabel("Attack rate")
handles = [plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.55, label="Mean"),
           plt.Rectangle((0, 0), 1, 1, color=MUTED, label="CVaR 10%")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
fig.suptitle("Baseline vs. Standard RL vs. Risk-Averse (CVaR) RL", fontsize=13.5, fontweight="bold", y=1.14)
fig.tight_layout()
fig.savefig(FIG_DIR / "1_three_way_comparison.png", dpi=170, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, run in zip(axes, runs):
    data = [run[k]["rollout_attack_rates"] for k in CATS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                     medianprops=dict(color=INK, lw=1.6),
                     flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgewidth=0))
    for patch, color in zip(bp["boxes"], CAT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(CAT_LABELS, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title(run["label"], loc="left", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Attack rate (100 stochastic rollouts)")
fig.suptitle("Rollout Spread by Policy Choice", fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG_DIR / "2_rollout_spread_by_policy.png", dpi=170)
plt.close(fig)

fig, axes = plt.subplots(len(runs), 1, figsize=(8, 6.5))
for ax, run in zip(axes, runs):
    nocvar_tail = run["no_cvar"]["cvar10_attack_rate"]
    cvar_tail = run["with_cvar"]["cvar10_attack_rate"]
    lo, hi = sorted([nocvar_tail, cvar_tail])
    pad = max(hi - lo, abs(hi) * 0.05, 1e-12) * 1.8
    ax.plot([nocvar_tail, cvar_tail], [0, 0], color=GRID, lw=2.5, zorder=1)
    ax.scatter([nocvar_tail], [0], s=140, color=C_NOCVAR, zorder=3, label="No CVaR")
    ax.scatter([cvar_tail], [0], s=140, color=C_CVAR, zorder=3, label="With CVaR",
               marker="D" if not run["same"] else "o")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_ylabel(run["label"].replace("\n", " "), fontsize=9.5, rotation=0,
                  ha="right", va="center", labelpad=10)
    tag = "same candidate" if run["same"] else "different candidate"
    ax.text(0.99, 0.85, tag, transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.6), ncol=2)
axes[-1].set_xlabel("CVaR 10% attack rate (worst-decile outcome) -- each row its own scale")
fig.suptitle("Did Choosing by CVaR Instead of Mean Change the Tail Risk?",
              fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "3_did_cvar_help_lollipop.png", dpi=170, bbox_inches="tight")
plt.close(fig)

run = runs[0]
fig, ax = plt.subplots(figsize=(7, 5))
labels = [f"No CVaR\n({run['no_cvar_name']})", f"With CVaR\n({run['with_cvar_name']})"]
means = [run["no_cvar"]["mean_attack_rate"], run["with_cvar"]["mean_attack_rate"]]
cvars = [run["no_cvar"]["cvar10_attack_rate"], run["with_cvar"]["cvar10_attack_rate"]]
x = np.arange(2)
width = 0.35
ax.bar(x - width / 2, means, width, color=[C_NOCVAR, C_CVAR], alpha=0.55, label="Mean")
ax.bar(x + width / 2, cvars, width, color=[C_NOCVAR, C_CVAR], label="CVaR 10%")
trans = ax.get_xaxis_transform()
for xi, v in zip(x, means):
    if v == 0.0:
        ax.text(xi - width / 2, 0.52, "= 0.0\nexactly", ha="center", va="bottom",
                fontsize=8, color=INK, transform=trans)
for xi, v in zip(x, cvars):
    if v == 0.0:
        ax.text(xi + width / 2, 0.52, "= 0.0\nexactly", ha="center", va="bottom",
                fontsize=8, color=INK, transform=trans)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.set_ylabel("Attack rate")
ax.set_title("PINN R0=1.01 — The One Run Where CVaR Picked a\nDifferent Candidate (both ~1e-8, i.e. both noise)",
              loc="left", fontsize=11.5, fontweight="bold")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "4_pinn_divergence_case.png", dpi=170)
plt.close(fig)

print("Saved ->", FIG_DIR)
for p in sorted(FIG_DIR.glob("*.png")):
    print(" ", p)
