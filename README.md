# PINN surrogate for large-scale epidemic control

Extends Libin et al., _Deep reinforcement learning for large-scale epidemic control_, ECML 2020,
with a **physics-informed neural network (PINN)** used as a fast, differentiable RL environment
in place of the SEIR SDE simulator, trained on the 2009 A(H1N1)pdm09 Great Britain data.
Stack: **PyTorch** · **DeepXDE** (`DDE_BACKEND=pytorch`) · **stable-baselines3** · **gymnasium** · **tqdm** · numpy ≥ 1.24.

---

## Installation

```shell
git clone https://github.com/plibin-vub/epi-rl.git
cd epi-rl
pip install -e .          # installs both epcontrol and pinn_seir
export DDE_BACKEND=pytorch
```

All commands below are run **from the repository root** (scripts use paths relative to it).

---

## Repository layout

| Path | Contents |
|---|---|
| `pinn_seir/` | The PINN surrogate: model, network, physics residuals, config, data loader, training entry point |
| `epcontrol/` | Original Libin SEIR SDE simulator and RL environment (unchanged, GPL) |
| `scripts/` | All runnable scripts (PINN training via `python -m pinn_seir.fit_pinn`, diagnostics, RL validation, plots, SLURM `.sbatch`) |
| `outputs/` | All results: checkpoints (`.pt`), `results.json`, `params.json`, figures |
| `logs/` | All training / job logs (`.out`, `.log`) |
| `notebooks/` | `kevin_fixes_review.ipynb` — every fix as runnable code + the `evaluate_pinn` interface |

---

## Data

| Path | Contents |
|---|---|
| `data/great_brittain/census.csv` | Full GB census, 378 districts × 4 age bands |
| `data/great_brittain/commute.csv` | Inter-district commute flux matrix (378 × 378) |
| `data/contacts/` | Age-structured contact matrices, school (`conversational_school.csv`) and holiday (`conversational_no_school.csv`) |
| `data/great_brittain/crosswalk.tsv` | District → nation map (England / Scotland / Wales), needed to aggregate to the per-nation target |
| `data/epidemic/uk_flu_per_100000.csv` | Observed 2009 A(H1N1) weekly ILI rate per 100k, per nation — the PINN's fit target |

---

## 1 — Train the PINN surrogate

Fits the policy-conditioned SEIAR PINN to the observed ILI curve. Writes `checkpoint.pt`,
`checkpoint_iterNNNNNN.pt` (every `--checkpoint-every`), `params.json`, and `fit.png` to `--out`.

```shell
python -m pinn_seir.fit_pinn \
  --census    data/great_brittain/census.csv \
  --commute   data/great_brittain/commute.csv \
  --crosswalk data/great_brittain/crosswalk.tsv \
  --contacts  data/contacts \
  --flu       data/epidemic/uk_flu_per_100000.csv \
  --adam-iters 20000 --dropout-rate 0.1 \
  --n-schedules 32 --n-collocation 64 --w-phys 100 \
  --checkpoint-every 5000 \
  --out outputs/seir_pinn/my_run
```

`--dropout-rate > 0` enables MC-dropout sampling (the PINN's analogue of the SDE noise).
Run `--help` to see every flag (`--hidden-layers`, `--hidden-width`, `--r0-init`, `--train-alpha`, …).

---

## 2 — The fixes (loss / training terms)

Every fix is a flag on the command in §1. The table gives the flags, the checkpoint each run
produced, and the measured result (susceptible-response range over 0–90 % closed, and final R₀).

| Fix | Distinguishing flags | Checkpoint | S range 0–90 % | R₀ |
|---|---|---|---|---|
| Before fix | (earlier over-trained run) | `outputs/seir_pinn/11426009` | 41,260 | 1.01 |
| **Fix 1 — schedule diversity + physics weight** | `--n-schedules 32 --n-collocation 64 --w-phys 100` | `outputs/seir_pinn/trillium_fixed_run` | **550,660** | 1.75 |
| Attempt 1 — self-referential activity weighting | *(discarded loss variant, see notebook)* | `outputs/seir_pinn/trillium_activity_weight_test` | 16 | 1.70 |
| Attempt 2 — data-prior activity weighting | `--phys-activity-floor 0.1` | `outputs/seir_pinn/trillium_activity_weight_v2` | 20,880 | 1.68 |
| Attempt 3 — monotonicity + cumulative anchor | `--phys-activity-floor 0.1 --w-monotonic 10 --w-cumulative 10` | `outputs/seir_pinn/trillium_mono_cumulative` | 58,972 | 0.72 |

The full training loss is `w_phys·L_phys + w_junction·L_junction + w_data·L_data + w_ic·L_ic + w_monotonic·L_mono + w_cumulative·L_cum`.
`L_phys` is importance-weighted per week by a fixed prior built from the observed incidence
(Attempt 2); `L_mono` is a hinge on `dS/dτ ≤ 0`, `dR/dτ ≥ 0` and `L_cum` anchors total R at
the last observed week to the data-implied attack rate (Attempt 3). The exact code of each is
in `notebooks/kevin_fixes_review.ipynb`.

> **Note:** only Fix 1 gave a confirmed, reproducible improvement. Attempts 1–3 either regressed
> the closure response or collapsed R₀ below the epidemic threshold. See §7.

---

## 3 — Diagnose a checkpoint

Sweeps the school-closure fraction 0 → 100 % at a fixed week and reports whether the network
responds smoothly to partial closures (RL-usable) or only reacts at the extremes.

```shell
python scripts/check_closure_sensitivity.py \
  --checkpoint outputs/seir_pinn/trillium_fixed_run/checkpoint_iter020000.pt \
  --flu data/epidemic/uk_flu_per_100000.csv \
  --week 20 --device cuda
```

Prints S and I at each closure fraction plus a PASS/FAIL verdict on the response range.

---

## 4 — RL validation: two-stage CVaR selection on the PINN

Trains 5 PPO seeds + 5 heuristic schedules, evaluates every candidate over 100 MC-dropout
rollouts, and selects the one minimising CVaR at 10 % (mean of the worst decile).
Set the target checkpoint by editing the `CKPT` and `OUT_DIR` constants at the top of the script.

```shell
python scripts/run_pinn_rl_two_stage_cvar.py
```

Writes `results.json`, `schedules.npz`, and `ppo_seed*.zip` to `OUT_DIR` under `outputs/`.

> **Warning:** on every PINN checkpoint tried so far the attack rate is ~1e-8 for **all**
> candidates (noise floor) — the surrogate does not yet produce a usable reward signal. See §7.

---

## 5 — RL validation: same procedure on the original Libin ODE

Identical two-stage CVaR procedure, but against the Libin SEIR SDE simulator (`sde=True`)
instead of the PINN, so results are directly comparable. This is the validated case.

```shell
python scripts/run_uk_rl_two_stage_cvar.py
```

Writes `results.json` to `outputs/uk_rl_two_stage_cvar/`. Here the candidates spread across
44.8–45.0 % attack rate (real signal): the CVaR-selected `heur_spread_out` beats the best
mean-only PPO seed by 0.18 percentage points.

---

## 6 — Plot the CVaR comparisons

No new experiments; pure analysis of the `results.json` files from §4 / §5. Each writes PNGs
to an `outputs/*_figures/` directory.

```shell
python scripts/plot_libin_cvar_vs_no_cvar.py     # standard-RL vs CVaR-RL, Libin ODE only
python scripts/plot_cvar_vs_no_cvar.py           # three-way: PINN R0=1.01, PINN R0=1.71, ODE
python scripts/plot_pinn_vs_uk_comparison.py     # PINN vs ODE candidate landscapes
python scripts/plot_two_stage_cvar.py            # per-candidate landscape + rollout spread
```

---

## 7 — Review notebook and the `evaluate_pinn` interface

`notebooks/kevin_fixes_review.ipynb` contains every fix as **runnable in-process code**
(no shell calls): the training config per fix, the exact loss functions, checkpoint loading,
and the diagnostics — with all cell outputs and plots already saved.

To test a new fix, train a checkpoint (§1) then connect it to the interface:

```python
evaluate_pinn("outputs/seir_pinn/my_run/checkpoint.pt")
# {'R0': ..., 'closure_range_0_90': ..., 'season_R_monotonic': ..., 'within_week_R_rising': ...}
```

`evaluate_pinn` also accepts a `PINNTrainer` object, so a modified network architecture can be
passed directly:

```python
tr = PINNTrainer(data, mcfg, TrainConfig(dropout_rate=0.1, device=DEVICE))
tr.load_checkpoint("outputs/seir_pinn/my_run/checkpoint.pt")
evaluate_pinn(tr)
```

---

## Status

Fix 1 (schedule diversity + physics weight) resolves the partial-closure response but not the
season-long coherence needed for a usable RL reward: the recovered count is non-monotonic across
weeks and frozen within a week, so §4 stays at the noise floor. Proposed next steps (self-adaptive
residual weighting, causality-respecting training, continuous-time encoding, hard constraints) are
documented separately.

---

## Original Libin RL scripts

The single- and multi-district PPO training / evaluation / exhaustive-search scripts from the
ECML 2020 paper remain under `scripts/` (`seir_environment_single_sb_ppo.py`,
`seir_environment_multi_sb_ppo.py`, `seir_environment_joint_run_ppo_policy.py`,
`UK_RL_school_weekly_search.py`, `plot_ppo_reward_curve.py`) and run against the ODE simulator
as before. Run any with `--help` for its flags.
