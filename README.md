# Deep reinforcement learning for large-scale epidemic control

Code accompanying Libin et al., _Deep reinforcement learning for large-scale epidemic control_, ECML 2020.
Stack: **gymnasium** · **stable-baselines3** (PyTorch) · **tqdm** · numpy ≥ 1.24.
See `MODERNIZATION.md` for the full dependency and code change log.

---

## Installation

```shell
git clone https://github.com/plibin-vub/epi-rl.git
cd epi-rl
pip install -e .
```

---

## Data

| Path | Contents |
|---|---|
| `data/single/census.csv` | Single-district census (Greenwich) |
| `data/great_brittain/census.csv` | Full GB census (378 districts) |
| `data/great_brittain/commute.csv` | Inter-district commute flux matrix |
| `data/contacts/` | Age-structured contact matrices (school / no-school) |

---

## 1 — Train PPO on a single district

```shell
python scripts/seir_environment_single_sb_ppo.py \
  --R0 1.8 \
  --district_name Greenwich \
  --budget_in_weeks 2 \
  --census ./data/single/census.csv \
  --total_timesteps 1000000 \
  --outcome ar \
  --monitor_path /tmp/SEIR_Greenwich_PPO
```

`--outcome` accepts `ar` (attack rate) or `pd` (peak day).
Run `--help` to see all hyperparameter flags (`--learning_rate`, `--n_hidden_layers`, `--n_steps`, …).

---

## 2 — Evaluate a single-district policy

```shell
python scripts/seir_environment_single_run_ppo_policy.py \
  --R0 1.8 \
  --district_name Greenwich \
  --budget_in_weeks 2 \
  --census ./data/single/census.csv \
  --runs 10 \
  --outcome ar \
  --path /tmp/SEIR_Greenwich_PPO
```

Prints one `<outcome>-improvement` value per run to stdout.

---

## 3 — Train PPO jointly on 11 districts

Districts: Cornwall, Plymouth, Torbay, East Devon, Exeter, Mid Devon, North Devon, South Hams, Teignbridge, Torridge, West Devon.

```shell
python scripts/seir_environment_multi_sb_ppo.py \
  --R0 1.8 \
  --district_name Cornwall \
  --budget_in_weeks 2 \
  --census ./data/great_brittain/census.csv \
  --flux ./data/great_brittain/commute.csv \
  --total_timesteps 1000000 \
  --monitor_path /tmp/SEIR_11districts_PPO
```

`--district_name` sets the epidemic seed district.
Run `--help` to see all hyperparameter flags.

---

## 4 — Evaluate the joint 11-district policy

```shell
python scripts/seir_environment_joint_run_ppo_policy.py \
  --R0 1.8 \
  --district_name Cornwall \
  --budget_in_weeks 2 \
  --census ./data/great_brittain/census.csv \
  --flux ./data/great_brittain/commute.csv \
  --runs 10 \
  --path /tmp/SEIR_11districts_PPO
```

Prints one `ar-improvement` value per run to stdout.

---

## 5 — Ground-truth exhaustive search (single district)

Enumerates every valid school-closure schedule and evaluates each one deterministically.
Output is a CSV printed to stdout: `combination,<outcome>_improvement`.

```shell
python scripts/UK_RL_school_weekly_search.py \
  --R0 1.8 \
  --district Greenwich \
  --grouped-census-fn ./data/single/census.csv \
  --weeks 43 \
  --budget-weeks 2 \
  --outcome ar
```

> **Warning:** the search space is C(`--weeks`, `--budget-weeks`) combinations.
> For `--weeks 43 --budget-weeks 2` that is 903 simulations; for larger budgets it grows fast.

---

## 6 — Plot a PPO reward curve

Reads the Monitor CSV written during training and saves a PNG.

```shell
python scripts/plot_ppo_reward_curve.py \
  --path /tmp/SEIR_Greenwich_PPO \
  --out_file /tmp/SEIR_Greenwich_reward.png
```
