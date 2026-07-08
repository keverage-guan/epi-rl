# Modernisation notes

This is a modernised fork of `plibin/epi-rl` (Libin et al., 2021) that runs on a
current Python and dependency stack. The epidemic model is unchanged; only the RL
tooling and deprecated calls were updated.

## Dependency changes
| Original | Modernised |
|---|---|
| `tensorflow<2.0.0` | removed |
| `stable-baselines` (TF1, `PPO2`) | `stable-baselines3` (PyTorch, `PPO`) |
| `gym` (legacy API) | `gymnasium` (`reset -> (obs, info)`, `step -> 5-tuple`) |
| `numpy` (with `np.product`, `np.bool`) | `numpy>=1.24,<2.1` (`np.prod`, `bool`) |
| implicit `PYTHONPATH` | installable package via `pyproject.toml` |

## What was kept unchanged
- `epcontrol/UK_SEIR_Eames.py`: the age-structured SEIR meta-population with the
  Euler-Maruyama SDE and the Cinlar import process (numba `_step`).
- `epcontrol/compartments/`, `epcontrol/census/`, `epcontrol/utils.py`,
  `epcontrol/UK_RL_school_weekly.py`, and all data files.

## Code changes
- `epcontrol/seir_environment.py`: ported to gymnasium; `np.product -> np.prod`;
  `np.bool -> bool`; `Box` given `dtype=np.float32`; `reset` and `step` follow the
  gymnasium API.
- `epcontrol/wrappers.py`: ported to gymnasium; wrapper `reset`/`step` return the
  new tuples.
- `epcontrol/compartments/contacts/Eames2012.py`: `compute_beta` uses
  `np.max(np.abs(eigvals))` (dominant eigenvalue by magnitude).
- `scripts/*_sb_ppo.py` and `scripts/*_run_ppo_policy.py`: rewritten for
  stable-baselines3 (`PPO`, `MlpPolicy`, `DummyVecEnv`, `Monitor`). Argument
  mapping: `nminibatches -> batch_size = n_steps // n_minibatches`,
  `noptepochs -> n_epochs`, `policy_kwargs["layers"] -> net_arch`.

## Out of scope (moved to `legacy/`)
The PyMARL/SMAC multi-agent experiments (`multiagent/seir.py`, `seir_pymarl.py`)
depend on external frameworks not maintained for current Python. The joint
multi-district PPO path in `scripts/` reproduces the multi-region experiment
without them.

## Install and validate
```shell
pip install -e .
python scripts/seir_environment_single_sb_ppo.py --R0 1.8 --district_name Greenwich \
  --budget_in_weeks 2 --census ./data/single/census.csv --total_timesteps 20000 \
  --outcome ar --monitor_path /tmp/SEIR_Greenwich_PPO
python scripts/seir_environment_single_run_ppo_policy.py --R0 1.8 --district_name Greenwich \
  --budget_in_weeks 2 --census ./data/single/census.csv --runs 10 --outcome ar \
  --path /tmp/SEIR_Greenwich_PPO
```
