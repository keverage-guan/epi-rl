# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### ✨ Added
- `epcontrol/wrappers.py`: `MultiAgentCVaRReward`, a multi-district reward wrapper that
  returns the mean of the worst `alpha` fraction of districts' susceptible loss (each
  normalized by the same `max_districts_susceptibles` constant `MultiAgentSelectReward`
  uses, so terms are on a comparable scale) instead of the aggregate outcome across all
  districts. `epcontrol/risk.py:AlphaAnnealingCallback` linearly anneals `alpha` from 1.0
  (full aggregate, same objective as `mean`) down to a target value over training, so the
  harder tail-only objective only kicks in once the policy already has a baseline of
  competence. Wired into `scripts/seir_environment_multi_sb_ppo.py` via
  `--reward_mode {mean,cvar}`, `--cvar_alpha`, `--cvar_alpha_start`, `--cvar_anneal_fraction`.
  `scripts/seir_environment_joint_run_ppo_policy.py` now reports
  `worst-district-ar-improvement` next to the existing aggregate `ar-improvement`, so `mean`-
  and `cvar`-trained policies can be compared on both metrics.
- `epcontrol/risk.py:probe_returns` and `RiskShapingCallback`, `epcontrol/wrappers.py:RiskShapedReward`:
  single-district risk-shaped PPO. Before each rollout, `RiskShapingCallback` probes the
  current policy over independent stochastic episodes and estimates `nu`, the empirical
  `alpha`-quantile of episode returns; `RiskShapedReward` then subtracts
  `gamma*max(0, nu-episode_return)/alpha` from the terminal reward, so episodes at or above
  `nu` are unpenalized and episodes below it are penalized in proportion to their shortfall.
  Demonstrated in `scripts/seir_environment_single_risk_ppo.py` (`--gamma_risk`, `--alpha`,
  `--n_probe_episodes`).
- `epcontrol/seir_environment.py`: `SEIREnvironment` accepts an optional `model_factory`.
  When given, every `reset()` builds a fresh model from it (e.g. a new district drawn from
  a training pool) instead of resetting the injected `model` in place, enabling ensemble
  training over a pool of transition models. `model` is still required in this case to size
  the observation/reward-normalization bounds.
- `epcontrol/transition_model.py`: `TransitionModel`, a `runtime_checkable` `Protocol`
  defining the contract `SEIREnvironment` requires of its simulation backend
  (`reset`, `seed`, `step`, `total_infected`, `total_susceptibles`,
  `total_susceptibles_district`, `district_idx`, `peak_day`, `district_names`,
  `seir_state`). `epcontrol/UK_SEIR_Eames.py:UK` satisfies it unchanged; any other
  simulator (e.g. a PINN with MC dropout) can be substituted as a drop-in
  replacement for the transition function as long as it satisfies the same
  protocol, without modifying `SEIREnvironment`.
- `pyproject.toml`: installable package via `pip install -e .` (setuptools backend, Python ≥ 3.10).
- `epcontrol/__init__.py`: package marker so the module is importable after `pip install -e .`.
- `MODERNIZATION.md`: comprehensive notes on every dependency and code change made during the port.
- `legacy/` directory: preserves PyMARL/SMAC experiments that depend on unmaintained external frameworks.
  - `legacy/multiagent_seir.py` (was `epcontrol/multiagent/seir.py`)
  - `legacy/seir_pymarl.py` (was `scripts/seir_pymarl.py`)
  - `legacy/seir_environment_multi_run_ppo_policy.py` (was `scripts/seir_environment_multi_run_ppo_policy.py`)
  - `legacy/README.md`

### ⬆️ Changed — dependencies
| Package | Before | After |
|---|---|---|
| `tensorflow<2.0.0` | required | **removed** |
| `stable-baselines` (TF1, `PPO2`) | required | **removed** |
| `gym` (legacy API) | required | **removed** |
| `stable-baselines3` (PyTorch, `PPO`) | — | `>=2.3` |
| `gymnasium` | — | `>=0.29` |
| `torch` | — | `>=2.2` |
| `numpy` | unpinned | `>=1.24,<2.1` |
| `scipy` | unpinned | `>=1.10` |
| `pandas` | unpinned | `>=2.0` |
| `numba` | unpinned | `>=0.60` |
| `matplotlib` | unpinned | `>=3.7` |
| `seaborn` | `>=0.9.0` | `>=0.12` |

### ♻️ Changed — `epcontrol/seir_environment.py`
- Import changed from `gym` → `gymnasium`.
- `np.product(...)` → `np.prod(...)` (removed in NumPy 2.x).
- `np.bool` → `bool` (removed in NumPy 1.24).
- `spaces.Box(low, high)` now explicitly sets `dtype=np.float32`.
- `reset()` signature updated to `reset(*, seed=None, options=None)` returning `(obs, info)`.
- `step()` now returns a 5-tuple `(obs, reward, terminated, truncated, info)` per gymnasium API.
- **Interface refactor**: `SEIREnvironment.__init__` now takes a pre-built `model: TransitionModel`
  instead of the raw epidemiological parameters (`grouped_census`, `flux`, `r0`, `rho`, `gamma`,
  `delta`, `mu`, `sde`). The environment no longer constructs `UK` itself (`_make_model` removed);
  callers construct the model and inject it. This decouples the RL pipeline from the concrete
  transition function so it can be swapped (e.g. for a PINN) without touching this class.
  `scripts/seir_environment_single_sb_ppo.py`, `scripts/seir_environment_single_run_ppo_policy.py`,
  `scripts/seir_environment_multi_sb_ppo.py`, and `scripts/seir_environment_joint_run_ppo_policy.py`
  were updated to construct `UK` directly and pass it in; behavior (default `rho=1`, `gamma=1/1.8`,
  `delta=0.5`, `mu=log(R0)*.6`, `sde=True`) is unchanged. `legacy/` scripts were not updated, per
  `legacy/README.md` they are already out of scope.

### ♻️ Changed — `epcontrol/wrappers.py`
- All `gym` / `gym.spaces` imports replaced with `gymnasium` / `gymnasium.spaces`.
- `MultiAgentSelectReward.reset` / `step` updated to handle gymnasium's 5-tuple step and `(obs, info)` reset.
- `MAACWrapper.step` updated to return 5-tuple.

### 🐛 Fixed — `epcontrol/compartments/contacts/Eames2012.py`
- `compute_beta`: replaced `np.amax(LA.eigvals(cm))` with `np.max(np.abs(LA.eigvals(cm)))` to use the dominant eigenvalue by **magnitude**, avoiding sign-sensitivity issues.

### ♻️ Changed — scripts (stable-baselines3 port)
- `scripts/seir_environment_single_sb_ppo.py`: rewritten for `stable-baselines3.PPO`; TF session config removed; default `learning_rate` 1e-4 → 3e-4; default hidden layers 0 → 2, hidden units 0 → 64; `nminibatches` → `batch_size = n_steps // n_minibatches`.
- `scripts/seir_environment_single_run_ppo_policy.py`: rewritten for `stable-baselines3`; `gym.envs.registration` boilerplate removed; `reset` / `step` calls updated to gymnasium API.
- `scripts/seir_environment_multi_sb_ppo.py`: same stable-baselines3 port as the single-district variant; updated hyperparameter defaults.
- `scripts/seir_environment_joint_run_ppo_policy.py`: rewritten for `stable-baselines3`; `districts_susceptibles` simplified to a generator expression; gymnasium API calls updated.

### ⚰️ Removed from active scripts
- `scripts/seir_pymarl.py` → moved to `legacy/`.
- `scripts/seir_environment_multi_run_ppo_policy.py` → moved to `legacy/`.
- `epcontrol/multiagent/seir.py` → moved to `legacy/multiagent_seir.py`.

### 📝 Changed — `README.md`
- Added modernisation header explaining the new stack and install instructions.

---

## [1.0.0] — 2020 (original release)

Initial publication accompanying Libin et al., _Deep reinforcement learning for large-scale epidemic control_, ECML 2020.
