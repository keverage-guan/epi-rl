import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import epcontrol.census.Flux as Flux
from epcontrol.UK_SEIR_Eames import UK
from epcontrol.seir_environment import Granularity, Outcome, SEIREnvironment
from epcontrol.wrappers import (MultiAgentSelectAction, MultiAgentSelectObservation,
                                 MultiAgentSelectReward, NormalizedObservationWrapper,
                                 NormalizedRewardWrapper)
from pinn_seir.config import ModelConfig
from pinn_seir.data import load_epi_data

OUT_DIR = Path("outputs/uk_rl_two_stage_cvar")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch device = {DEVICE}")

N_WEEKS = 43
GRANULARITY = Granularity.WEEK
BUDGET_WEEKS = 6
TRAIN_EPISODES = 250
TOTAL_TIMESTEPS = N_WEEKS * TRAIN_EPISODES
N_STEPS = N_WEEKS * 3
PPO_SEEDS = [0, 1, 2, 3, 4]
N_STOCHASTIC_ROLLOUTS = 100
R0 = 1.8
GAMMA = 1 / 1.8
RHO = 1.0
DELTA = 0.5
SEED_DISTRICT = "Falkirk"

CONTROLLED_DISTRICTS = [
    "Birmingham", "Leeds", "Sheffield", "Cornwall", "Bradford",
    "Glasgow City", "City of Edinburgh", "Fife", "North Lanarkshire", "South Lanarkshire",
    "Cardiff", "Swansea", "Rhondda Cynon Taf", "Carmarthenshire", "Caerphilly",
]
N_CONTROLLED = len(CONTROLLED_DISTRICTS)

print("Loading the same 378-district population / ordering the PINN used...")
mcfg = ModelConfig(flu_path=Path("data/epidemic/uk_flu_per_100000.csv"))
data = load_epi_data(mcfg)
district_names = data.district_names
print(f"  {len(district_names)} districts")

grouped_census = pd.read_csv("data/great_brittain/census.csv", index_col=0).loc[district_names]
fl = Flux.Table("data/great_brittain/commute.csv")
mu = np.log(R0) * 0.6

def make_model():
    return UK(DELTA, R0, RHO, GAMMA, district_names, grouped_census, fl, mu, sde=True)

def make_env(monitor_name=None):
    model = make_model()
    env = SEIREnvironment(model=model, n_weeks=N_WEEKS, step_granularity=GRANULARITY,
                           outcome=Outcome.ATTACK_RATE, budget_per_district_in_weeks=BUDGET_WEEKS,
                           model_seed=SEED_DISTRICT)
    ids = [env.district_idx(name) for name in CONTROLLED_DISTRICTS]
    env = NormalizedObservationWrapper(env)
    env = NormalizedRewardWrapper(env)
    env = MultiAgentSelectObservation(env, ids)
    env = MultiAgentSelectAction(env, ids, other_agents_action=1)
    env = MultiAgentSelectReward(env, ids)
    if monitor_name is not None:
        env = Monitor(env, filename=str(OUT_DIR / monitor_name))
    return env

controlled_ids = [district_names.index(n) for n in CONTROLLED_DISTRICTS]

def enforce_budget(requested: np.ndarray, budget_weeks: int) -> np.ndarray:
    n_weeks, n_districts = requested.shape
    enforced = requested.copy()
    budget = np.full(n_districts, budget_weeks, dtype=np.float32)
    for w in range(n_weeks):
        budget -= (requested[w] == 0)
        enforced[w, budget < 0] = 1
    return enforced

print(f"\n=== Stage 1a: training {len(PPO_SEEDS)} unmodified-PPO candidates "
      f"({TRAIN_EPISODES} episodes each) against the UK SDE simulator ===")
ppo_models = {}
t_all = time.time()
for seed in PPO_SEEDS:
    t0 = time.time()
    env = make_env(f"monitor_train_seed{seed}")
    venv = DummyVecEnv([lambda env=env: env])
    ppo = PPO("MlpPolicy", venv, verbose=0, seed=seed, device=DEVICE,
              ent_coef=0.01, learning_rate=3e-4, n_epochs=4,
              batch_size=N_STEPS // 4, n_steps=N_STEPS, clip_range=0.2,
              max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64]))
    ppo.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    venv.close()
    ppo.save(str(OUT_DIR / f"ppo_seed{seed}"))
    ppo_models[f"ppo_seed{seed}"] = ppo
    print(f"  seed {seed} done in {time.time() - t0:.0f}s")
print(f"  total stage 1a: {time.time() - t_all:.0f}s")

def schedule_block(start_week, length=BUDGET_WEEKS):
    s = np.ones((N_WEEKS, N_CONTROLLED), dtype=np.float32)
    s[start_week:start_week + length, :] = 0.0
    return s

def schedule_spread(stride=7, n=BUDGET_WEEKS):
    s = np.ones((N_WEEKS, N_CONTROLLED), dtype=np.float32)
    weeks = [min(i * stride, N_WEEKS - 1) for i in range(n)]
    s[weeks, :] = 0.0
    return s

heuristic_schedules = {
    "heur_no_closures": np.ones((N_WEEKS, N_CONTROLLED), dtype=np.float32),
    "heur_early_lockdown": schedule_block(0),
    "heur_mid_lockdown": schedule_block(18),
    "heur_late_lockdown": schedule_block(36),
    "heur_spread_out": schedule_spread(),
}

print(f"\n=== Stage 2: evaluating every candidate over {N_STOCHASTIC_ROLLOUTS} "
      f"stochastic (SDE) rollouts ===")

def full_district_schedule(controlled_schedule):
    full = np.ones((N_WEEKS, len(district_names)), dtype=np.float32)
    for j, idx in enumerate(controlled_ids):
        full[:, idx] = controlled_schedule[:, j]
    return full

def rollout_attack_rate(full_schedule) -> float:
    model = make_model()
    model.seed(SEED_DISTRICT)
    s0 = model.total_susceptibles()
    for t in range(N_WEEKS * 7):
        week = t // 7
        model.step(t, full_schedule[week])
    s1 = model.total_susceptibles()
    return 1.0 - (s1 / s0)

def evaluate_ppo_schedule(ppo) -> np.ndarray:
    env = make_env()
    obs, _ = env.reset()
    requested = np.zeros((N_WEEKS, N_CONTROLLED), dtype=np.float32)
    for week in range(N_WEEKS):
        action, _ = ppo.predict(obs, deterministic=True)
        requested[week] = action
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    return enforce_budget(requested, BUDGET_WEEKS)

candidate_schedules = {}
for name, ppo in ppo_models.items():
    candidate_schedules[name] = evaluate_ppo_schedule(ppo)
candidate_schedules.update(heuristic_schedules)

results = {}
t0 = time.time()
for name, sched in candidate_schedules.items():
    full_sched = full_district_schedule(sched)
    rates = np.array([rollout_attack_rate(full_sched) for _ in range(N_STOCHASTIC_ROLLOUTS)])
    k = int(np.ceil(0.1 * N_STOCHASTIC_ROLLOUTS))
    worst_decile = np.sort(rates)[-k:]
    cvar10 = float(worst_decile.mean())
    mean_ar = float(rates.mean())
    results[name] = {
        "mean_attack_rate": mean_ar,
        "cvar10_attack_rate": cvar10,
        "std_attack_rate": float(rates.std()),
        "rollout_attack_rates": rates.tolist(),
        "closed_weeks_per_district": {
            CONTROLLED_DISTRICTS[j]: int((sched[:, j] == 0).sum())
            for j in range(N_CONTROLLED)
        },
    }
    print(f"  {name:20s} mean_AR={mean_ar:.6f}  CVaR10%={cvar10:.6f}  "
          f"std={rates.std():.4f}   ({time.time() - t0:.0f}s elapsed)")

selected = min(results, key=lambda k: results[k]["cvar10_attack_rate"])
best_mean = min(results, key=lambda k: results[k]["mean_attack_rate"])
print(f"\nSelected by CVaR10%: {selected}  (CVaR10%={results[selected]['cvar10_attack_rate']:.6f})")
print(f"Best by mean attack rate: {best_mean}  (mean={results[best_mean]['mean_attack_rate']:.6f})")
print(f"Same candidate: {selected == best_mean}")

out = {
    "device": DEVICE,
    "model": "UK (Libin et al. 2020), R0=1.8, SDE",
    "n_stochastic_rollouts": N_STOCHASTIC_ROLLOUTS,
    "controlled_districts": CONTROLLED_DISTRICTS,
    "selected_by_cvar10": selected,
    "best_by_mean": best_mean,
    "candidates": results,
}
with open(OUT_DIR / "results.json", "w") as fh:
    json.dump(out, fh, indent=2)

np.savez(OUT_DIR / "schedules.npz",
          **{name: sched for name, sched in candidate_schedules.items()},
          controlled_districts=np.array(CONTROLLED_DISTRICTS))

print(f"\nSaved -> {OUT_DIR}")
