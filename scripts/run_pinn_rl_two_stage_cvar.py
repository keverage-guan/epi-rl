import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from epcontrol.pinn_transition_model import PINNTransitionModel
from epcontrol.seir_environment import Granularity, Outcome, SEIREnvironment
from epcontrol.wrappers import (MultiAgentSelectAction, MultiAgentSelectObservation,
                                 MultiAgentSelectReward, NormalizedObservationWrapper,
                                 NormalizedRewardWrapper)
from pinn_seir.config import ModelConfig, TrainConfig
from pinn_seir.data import load_epi_data
from pinn_seir.model import PINNTrainer

CKPT = "outputs/seir_pinn/trillium_fixed_run/checkpoint_iter020000.pt"
OUT_DIR = Path("outputs/pinn_rl_two_stage_cvar_trillium_iter020000")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch device = {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'})")

N_WEEKS = 43
GRANULARITY = Granularity.WEEK
BUDGET_WEEKS = 6
TRAIN_EPISODES = 250
TOTAL_TIMESTEPS = N_WEEKS * TRAIN_EPISODES
N_STEPS = N_WEEKS * 3
PPO_SEEDS = [0, 1, 2, 3, 4]
N_STOCHASTIC_ROLLOUTS = 100

CONTROLLED_DISTRICTS = [
    "Birmingham", "Leeds", "Sheffield", "Cornwall", "Bradford",
    "Glasgow City", "City of Edinburgh", "Fife", "North Lanarkshire", "South Lanarkshire",
    "Cardiff", "Swansea", "Rhondda Cynon Taf", "Carmarthenshire", "Caerphilly",
]
N_CONTROLLED = len(CONTROLLED_DISTRICTS)

print("Loading epidemiological data (Kevin's own data/great_brittain census+commute+contacts)...")
mcfg = ModelConfig(flu_path=Path("data/epidemic/uk_flu_per_100000.csv"))
tcfg = TrainConfig(dropout_rate=0.1, device=DEVICE)
data = load_epi_data(mcfg)
print(f"  {data.n_patches} districts -> nations {data.nation_names}")

trainer = PINNTrainer(data, mcfg, tcfg)
trainer.load_checkpoint(CKPT)
print(f"Loaded Kevin's checkpoint {CKPT}")
print(f"  R0={trainer.r0.item():.4f} mu={trainer.mu.item():.4f} "
      f"kappa={trainer.kappa.item():.4f} alpha={trainer.alpha.item():.4f}")

pinn_train = PINNTransitionModel(trainer, data, deterministic=True)
controlled_ids = [pinn_train.district_idx(n) for n in CONTROLLED_DISTRICTS]

def make_env(pinn_model, monitor_name=None):
    env = SEIREnvironment(model=pinn_model, n_weeks=N_WEEKS, step_granularity=GRANULARITY,
                           outcome=Outcome.ATTACK_RATE, budget_per_district_in_weeks=BUDGET_WEEKS,
                           model_seed=mcfg.seed_district)
    ids = [env.district_idx(name) for name in CONTROLLED_DISTRICTS]
    env = NormalizedObservationWrapper(env)
    env = NormalizedRewardWrapper(env)
    env = MultiAgentSelectObservation(env, ids)
    env = MultiAgentSelectAction(env, ids, other_agents_action=1)
    env = MultiAgentSelectReward(env, ids)
    if monitor_name is not None:
        env = Monitor(env, filename=str(OUT_DIR / monitor_name))
    return env

def enforce_budget(requested: np.ndarray, budget_weeks: int) -> np.ndarray:
    n_weeks, n_districts = requested.shape
    enforced = requested.copy()
    budget = np.full(n_districts, budget_weeks, dtype=np.float32)
    for w in range(n_weeks):
        budget -= (requested[w] == 0)
        enforced[w, budget < 0] = 1
    return enforced

print(f"\n=== Stage 1a: training {len(PPO_SEEDS)} unmodified-PPO candidates "
      f"({TRAIN_EPISODES} episodes each, plain mean reward, no risk shaping) ===")
ppo_models = {}
for seed in PPO_SEEDS:
    t0 = time.time()
    env = make_env(pinn_train, f"monitor_train_seed{seed}")
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
      f"stochastic (MC-dropout) rollouts ===")

def full_district_schedule(controlled_schedule):
    full = np.ones((N_WEEKS, data.n_patches), dtype=np.float32)
    for j, idx in enumerate(controlled_ids):
        full[:, idx] = controlled_schedule[:, j]
    return full

def rollout_attack_rate(full_schedule, stoch_model) -> float:
    stoch_model.reset()
    s0 = stoch_model.total_susceptibles()
    for t in range(N_WEEKS * 7):
        week = t // 7
        stoch_model.step(t, full_schedule[week])
    s1 = stoch_model.total_susceptibles()
    return 1.0 - (s1 / s0)

def evaluate_ppo_schedule(ppo, det_model) -> np.ndarray:
    env = make_env(det_model)
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
    candidate_schedules[name] = evaluate_ppo_schedule(ppo, pinn_train)
candidate_schedules.update(heuristic_schedules)

stoch_model = PINNTransitionModel(trainer, data, deterministic=False)

results = {}
t0 = time.time()
for name, sched in candidate_schedules.items():
    full_sched = full_district_schedule(sched)
    rates = np.array([rollout_attack_rate(full_sched, stoch_model)
                       for _ in range(N_STOCHASTIC_ROLLOUTS)])
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
    print(f"  {name:20s} mean_AR={mean_ar:.6e}  CVaR10%={cvar10:.6e}  "
          f"std={rates.std():.2e}   ({time.time() - t0:.0f}s elapsed)")

selected = min(results, key=lambda k: results[k]["cvar10_attack_rate"])
best_mean = min(results, key=lambda k: results[k]["mean_attack_rate"])
print(f"\nSelected by CVaR10%: {selected}  (CVaR10%={results[selected]['cvar10_attack_rate']:.6e})")
print(f"Best by mean attack rate: {best_mean}  (mean={results[best_mean]['mean_attack_rate']:.6e})")
print(f"Same candidate: {selected == best_mean}")

out = {
    "device": DEVICE,
    "checkpoint": CKPT,
    "pinn_params": {"R0": trainer.r0.item(), "mu": trainer.mu.item(),
                     "kappa": trainer.kappa.item(), "alpha": trainer.alpha.item()},
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
