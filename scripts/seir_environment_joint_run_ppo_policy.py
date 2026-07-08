import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from tqdm import tqdm

import epcontrol.census.Flux as Flux
from epcontrol.seir_environment import Granularity, SEIREnvironment
from epcontrol.UK_RL_school_weekly import run_model
from epcontrol.wrappers import (MultiAgentSelectAction, MultiAgentSelectObservation,
                                NormalizedObservationWrapper, NormalizedRewardWrapper)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--district_name", required=True)
parser.add_argument("--budget_in_weeks", type=int, required=True)
parser.add_argument("--census", type=Path, required=True)
parser.add_argument("--flux", type=Path, required=True)
parser.add_argument("--R0", type=float, required=True)
parser.add_argument("--runs", type=int, required=True)
parser.add_argument("--path", type=Path, required=True)
args = parser.parse_args()

DISTRICTS_GROUP = ["Cornwall", "Plymouth", "Torbay", "East Devon", "Exeter", "Mid Devon",
                   "North Devon", "South Hams", "Teignbridge", "Torridge", "West Devon"]
N_WEEKS = 43
GRANULARITY = Granularity.WEEK

def districts_susceptibles(env, ids):
    return sum(env.unwrapped._model.total_susceptibles_district(i) for i in ids)

grouped_census = pd.read_csv(args.census, index_col=0)
base = SEIREnvironment(grouped_census=grouped_census, flux=Flux.Table(args.flux), r0=args.R0,
                       n_weeks=N_WEEKS, step_granularity=GRANULARITY, model_seed=args.district_name,
                       budget_per_district_in_weeks=args.budget_in_weeks)
ids = [base.district_idx(name) for name in DISTRICTS_GROUP]
env = NormalizedObservationWrapper(base)
env = NormalizedRewardWrapper(env)
env = MultiAgentSelectObservation(env, ids)
env = MultiAgentSelectAction(env, ids, 1)

def evaluate(env, model, ids, num_steps):
    obs, _ = env.reset()
    sus_before = districts_susceptibles(env, ids)
    for _ in tqdm(range(num_steps), desc="  steps", unit="week", leave=False):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = env.step(action)
    sus_after = districts_susceptibles(env, ids)
    return 1.0 - (sus_after / sus_before)

no_closures = [1] * N_WEEKS
(_, baseline_ar, _) = run_model(env.unwrapped._model, N_WEEKS, False, args.district_name, no_closures)

model = PPO.load(str(args.path / "params"))
print("ar-improvement")
for _ in tqdm(range(args.runs), desc="evaluation runs", unit="run"):
    tqdm.write(str(baseline_ar - evaluate(env, model, ids, N_WEEKS)))
env.close()
