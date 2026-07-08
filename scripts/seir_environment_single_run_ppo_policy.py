import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import epcontrol.census.Flux as flux
from epcontrol.seir_environment import Granularity, Outcome, SEIREnvironment
from epcontrol.UK_RL_school_weekly import run_model
from epcontrol.UK_SEIR_Eames import UK
from epcontrol.wrappers import NormalizedObservationWrapper, NormalizedRewardWrapper

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--district_name", required=True)
parser.add_argument("--budget_in_weeks", type=int, required=True)
parser.add_argument("--census", type=Path, required=True)
parser.add_argument("--R0", type=float, required=True)
parser.add_argument("--runs", type=int, required=True)
parser.add_argument("--path", type=Path, required=True)
parser.add_argument("--outcome", choices=["ar", "pd"], required=True)
args = parser.parse_args()

N_WEEKS = 43
GRANULARITY = Granularity.WEEK
OUTCOME = Outcome.ATTACK_RATE if args.outcome == "ar" else Outcome.PEAK_DAY

def evaluate(env, model, num_steps):
    _model = env.unwrapped._model
    obs, _ = env.reset()
    sus_before = _model.total_susceptibles()
    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = env.step(action)
    sus_after = _model.total_susceptibles()
    attack_rate = 1.0 - (sus_after / sus_before)
    peak_day = _model.peak_day(env.unwrapped.infected_history)
    return attack_rate, peak_day, env.unwrapped.infected_history

grouped_census = pd.read_csv(args.census, index_col=0).filter(items=[args.district_name], axis=0)
fl = flux.SingleDistrictStub(args.district_name)

env = SEIREnvironment(grouped_census=grouped_census, flux=fl, r0=args.R0, n_weeks=N_WEEKS * 2,
                      step_granularity=GRANULARITY, outcome=OUTCOME,
                      model_seed=args.district_name, budget_per_district_in_weeks=args.budget_in_weeks)
env = NormalizedObservationWrapper(env)
env = NormalizedRewardWrapper(env)

no_closures = [1] * N_WEEKS
district_names = grouped_census.index.to_list()
mu = np.log(args.R0) * .6
baseline_model = UK(.5, args.R0, 1, (1 / 1.8), district_names, grouped_census, fl, mu, sde=False)
(baseline_pd, baseline_ar, _) = run_model(baseline_model, N_WEEKS, False, args.district_name, no_closures)

model = PPO.load(str(args.path / "params"))
print(args.outcome + "-improvement")
for _ in range(args.runs):
    attack_rate, peak_day, _ = evaluate(env, model, N_WEEKS)
    if args.outcome == "ar":
        print(baseline_ar - attack_rate)
    else:
        print(peak_day - baseline_pd)
env.close()
