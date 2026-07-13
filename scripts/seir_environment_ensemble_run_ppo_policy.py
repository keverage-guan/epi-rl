import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from tqdm import tqdm

import epcontrol.census.Flux as flux
from epcontrol.seir_environment import Granularity, Outcome, SEIREnvironment
from epcontrol.UK_RL_school_weekly import run_model
from epcontrol.UK_SEIR_Eames import UK
from epcontrol.wrappers import NormalizedObservationWrapper, NormalizedRewardWrapper

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--budget_in_weeks", type=int, required=True)
parser.add_argument("--census", type=Path, required=True)
parser.add_argument("--R0", type=float, required=True)
parser.add_argument("--runs_per_district", type=int, required=True)
parser.add_argument("--path", type=Path, required=True)
parser.add_argument("--districts", nargs="+", required=True,
                    help="Districts to evaluate on, e.g. the held-out (never trained on) pool.")
args = parser.parse_args()

N_WEEKS = 43
GRANULARITY = Granularity.WEEK
OUTCOME = Outcome.ATTACK_RATE
RHO = 1.0
GAMMA = 1 / 1.8
DELTA = 0.5

full_census = pd.read_csv(args.census, index_col=0)

def make_model(district_name: str, sde: bool) -> UK:
    grouped_census = full_census.filter(items=[district_name], axis=0)
    fl = flux.SingleDistrictStub(district_name)
    mu = np.log(args.R0) * .6
    return UK(DELTA, args.R0, RHO, GAMMA, [district_name], grouped_census, fl, mu, sde=sde)

model = PPO.load(str(args.path / "params"))
# PPO.load() reseeds numpy's global RNG to the training seed; reseed from OS entropy
# so the rollouts below are independent
np.random.seed(None)

print("district,ar-improvement")
for district_name in tqdm(args.districts, desc="districts", unit="district"):
    env_model = make_model(district_name, sde=True)
    env = SEIREnvironment(model=env_model, n_weeks=N_WEEKS, step_granularity=GRANULARITY,
                          outcome=OUTCOME, model_seed=district_name,
                          budget_per_district_in_weeks=args.budget_in_weeks)
    env = NormalizedObservationWrapper(env)
    env = NormalizedRewardWrapper(env)

    baseline_model = make_model(district_name, sde=False)
    no_closures = [1] * N_WEEKS
    (_, baseline_ar, _) = run_model(baseline_model, N_WEEKS, False, district_name, no_closures)

    for _ in tqdm(range(args.runs_per_district), desc=district_name, unit="run", leave=False):
        obs, _ = env.reset()
        sus_before = env.unwrapped._model.total_susceptibles()
        for _ in range(N_WEEKS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
        sus_after = env.unwrapped._model.total_susceptibles()
        attack_rate = 1.0 - (sus_after / sus_before)
        tqdm.write(f"{district_name},{baseline_ar - attack_rate}")
    env.close()
