import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import epcontrol.census.Flux as flux
from epcontrol.progress import ProgressLoggingCallback
from epcontrol.risk import RiskShapingCallback
from epcontrol.seir_environment import Granularity, Outcome, SEIREnvironment
from epcontrol.UK_SEIR_Eames import UK
from epcontrol.wrappers import NormalizedObservationWrapper, NormalizedRewardWrapper, RiskShapedReward

# districts with similar total population, so swapping district each episode stays
# within the observation/reward bounds sized from a single reference district
TRAIN_DISTRICTS = ["Folkestone and Hythe", "South Staffordshire", "Scarborough", "South Ribble",
                   "Mendip", "Broxtowe", "Taunton Deane", "Welwyn Hatfield", "West Lancashire",
                   "St Edmundsbury", "Rushcliffe", "Fareham", "Dover", "Erewash", "Stroud",
                   "South Ayrshire", "Bassetlaw", "Bracknell Forest", "Gedling", "East Staffordshire"]
HOLDOUT_DISTRICTS = ["Chichester", "Scottish Borders", "Sedgemoor", "Newark and Sherwood",
                     "Sevenoaks", "Tunbridge Wells", "Conwy", "Waveney", "East Hampshire",
                     "Cheltenham"]

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--budget_in_weeks", type=int, required=True)
parser.add_argument("--census", type=Path, required=True)
parser.add_argument("--R0", type=float, required=True)
parser.add_argument("--monitor_path", type=str,
                    default=f"/tmp/SEIR-EnsemblePPO-{datetime.datetime.now():%Y-%m-%d-%H-%M-%S-%f}/")
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--n_hidden_layers", type=int, default=2)
parser.add_argument("--n_hidden_units", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--n_epochs", type=int, default=4)
parser.add_argument("--n_minibatches", type=int, default=4)
parser.add_argument("--n_steps", type=int, default=128)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--clip_range", type=float, default=0.2)
parser.add_argument("--total_timesteps", type=int, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gamma_risk", type=float, default=0.0,
                    help="0 = plain mean-reward training over the district ensemble. "
                         ">0 = CVaR risk shaping over the ensemble's episode returns.")
parser.add_argument("--alpha", type=float, default=0.2,
                    help="CVaR tail fraction, only used when gamma_risk>0.")
parser.add_argument("--n_probe_episodes", type=int, default=20,
                    help="Episodes used to estimate nu before each rollout, only used when gamma_risk>0.")
args = parser.parse_args()

N_WEEKS = 43
GRANULARITY = Granularity.WEEK
os.makedirs(args.monitor_path, exist_ok=True)
OUTCOME = Outcome.ATTACK_RATE
RHO = 1.0
GAMMA = 1 / 1.8
DELTA = 0.5

full_census = pd.read_csv(args.census, index_col=0)

def make_model(district_name: str) -> UK:
    grouped_census = full_census.filter(items=[district_name], axis=0)
    fl = flux.SingleDistrictStub(district_name)
    mu = np.log(args.R0) * .6
    return UK(DELTA, args.R0, RHO, GAMMA, [district_name], grouped_census, fl, mu, sde=True)

def ensemble_model_factory() -> UK:
    district_name = np.random.choice(TRAIN_DISTRICTS)
    return make_model(district_name)

# largest-population district across train+holdout, sizes the observation/reward bounds
reference_district = full_census.loc[TRAIN_DISTRICTS + HOLDOUT_DISTRICTS].sum(axis=1).idxmax()

def wrap_env(env):
    env = NormalizedObservationWrapper(env)
    env = NormalizedRewardWrapper(env)
    if args.gamma_risk > 0:
        env = RiskShapedReward(env, alpha=args.alpha)
    return env

def make_probe_env():
    env = SEIREnvironment(model=make_model(reference_district), model_factory=ensemble_model_factory,
                          n_weeks=N_WEEKS, step_granularity=GRANULARITY, outcome=OUTCOME,
                          model_seed=reference_district,
                          budget_per_district_in_weeks=args.budget_in_weeks)
    return wrap_env(env)

def make_env():
    env = SEIREnvironment(model=make_model(reference_district), model_factory=ensemble_model_factory,
                          n_weeks=N_WEEKS, step_granularity=GRANULARITY, outcome=OUTCOME,
                          model_seed=reference_district,
                          budget_per_district_in_weeks=args.budget_in_weeks)
    env = wrap_env(env)
    env = Monitor(env, filename=os.path.join(args.monitor_path, "monitor"))
    return env

venv = DummyVecEnv([make_env])
layers = [args.n_hidden_units] * args.n_hidden_layers
net_arch = layers if layers else None
batch_size = max(1, args.n_steps // args.n_minibatches)

model = PPO("MlpPolicy", venv, verbose=0, seed=args.seed, device="cpu",
            ent_coef=args.entropy_coef, learning_rate=args.learning_rate,
            n_epochs=args.n_epochs, batch_size=batch_size, n_steps=args.n_steps,
            clip_range=args.clip_range, max_grad_norm=args.max_grad_norm,
            policy_kwargs=dict(net_arch=net_arch) if net_arch is not None else None)

callbacks = [ProgressLoggingCallback(total_timesteps=args.total_timesteps)]
if args.gamma_risk > 0:
    callbacks.append(RiskShapingCallback(env_factory=make_probe_env, gamma=args.gamma_risk,
                                         alpha=args.alpha, n_probe_episodes=args.n_probe_episodes))
model.learn(total_timesteps=args.total_timesteps, progress_bar=False, callback=CallbackList(callbacks))
model.save(os.path.join(args.monitor_path, "params"))
venv.close()
