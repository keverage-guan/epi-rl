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

import epcontrol.census.Flux as Flux
from epcontrol.progress import ProgressLoggingCallback
from epcontrol.risk import AlphaAnnealingCallback
from epcontrol.seir_environment import Granularity, SEIREnvironment
from epcontrol.UK_SEIR_Eames import UK
from epcontrol.wrappers import (MultiAgentCVaRReward, MultiAgentSelectAction,
                                MultiAgentSelectObservation, MultiAgentSelectReward,
                                NormalizedObservationWrapper, NormalizedRewardWrapper)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--district_name", required=True)
parser.add_argument("--budget_in_weeks", type=int, required=True)
parser.add_argument("--census", type=Path, required=True)
parser.add_argument("--flux", type=Path, required=True)
parser.add_argument("--R0", type=float, required=True)
parser.add_argument("--monitor_path", type=str,
                    default=f"/tmp/SEIR-PPO-{datetime.datetime.now():%Y-%m-%d-%H-%M-%S-%f}/")
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
parser.add_argument("--reward_mode", choices=["mean", "cvar"], default="mean",
                    help="'mean' rewards the aggregate outcome across districts. 'cvar' rewards "
                         "the mean outcome of the worst cvar_alpha fraction of districts.")
parser.add_argument("--cvar_alpha", type=float, default=0.2,
                    help="Fraction of worst-performing districts averaged for the cvar reward_mode.")
parser.add_argument("--cvar_alpha_start", type=float, default=1.0,
                    help="cvar reward_mode: alpha at the start of training (1.0 = full aggregate).")
parser.add_argument("--cvar_anneal_fraction", type=float, default=0.5,
                    help="cvar reward_mode: fraction of total_timesteps to anneal alpha over.")
args = parser.parse_args()

N_WEEKS = 43
GRANULARITY = Granularity.WEEK
os.makedirs(args.monitor_path, exist_ok=True)
DISTRICTS_GROUP = ["Cornwall", "Plymouth", "Torbay", "East Devon", "Exeter", "Mid Devon",
                   "North Devon", "South Hams", "Teignbridge", "Torridge", "West Devon"]
RHO = 1.0
GAMMA = 1 / 1.8
DELTA = 0.5

def make_env():
    grouped_census = pd.read_csv(args.census, index_col=0)
    fl = Flux.Table(args.flux)
    district_names = grouped_census.index.to_list()
    mu = np.log(args.R0) * .6
    model = UK(DELTA, args.R0, RHO, GAMMA, district_names, grouped_census, fl, mu, sde=True)
    env = SEIREnvironment(model=model, n_weeks=N_WEEKS, step_granularity=GRANULARITY,
                          model_seed=args.district_name, budget_per_district_in_weeks=args.budget_in_weeks)
    ids = [env.district_idx(name) for name in DISTRICTS_GROUP]
    env = NormalizedObservationWrapper(env)
    env = NormalizedRewardWrapper(env)
    env = MultiAgentSelectObservation(env, ids)
    env = MultiAgentSelectAction(env, ids, 1)
    if args.reward_mode == "cvar":
        env = MultiAgentCVaRReward(env, ids, alpha=args.cvar_alpha_start)
    else:
        env = MultiAgentSelectReward(env, ids)
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
if args.reward_mode == "cvar":
    callbacks.append(AlphaAnnealingCallback(total_timesteps=args.total_timesteps,
                                            alpha_start=args.cvar_alpha_start,
                                            alpha_end=args.cvar_alpha,
                                            anneal_fraction=args.cvar_anneal_fraction))
model.learn(total_timesteps=args.total_timesteps, progress_bar=False, callback=CallbackList(callbacks))
model.save(os.path.join(args.monitor_path, "params"))
venv.close()
