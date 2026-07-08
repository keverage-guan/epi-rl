import argparse
import datetime
import os
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import epcontrol.census.Flux as flux
from epcontrol.seir_environment import Granularity, Outcome, SEIREnvironment
from epcontrol.wrappers import NormalizedObservationWrapper, NormalizedRewardWrapper

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--outcome", choices=["ar", "pd"], required=True)
parser.add_argument("--district_name", required=True)
parser.add_argument("--budget_in_weeks", type=int, required=True)
parser.add_argument("--census", type=Path, required=True)
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
args = parser.parse_args()

N_WEEKS = 43
GRANULARITY = Granularity.WEEK
os.makedirs(args.monitor_path, exist_ok=True)
OUTCOME = Outcome.ATTACK_RATE if args.outcome == "ar" else Outcome.PEAK_DAY

def make_env():
    grouped_census = pd.read_csv(args.census, index_col=0).filter(items=[args.district_name], axis=0)
    fl = flux.SingleDistrictStub(args.district_name)
    env = SEIREnvironment(grouped_census=grouped_census, flux=fl, r0=args.R0, n_weeks=N_WEEKS,
                          step_granularity=GRANULARITY, outcome=OUTCOME,
                          model_seed=args.district_name, budget_per_district_in_weeks=args.budget_in_weeks)
    env = NormalizedObservationWrapper(env)
    if args.outcome == "ar":
        env = NormalizedRewardWrapper(env)
    env = Monitor(env, filename=os.path.join(args.monitor_path, "monitor"))
    return env

venv = DummyVecEnv([make_env])
layers = [args.n_hidden_units] * args.n_hidden_layers
net_arch = layers if layers else None
batch_size = max(1, args.n_steps // args.n_minibatches)

model = PPO("MlpPolicy", venv, verbose=0, seed=args.seed,
            ent_coef=args.entropy_coef, learning_rate=args.learning_rate,
            n_epochs=args.n_epochs, batch_size=batch_size, n_steps=args.n_steps,
            clip_range=args.clip_range, max_grad_norm=args.max_grad_norm,
            policy_kwargs=dict(net_arch=net_arch) if net_arch is not None else None)
model.learn(total_timesteps=args.total_timesteps)
model.save(os.path.join(args.monitor_path, "params"))
venv.close()
