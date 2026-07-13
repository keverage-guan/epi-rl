# Deep reinforcement learning for large-scale epidemic control
# Copyright (C) 2020  Pieter Libin, Arno Moonens, Fabian Perez-Sanjines.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to pieter.libin@ai.vub.ac.be or arno.moonens@vub.be.

from typing import Callable, Optional

import gymnasium
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def probe_returns(env_factory: Callable[[], gymnasium.Env],
                   policy,
                   n_probe_episodes: int,
                   deterministic: bool = True) -> np.ndarray:
    # total episode return of policy over n_probe_episodes independent stochastic rollouts
    returns = np.empty(n_probe_episodes)
    for i in range(n_probe_episodes):
        env = env_factory()
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        env.close()
        returns[i] = episode_return
    return returns


class RiskShapingCallback(BaseCallback):
    """Re-estimates nu (VaR_alpha of probe returns) each rollout and pushes it to the
    training envs."""

    def __init__(self,
                 env_factory: Callable[[], gymnasium.Env],
                 gamma: float,
                 alpha: float = 0.2,
                 n_probe_episodes: int = 8,
                 beta: float = 0.0,
                 epistemic_std_estimator: Optional[Callable[[], float]] = None,
                 verbose: int = 0):
        super(RiskShapingCallback, self).__init__(verbose)
        self.env_factory = env_factory
        self.gamma = gamma
        self.alpha = alpha
        self.n_probe_episodes = n_probe_episodes
        self.beta = beta
        self.epistemic_std_estimator = epistemic_std_estimator

    def _on_rollout_start(self) -> None:
        returns = probe_returns(self.env_factory, self.model, self.n_probe_episodes)
        nu = float(np.quantile(returns, self.alpha))
        epistemic_std = self.epistemic_std_estimator() if self.epistemic_std_estimator else 0.0
        self.training_env.env_method("set_risk_params", self.gamma, nu, self.beta * epistemic_std)
        self.logger.record("risk/nu_var_alpha", nu)
        self.logger.record("risk/probe_return_mean", float(returns.mean()))
        self.logger.record("risk/probe_return_std", float(returns.std()))
        self.logger.record("risk/epistemic_std", epistemic_std)

    def _on_step(self) -> bool:
        return True


class AlphaAnnealingCallback(BaseCallback):
    """Anneals a CVaR-style wrapper's alpha from alpha_start to alpha_end over the first
    anneal_fraction of training, then holds at alpha_end."""

    def __init__(self, total_timesteps: int, alpha_start: float = 1.0, alpha_end: float = 0.2,
                 anneal_fraction: float = 0.5, verbose: int = 0):
        super(AlphaAnnealingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.anneal_fraction = anneal_fraction

    def _on_rollout_start(self) -> None:
        progress = min(1.0, self.num_timesteps / (self.total_timesteps * self.anneal_fraction))
        alpha = self.alpha_start + progress * (self.alpha_end - self.alpha_start)
        self.training_env.env_method("set_alpha", alpha)
        self.logger.record("cvar/alpha", alpha)

    def _on_step(self) -> bool:
        return True
