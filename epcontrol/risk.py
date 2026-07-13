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

from stable_baselines3.common.callbacks import BaseCallback


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
