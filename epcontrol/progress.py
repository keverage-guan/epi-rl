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

import time

from stable_baselines3.common.callbacks import BaseCallback


class ProgressLoggingCallback(BaseCallback):
    # progress_bar=True uses a \r-updated tqdm bar, which does not flush when stdout is
    # redirected to a file; this prints discrete flushed lines instead
    def __init__(self, total_timesteps: int, log_every: int = 5000, verbose: int = 0):
        super(ProgressLoggingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_every = log_every
        self._next_log = log_every
        self._start_time = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        print(f"[progress] starting: 0/{self.total_timesteps} (0.0%)", flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_log:
            elapsed = time.time() - self._start_time
            pct = 100 * self.num_timesteps / self.total_timesteps
            rate = self.num_timesteps / elapsed if elapsed > 0 else 0.0
            eta = (self.total_timesteps - self.num_timesteps) / rate if rate > 0 else float("inf")
            print(f"[progress] {self.num_timesteps}/{self.total_timesteps} ({pct:.1f}%) "
                  f"elapsed={elapsed:.0f}s rate={rate:.1f}steps/s eta={eta:.0f}s", flush=True)
            self._next_log += self.log_every
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self._start_time
        print(f"[progress] done: {self.num_timesteps}/{self.total_timesteps} (100%) "
              f"elapsed={elapsed:.0f}s", flush=True)
