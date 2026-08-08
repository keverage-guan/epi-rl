from typing import List, Sequence

import numpy as np
import torch

class PINNTransitionModel:

    def __init__(self, trainer, data, deterministic: bool = True):
        self.trainer = trainer
        self.data = data
        self.district_names: List[str] = list(data.district_names)
        self.n_districts = data.n_patches
        self.n_age_groups = trainer.mcfg.n_age_groups
        self.n_weeks = trainer.mcfg.n_weeks
        self.deterministic = deterministic
        self._name_to_idx = {name: i for i, name in enumerate(self.district_names)}
        self.seir_state = np.zeros((self.n_districts, 5, self.n_age_groups), dtype=np.float32)
        self._schedule = np.ones((self.n_weeks, self.n_districts), dtype=np.float32)
        self._query(week=0, tau=0.0)

    def reset(self) -> None:
        self._schedule[:] = 1.0
        self._query(week=0, tau=0.0)

    def seed(self, region: str) -> None:
        pass

    def step(self, t: int, actions: Sequence[int]) -> None:
        week = min(t // 7, self.n_weeks - 1)
        tau = ((t % 7) + 0.5) / 7.0
        self._schedule[week] = np.asarray(actions, dtype=np.float32)
        self._query(week=week, tau=tau)

    def _query(self, week: int, tau: float) -> None:
        net = self.trainer.net
        device = self.trainer.device
        dtype = self.trainer.dtype
        tau_t = torch.full((1, 1), float(tau), dtype=dtype, device=device)
        week_t = torch.tensor([week], dtype=torch.int64, device=device)
        closure_t = torch.as_tensor(
            self._schedule[week], dtype=dtype, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            if self.deterministic:
                state = net(tau_t, week_t, closure_t)
            else:
                with net.mc_dropout():
                    state = net(tau_t, week_t, closure_t)
        self.seir_state = state.squeeze(0).cpu().numpy()

    def total_infected(self) -> float:
        return float(self.seir_state[:, 2, :].sum() + self.seir_state[:, 3, :].sum())

    def total_susceptibles(self) -> float:
        return float(self.seir_state[:, 0, :].sum())

    def total_susceptibles_district(self, district_idx: int) -> float:
        return float(self.seir_state[district_idx, 0, :].sum())

    def district_idx(self, district_name: str) -> int:
        return self._name_to_idx[district_name]

    def peak_day(self, infected_history: np.ndarray) -> int:
        return int(np.argmax(infected_history))
