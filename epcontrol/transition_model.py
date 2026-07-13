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

from typing import List, Protocol, Sequence, runtime_checkable
import numpy as np


@runtime_checkable
class TransitionModel(Protocol):
    """Contract that SEIREnvironment requires of its simulation backend."""

    # names of the districts/patches, same order as the district axis of seir_state
    district_names: List[str]

    # shape (n_districts, n_compartments, n_age_groups)
    seir_state: np.ndarray

    def reset(self) -> None:
        ...

    def seed(self, region: str) -> None:
        ...

    def step(self, t: int, actions: Sequence[int]) -> None:
        ...

    def total_infected(self) -> float:
        ...

    def total_susceptibles(self) -> float:
        ...

    def total_susceptibles_district(self, district_idx: int) -> float:
        ...

    def district_idx(self, district_name: str) -> int:
        ...

    def peak_day(self, infected_history: np.ndarray) -> int:
        ...
