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
    """Contract that SEIREnvironment requires of its simulation backend.

    epcontrol.UK_SEIR_Eames.UK (age-structured SEIR with Euler-Maruyama SDE sampling)
    is the current implementation. Any other simulator, e.g. a PINN trained with MC
    dropout in place of SDE sampling, can be substituted as a drop-in replacement for
    SEIREnvironment as long as it satisfies this protocol, so that the transition
    function and the RL pipeline (environment, PPO, evaluation) can be developed and
    swapped independently.

    Compartment ordering in `seir_state` must follow
    epcontrol.compartments.AgeSEIR.Compartment (S, E, I, R); the age axis must follow
    epcontrol.compartments.contacts.Eames2012.
    """

    #: names of the districts/patches, in the same order as the district axis of `seir_state`
    district_names: List[str]

    #: shape (n_districts, n_compartments, n_age_groups)
    seir_state: np.ndarray

    def reset(self) -> None:
        """Reset all districts to their initial (disease-free) state."""
        ...

    def seed(self, region: str) -> None:
        """Mark `region` as the district where the epidemic is seeded."""
        ...

    def step(self, t: int, actions: Sequence[int]) -> None:
        """Advance the model by one day given the per-district action for day `t`."""
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
