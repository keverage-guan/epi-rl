"""Policy-conditioned PINN surrogate for the age-structured SEIR meta-population model.

This package fits the Libin et al. meta-population SEIR model (districts as patches,
mean-field inter-patch coupling, term/holiday school-closure switch) to real 2009
H1N1 ILI data using a physics-informed neural network in DeepXDE (PyTorch backend).

Design (see the accompanying write-up):
  * Patches P = GB districts (native resolution from the repo census/commute data).
  * Observation units R = nations (England, Scotland, Wales) via a district->nation
    crosswalk; the data loss aggregates district predictions up to each nation.
  * Population conservation S+E+I+R = N is enforced exactly by a softmax head.
  * The weekly term/holiday switch is handled by domain decomposition (one smooth
    sub-domain per week) with overlapping junction losses at week boundaries.
  * Parameters R0, mu, kappa, alpha are trained jointly with the network.
"""

from .config import ModelConfig, TrainConfig
from .data import EpiData, load_epi_data
from .network import SEIRPINN
from .model import PINNTrainer

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "EpiData",
    "load_epi_data",
    "SEIRPINN",
    "PINNTrainer",
]