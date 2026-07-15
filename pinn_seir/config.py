"""Configuration objects for the SEIR PINN.

All tunable quantities live here so that scripts stay declarative and the physics
code never hard-codes a magic number. Values follow Libin et al. unless noted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Epidemiological / model structure
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Fixed structural constants and file locations for the meta-population model.

    Rates are expressed per DAY (matching the reference model), while the PINN's
    time axis is in WEEKS; :attr:`days_per_week` bridges the two inside the ODE
    residual so that d/d(week) = 7 * d/d(day).
    """

    # ---- data locations (point these at your repo's data/ tree) ----------- #
    census_path: Path = Path("data/great_brittain/census.csv")
    commute_path: Path = Path("data/great_brittain/commute.csv")
    crosswalk_path: Path = Path("data/great_brittain/crosswalk.tsv")
    contacts_dir: Path = Path("data/contacts")
    flu_path: Path = Path("data/uk_flu_per_100000.csv")

    # ---- age structure (Eames2012 order: Children, Adolescents, Adults, Elderly)
    n_age_groups: int = 4
    adult_index: int = 2  # index of the Adults age group in the contact/census order

    # ---- time axis -------------------------------------------------------- #
    n_weeks: int = 43
    days_per_week: float = 7.0

    # ---- fixed epidemiological rates (per DAY) ---------------------------- #
    zeta: float = 1.0            # latency rate  (1-day latent period): E -> I
    gamma: float = 1.0 / 1.8     # recovery rate (1.8-day infectious period): I -> R

    # ---- initial guesses for trainable parameters ------------------------- #
    r0_init: float = 1.8         # basic reproduction number (trainable)
    mu_init: float = 0.5         # susceptible-modulation exponent in (0,1) (trainable)
    kappa_init: float = 1.0      # global coupling scale (fix to 1 first)
    alpha_init: float = 0.25     # ascertainment fraction (symptomatic 1/4 scaling)

    train_r0: bool = True
    train_mu: bool = True
    train_kappa: bool = False    # keep fixed at 1.0 unless the fit demands otherwise
    train_alpha: bool = False    # fixed at 1/4 per Libin et al.; free it only if needed

    # ---- observation model ------------------------------------------------ #
    # The ILI series is a *rate per 100k* and is treated as weekly INCIDENCE, i.e.
    # new symptomatic infections that week = alpha * (zeta * E) integrated over the
    # week. observation_scale converts a per-capita weekly incidence into the units
    # of the data (per 100,000 population).
    observation_scale: float = 1.0e5

    # ---- seeding (week-1 initial condition) ------------------------------- #
    # 2009 H1N1: 2 confirmed cases arrived in Falkirk (Scotland) on 27 Apr 2009.
    # This date also anchors model "week 0": week k spans [start + 7k, start + 7(k+1)).
    seed_district: str = "Falkirk"
    seed_exposed_count: float = 2.0
    epidemic_start: str = "2009-04-27"  # ISO date; defines the model week grid

    # ---- fixed historical school calendar, PER NATION (daily) ------------- #
    # Inclusive holiday date ranges keyed by nation. Each district inherits its
    # nation's calendar via the district->nation crosswalk (§1). A day is "term-time"
    # for a district iff it falls in none of its nation's ranges. This is separate
    # from POLICY closure (the weekly, per-patch RL action): schools are open in a
    # district on a day iff (term-time that day for its nation) AND (not policy-closed
    # that week in that district). See schedules.py.
    holiday_ranges_by_nation: Dict[str, List[Tuple[str, str]]] = field(
        default_factory=lambda: {
            "Scotland": [
                ("2009-07-03", "2009-08-16"),  # summer
                ("2009-10-12", "2009-10-23"),  # autumn
                ("2009-12-23", "2010-01-06"),  # christmas
                ("2010-03-29", "2010-04-09"),  # spring
            ],
            "England": [
                ("2009-07-21", "2009-09-02"),  # summer
                ("2009-10-26", "2009-11-01"),  # autumn half-term
                ("2009-12-19", "2010-01-03"),  # christmas
                ("2010-02-13", "2010-02-21"),  # february half-term
                ("2010-04-02", "2010-04-18"),  # spring
                ("2010-05-29", "2010-06-06"),  # may half-term
            ],
            "Wales": [
                ("2009-07-21", "2009-09-02"),  # summer
                ("2009-10-26", "2009-11-01"),  # autumn half-term
                ("2009-12-19", "2010-01-03"),  # christmas
                ("2010-02-13", "2010-02-21"),  # february half-term
                ("2010-04-02", "2010-04-18"),  # spring
                ("2010-05-29", "2010-06-06"),  # may half-term
            ],
        }
    )

    # ---- district subset (None => use every district in the census) ------- #
    districts: Optional[List[str]] = None


# --------------------------------------------------------------------------- #
# Network + optimisation
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Network architecture, loss weights, sampling counts and optimiser schedule."""

    # ---- network ---------------------------------------------------------- #
    hidden_layers: int = 5
    hidden_width: int = 128
    activation: str = "tanh"
    initializer: str = "Glorot normal"
    week_embed_dim: int = 8      # size of the learned per-week index embedding

    # ---- loss weights ----------------------------------------------------- #
    w_phys: float = 1.0
    w_junction: float = 1.0
    w_data: float = 1.0
    w_ic: float = 10.0           # IC is a small, precise anchor -> weight it up

    # ---- domain decomposition -------------------------------------------- #
    overlap_delta: float = 0.12  # half-width of the junction strip, in weeks (0.1-0.15)
    junction_weight: str = "triangle"  # {"uniform", "triangle", "gaussian"}

    # ---- collocation / sampling counts ----------------------------------- #
    n_collocation: int = 32      # collocation points per (week, schedule) per step
    n_junction: int = 16         # points per week-boundary strip per step
    n_schedules: int = 4         # sampled closure schedules per step (incl. the true one)

    # ---- closure-schedule sampling --------------------------------------- #
    budget_weeks: int = 6        # closure budget per patch for random schedules
    include_all_open: bool = True
    include_all_closed: bool = True

    # ---- optimisation ----------------------------------------------------- #
    adam_iters: int = 40_000
    adam_lr: float = 1.0e-3
    lr_decay_step: int = 15_000
    lr_decay_gamma: float = 0.5
    lbfgs_iters: int = 0         # optional L-BFGS polish after Adam (0 = skip)

    # ---- bookkeeping ------------------------------------------------------ #
    log_every: int = 500
    seed: int = 0
    device: str = "cuda"         # falls back to cpu automatically if unavailable
    dtype: str = "float32"

    def weight_kwargs(self) -> Tuple[str, float]:
        return self.junction_weight, self.overlap_delta