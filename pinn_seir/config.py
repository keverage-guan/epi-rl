from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

@dataclass
class ModelConfig:

    census_path: Path = Path("data/great_brittain/census.csv")
    commute_path: Path = Path("data/great_brittain/commute.csv")
    crosswalk_path: Path = Path("data/great_brittain/crosswalk.tsv")
    contacts_dir: Path = Path("data/contacts")
    flu_path: Path = Path("data/uk_flu_per_100000.csv")

    n_age_groups: int = 4
    adult_index: int = 2

    n_weeks: int = 43
    days_per_week: float = 7.0

    zeta: float = 1.0
    gamma: float = 1.0 / 1.8

    r0_init: float = 1.8
    mu_init: float = 0.5
    kappa_init: float = 1.0
    alpha_init: float = 0.25

    train_r0: bool = True
    train_mu: bool = True
    train_kappa: bool = False
    train_alpha: bool = False

    f_sym: float = 0.5
    r_asym: float = 0.5

    train_f_sym: bool = False
    train_r_asym: bool = False

    observation_scale: float = 1.0e5

    seed_district: str = "Falkirk"
    seed_exposed_count: float = 2.0
    epidemic_start: str = "2009-04-27"

    holiday_ranges_by_nation: Dict[str, List[Tuple[str, str]]] = field(
        default_factory=lambda: {
            "Scotland": [
                ("2009-07-03", "2009-08-16"),
                ("2009-10-12", "2009-10-23"),
                ("2009-12-23", "2010-01-06"),
                ("2010-03-29", "2010-04-09"),
            ],
            "England": [
                ("2009-07-21", "2009-09-02"),
                ("2009-10-26", "2009-11-01"),
                ("2009-12-19", "2010-01-03"),
                ("2010-02-13", "2010-02-21"),
                ("2010-04-02", "2010-04-18"),
                ("2010-05-29", "2010-06-06"),
            ],
            "Wales": [
                ("2009-07-21", "2009-09-02"),
                ("2009-10-26", "2009-11-01"),
                ("2009-12-19", "2010-01-03"),
                ("2010-02-13", "2010-02-21"),
                ("2010-04-02", "2010-04-18"),
                ("2010-05-29", "2010-06-06"),
            ],
        }
    )

    districts: Optional[List[str]] = None

@dataclass
class TrainConfig:

    hidden_layers: int = 5
    hidden_width: int = 128
    activation: str = "tanh"
    initializer: str = "Glorot normal"
    week_embed_dim: int = 8

    dropout_rate: float = 0.0

    w_phys: float = 1.0
    w_junction: float = 1.0
    w_data: float = 1.0
    w_ic: float = 10.0

    phys_activity_floor: float = 0.1

    w_monotonic: float = 10.0

    w_cumulative: float = 10.0

    alpha_prior_weight: float = 1.0
    alpha_prior_mean: float = 0.25

    overlap_delta: float = 0.12
    junction_weight: str = "triangle"

    n_collocation: int = 32
    n_junction: int = 16
    n_schedules: int = 4

    budget_weeks: int = 6
    include_all_open: bool = True
    include_all_closed: bool = True

    adam_iters: int = 40_000
    adam_lr: float = 1.0e-3
    lr_decay_step: int = 15_000
    lr_decay_gamma: float = 0.5
    lbfgs_iters: int = 0

    log_every: int = 500
    seed: int = 0
    device: str = "cuda"
    dtype: str = "float32"

    def weight_kwargs(self) -> Tuple[str, float]:
        return self.junction_weight, self.overlap_delta
