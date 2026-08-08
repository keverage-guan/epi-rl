from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ModelConfig

def _read_contact_matrix(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", dtype=np.float64)

def _make_reciprocal(cm: np.ndarray, census_row: np.ndarray) -> np.ndarray:
    census_row = np.asarray(census_row, dtype=np.float64)
    dim = cm.shape[0]
    out = np.empty((dim, dim), dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            out[i, j] = (
                census_row[i] * cm[i, j] + census_row[j] * cm[j, i]
            ) / (2.0 * census_row[i])
    return out

def _spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))

@dataclass
class EpiData:

    district_names: List[str]
    nation_names: List[str]
    district_nations: List[str]
    N: np.ndarray
    cms_school: np.ndarray
    cms_holiday: np.ndarray
    M_AA_school: np.ndarray
    flux_Tij: np.ndarray
    ngm_radius: np.ndarray
    nation_membership: np.ndarray
    seed_district_index: int
    y_obs: np.ndarray
    obs_week_index: np.ndarray
    nation_population: np.ndarray

    @property
    def n_patches(self) -> int:
        return len(self.district_names)

    @property
    def n_nations(self) -> int:
        return len(self.nation_names)

    def beta_per_patch(self, r0: float) -> np.ndarray:
        return r0 * (1.0 / 1.8) / self.ngm_radius

def load_epi_data(cfg: ModelConfig) -> EpiData:

    census_df = pd.read_csv(cfg.census_path, index_col=0)
    flux_df = pd.read_csv(cfg.commute_path, index_col=0)
    crosswalk = _load_crosswalk(cfg.crosswalk_path)

    flux_order = [str(c) for c in flux_df.columns]
    district_names = [d for d in flux_order if d in census_df.index]
    if cfg.districts is not None:
        wanted = set(cfg.districts)
        district_names = [d for d in district_names if d in wanted]
        missing = wanted - set(district_names)
        if missing:
            raise ValueError(f"Requested districts absent from data: {sorted(missing)}")
    if not district_names:
        raise ValueError("No districts left after applying census/flux/subset filters.")

    p_index = {name: i for i, name in enumerate(district_names)}
    n_patches = len(district_names)

    census = census_df.loc[district_names].to_numpy(dtype=np.float64)
    if census.shape[1] != cfg.n_age_groups:
        raise ValueError(
            f"Census has {census.shape[1]} age columns; expected {cfg.n_age_groups}."
        )

    cm_school = _read_contact_matrix(cfg.contacts_dir / "conversational_school.csv")
    cm_holiday = _read_contact_matrix(cfg.contacts_dir / "conversational_no_school.csv")

    cms_school = np.empty((n_patches, cfg.n_age_groups, cfg.n_age_groups), dtype=np.float64)
    cms_holiday = np.empty_like(cms_school)
    ngm_radius = np.empty(n_patches, dtype=np.float64)
    for i in range(n_patches):
        cms_school[i] = _make_reciprocal(cm_school, census[i])
        cms_holiday[i] = _make_reciprocal(cm_holiday, census[i])
        ngm_radius[i] = _spectral_radius(cms_school[i])

    m_aa_school = cms_school[:, cfg.adult_index, cfg.adult_index].copy()

    flux_full = flux_df.reindex(index=flux_df.columns)
    flux_Tij = flux_full.loc[district_names, district_names].to_numpy(dtype=np.float64)

    nation_of = {}
    for d in district_names:
        if d not in crosswalk:
            raise ValueError(f"District '{d}' has no nation in the crosswalk.")
        nation_of[d] = crosswalk[d]
    nation_names = sorted(set(nation_of.values()))
    r_index = {name: r for r, name in enumerate(nation_names)}
    membership = np.zeros((len(nation_names), n_patches), dtype=np.float64)
    for d, r in nation_of.items():
        membership[r_index[r], p_index[d]] = 1.0
    district_nations = [nation_of[d] for d in district_names]

    nation_population = membership @ census.sum(axis=1)

    if cfg.seed_district not in p_index:
        raise ValueError(
            f"Seed district '{cfg.seed_district}' is not in the active district set."
        )
    seed_idx = p_index[cfg.seed_district]

    y_obs, obs_week_index = _load_flu_series(cfg, nation_names)

    return EpiData(
        district_names=district_names,
        nation_names=nation_names,
        district_nations=district_nations,
        N=census.astype(np.float64),
        cms_school=cms_school,
        cms_holiday=cms_holiday,
        M_AA_school=m_aa_school,
        flux_Tij=flux_Tij,
        ngm_radius=ngm_radius,
        nation_membership=membership,
        seed_district_index=seed_idx,
        y_obs=y_obs,
        obs_week_index=obs_week_index,
        nation_population=nation_population,
    )

def _load_crosswalk(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    if "district" not in cols or "nation" not in cols:
        raise ValueError("crosswalk.tsv must have 'district' and 'nation' columns.")
    return dict(zip(df[cols["district"]].astype(str), df[cols["nation"]].astype(str)))

def _load_flu_series(
    cfg: ModelConfig, nation_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(cfg.flu_path)
    lower = {c.lower(): c for c in df.columns}

    if "week_end_date" not in lower:
        raise ValueError(
            "Flu CSV must contain a 'week_end_date' column (format M/D/Y) to align "
            f"observations to model weeks. Found columns: {list(df.columns)}"
        )

    start = datetime.strptime(cfg.epidemic_start, "%Y-%m-%d").date()
    end_dates = pd.to_datetime(df[lower["week_end_date"]], format="%m/%d/%Y").dt.date

    model_week = np.array(
        [(d - start).days // 7 for d in end_dates], dtype=np.int64
    )
    keep = (model_week >= 0) & (model_week < cfg.n_weeks)
    if not keep.any():
        raise ValueError(
            "No flu observations fall within the model horizon "
            f"[{cfg.epidemic_start}, +{cfg.n_weeks} weeks). Check epidemic_start and "
            "the week_end_date column."
        )

    week_index = model_week[keep]
    rows = []
    for nation in nation_names:
        key = nation.lower()
        if key not in lower:
            raise ValueError(
                f"Flu CSV has no column for nation '{nation}'. "
                f"Available: {list(df.columns)}"
            )
        rows.append(df[lower[key]].to_numpy(dtype=np.float64)[keep])
    y = np.vstack(rows)

    order = np.argsort(week_index, kind="stable")
    return y[:, order], week_index[order]
