from __future__ import annotations

from dataclasses import dataclass

import torch

@dataclass
class PhysicsConstants:

    N: torch.Tensor
    cms_school: torch.Tensor
    cms_holiday: torch.Tensor
    M_AA_school: torch.Tensor
    flux_Tij: torch.Tensor
    ngm_radius: torch.Tensor
    zeta: float
    gamma: float
    adult_index: int
    days_per_week: float
    f_sym: float
    r_asym: float

    @property
    def N_adult(self) -> torch.Tensor:
        return self.N[:, self.adult_index]

def beta_per_patch(consts: PhysicsConstants, r0: torch.Tensor) -> torch.Tensor:
    return r0 * consts.gamma / consts.ngm_radius

def contact_matrices(consts: PhysicsConstants, effective_open: torch.Tensor) -> torch.Tensor:
    c = effective_open.unsqueeze(-1).unsqueeze(-1)
    school = consts.cms_school.unsqueeze(0)
    holiday = consts.cms_holiday.unsqueeze(0)
    return c * school + (1.0 - c) * holiday

def force_of_infection(
    consts: PhysicsConstants,
    beta_p: torch.Tensor,
    M: torch.Tensor,
    I: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    infectious = I + consts.r_asym * A
    rel = infectious / consts.N.unsqueeze(0)
    mixed = torch.matmul(M, rel.unsqueeze(-1)).squeeze(-1)
    return beta_p.view(1, -1, 1) * mixed

def meanfield_inflow(
    consts: PhysicsConstants,
    beta_p: torch.Tensor,
    mu: torch.Tensor,
    kappa: torch.Tensor,
    S: torch.Tensor,
    I: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    adult = consts.adult_index
    S_adult = S[:, :, adult]
    I_adult = I[:, :, adult]
    A_adult = A[:, :, adult]
    infectious_adult = I_adult + consts.r_asym * A_adult
    rel_inf_adult = infectious_adult / consts.N_adult.unsqueeze(0)

    flux = consts.flux_Tij.clone()
    flux.fill_diagonal_(0.0)
    inflow_sum = torch.matmul(rel_inf_adult, flux)

    lam_adult = (
        kappa
        * beta_p.view(1, -1)
        * torch.pow(S_adult.clamp_min(0.0), mu)
        * consts.M_AA_school.view(1, -1)
        * inflow_sum
    )

    lam = torch.zeros_like(S)
    lam[:, :, adult] = lam_adult
    return lam

def seir_residuals(
    consts: PhysicsConstants,
    state: torch.Tensor,
    dstate_dt_week: torch.Tensor,
    beta_p: torch.Tensor,
    mu: torch.Tensor,
    kappa: torch.Tensor,
    effective_open: torch.Tensor,
) -> torch.Tensor:
    S, E, I, A, R = (state[..., k] for k in range(5))
    dS = dstate_dt_week[..., 0] * consts.days_per_week
    dE = dstate_dt_week[..., 1] * consts.days_per_week
    dI = dstate_dt_week[..., 2] * consts.days_per_week
    dA = dstate_dt_week[..., 3] * consts.days_per_week
    dR = dstate_dt_week[..., 4] * consts.days_per_week

    M = contact_matrices(consts, effective_open)
    phi = force_of_infection(consts, beta_p, M, I, A)
    lam = meanfield_inflow(consts, beta_p, mu, kappa, S, I, A)

    f = consts.f_sym

    r_S = dS + phi * S + lam
    r_E = dE - phi * S + consts.zeta * E - lam
    r_I = dI - f * consts.zeta * E + consts.gamma * I
    r_A = dA - (1.0 - f) * consts.zeta * E + consts.gamma * A
    r_R = dR - consts.gamma * (I + A)

    res = torch.stack([r_S, r_E, r_I, r_A, r_R], dim=-1)
    return res / consts.N.unsqueeze(0).unsqueeze(-1)

def nation_weekly_incidence(
    consts: PhysicsConstants,
    E_week: torch.Tensor,
    tau_nodes: torch.Tensor,
    membership: torch.Tensor,
    nation_population: torch.Tensor,
    alpha: torch.Tensor,
    observation_scale: float,
) -> torch.Tensor:
    flux = consts.f_sym * consts.zeta * E_week
    per_patch_flux = flux.sum(dim=-1)

    weekly = torch.trapezoid(per_patch_flux, x=tau_nodes.unsqueeze(-1), dim=1)
    weekly = weekly * consts.days_per_week

    nation_counts = torch.matmul(weekly, membership.t())
    rate = (
        alpha
        * observation_scale
        * nation_counts
        / nation_population.unsqueeze(0)
    )
    return rate.t()
