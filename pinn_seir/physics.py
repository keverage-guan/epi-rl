"""Vectorised SEIAR meta-population physics in PyTorch.

Everything here operates on batched states of shape (B, P, A) and returns residuals
in the same shape (or scalars). Rates are per DAY; the network's time coordinate is
in WEEKS, so the caller multiplies autodiff time-derivatives by ``days_per_week`` to
convert d/d(week) into d/d(day) before forming residuals.

Residuals are expressed in FRACTIONS of N so that every compartment is O(1); this is
the single most important numerical detail for a usable fit.

Compartments: S, E, I (symptomatic infectious), A (asymptomatic infectious), R.
On leaving E, a fraction ``f`` becomes symptomatic (I) and ``1-f`` asymptomatic (A);
both transmit, with asymptomatics scaled by ``r_A`` in [0,1]. Only symptomatic
*incidence* (f * zeta * E) is ascertained into the ILI data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PhysicsConstants:
    """Device-resident fixed tensors + scalar rates for the residual computation."""

    N: torch.Tensor             # (P, A) population
    cms_school: torch.Tensor    # (P, A, A)
    cms_holiday: torch.Tensor   # (P, A, A)
    M_AA_school: torch.Tensor   # (P,)
    flux_Tij: torch.Tensor      # (P, P), Tij[i,j] = flux i->j
    ngm_radius: torch.Tensor    # (P,)
    zeta: float                 # latency rate  (per day)
    gamma: float                # recovery rate (per day)
    adult_index: int
    days_per_week: float
    f_sym: float                # symptomatic fraction on leaving E (E -> I)
    r_asym: float               # relative infectiousness of asymptomatics in [0,1]

    @property
    def N_adult(self) -> torch.Tensor:
        return self.N[:, self.adult_index]  # (P,)


def beta_per_patch(consts: PhysicsConstants, r0: torch.Tensor) -> torch.Tensor:
    """Per-district beta from a shared scalar R0 via the next-generation matrix.

    beta_p = R0 * gamma / rho(reciprocal school contact matrix of patch p).
    Mirrors ``Eames2012.compute_beta`` but keeps R0 differentiable.
    """
    return r0 * consts.gamma / consts.ngm_radius  # (P,)


def contact_matrices(consts: PhysicsConstants, effective_open: torch.Tensor) -> torch.Tensor:
    """Select each patch's contact matrix from its EFFECTIVE open state.

    effective_open: (B, P) in {0,1}, 1 = schools open (term-time AND not policy-closed),
    0 = schools closed (holiday OR policy-closed). Returns (B, P, A, A). No smoothing of
    the control -- the switch is genuine. The caller forms effective_open by combining
    the fixed daily calendar (term-time) with the weekly policy closure.
    """
    c = effective_open.unsqueeze(-1).unsqueeze(-1)  # (B, P, 1, 1)
    school = consts.cms_school.unsqueeze(0)   # (1, P, A, A)
    holiday = consts.cms_holiday.unsqueeze(0)
    return c * school + (1.0 - c) * holiday   # (B, P, A, A)


def force_of_infection(
    consts: PhysicsConstants,
    beta_p: torch.Tensor,   # (P,)
    M: torch.Tensor,        # (B, P, A, A) per-patch contact matrix for this week
    I: torch.Tensor,        # (B, P, A) symptomatic infectious counts
    A: torch.Tensor,        # (B, P, A) asymptomatic infectious counts
) -> torch.Tensor:
    """phi_{p,i} = beta_p * sum_j M_{p,ij} * (I_{p,j} + r_A * A_{p,j}) / N_{p,j}.

    Both infectious classes contribute to transmission; asymptomatics are scaled by
    the relative infectiousness r_A. Returns (B, P, A).
    """
    infectious = I + consts.r_asym * A                    # (B, P, A)
    rel = infectious / consts.N.unsqueeze(0)              # (B, P, A) = (I + r_A A)/N
    # (B,P,A,A) @ (B,P,A,1) -> (B,P,A,1)
    mixed = torch.matmul(M, rel.unsqueeze(-1)).squeeze(-1)  # (B, P, A)
    return beta_p.view(1, -1, 1) * mixed                  # (B, P, A)


def meanfield_inflow(
    consts: PhysicsConstants,
    beta_p: torch.Tensor,   # (P,)
    mu: torch.Tensor,       # scalar in (0,1)
    kappa: torch.Tensor,    # scalar
    S: torch.Tensor,        # (B, P, A) susceptibles
    I: torch.Tensor,        # (B, P, A) symptomatic infectious
    A: torch.Tensor,        # (B, P, A) asymptomatic infectious
) -> torch.Tensor:
    """Deterministic mean-field analogue of the Poisson inter-patch ignition term.

    Lambda_{p,A} = kappa * beta_p * (S^A_p)^mu * M_AA
                   * sum_{p'!=p} T_{p'p} * (I^A_{p'} + r_A * A^A_{p'}) / N^A_{p'}
    and zero for non-adult age groups. Returns (B, P, A). Asymptomatic adults also
    seed other patches, scaled by r_A.

    The flux enters as the column toward the receiving patch p (T_{p'p}), matching the
    reference implementation's ``flux_k = flux_tij[:, target]`` convention.
    """
    adult = consts.adult_index
    S_adult = S[:, :, adult]                              # (B, P)
    I_adult = I[:, :, adult]                              # (B, P)
    A_adult = A[:, :, adult]                              # (B, P)
    infectious_adult = I_adult + consts.r_asym * A_adult  # (B, P)
    rel_inf_adult = infectious_adult / consts.N_adult.unsqueeze(0)  # (B, P)

    # sum over sources p': (B, P') @ (P', P) -> (B, P), with self-flux removed.
    flux = consts.flux_Tij.clone()
    flux.fill_diagonal_(0.0)                              # exclude p' == p
    inflow_sum = torch.matmul(rel_inf_adult, flux)        # (B, P)

    lam_adult = (
        kappa
        * beta_p.view(1, -1)
        * torch.pow(S_adult.clamp_min(0.0), mu)
        * consts.M_AA_school.view(1, -1)
        * inflow_sum
    )                                                     # (B, P)

    lam = torch.zeros_like(S)                             # (B, P, A)
    lam[:, :, adult] = lam_adult
    return lam


def seir_residuals(
    consts: PhysicsConstants,
    state: torch.Tensor,        # (B, P, A, 5) counts (S,E,I,A,R)
    dstate_dt_week: torch.Tensor,  # (B, P, A, 5) d/d(week) from autodiff
    beta_p: torch.Tensor,       # (P,)
    mu: torch.Tensor,
    kappa: torch.Tensor,
    effective_open: torch.Tensor,  # (B, P) 1 = schools open (term AND not policy-closed)
) -> torch.Tensor:
    """Return the five SEIAR residuals stacked as (B, P, A, 5), in fractions of N.

    d/d(day) = days_per_week * d/d(week). All physics rates are per day.
    """
    S, E, I, A, R = (state[..., k] for k in range(5))
    dS = dstate_dt_week[..., 0] * consts.days_per_week
    dE = dstate_dt_week[..., 1] * consts.days_per_week
    dI = dstate_dt_week[..., 2] * consts.days_per_week
    dA = dstate_dt_week[..., 3] * consts.days_per_week
    dR = dstate_dt_week[..., 4] * consts.days_per_week

    M = contact_matrices(consts, effective_open)         # (B, P, A, A)
    phi = force_of_infection(consts, beta_p, M, I, A)    # (B, P, A)
    lam = meanfield_inflow(consts, beta_p, mu, kappa, S, I, A)  # (B, P, A)

    f = consts.f_sym

    # Reference ODE with symptomatic/asymptomatic split and mean-field inflow Lambda
    # entering S (out) and E (in):
    #   dS/dt = -phi S                       - Lambda
    #   dE/dt =  phi S - zeta E              + Lambda
    #   dI/dt =  f * zeta E     - gamma I
    #   dA/dt =  (1-f) * zeta E - gamma A
    #   dR/dt =  gamma (I + A)
    r_S = dS + phi * S + lam
    r_E = dE - phi * S + consts.zeta * E - lam
    r_I = dI - f * consts.zeta * E + consts.gamma * I
    r_A = dA - (1.0 - f) * consts.zeta * E + consts.gamma * A
    r_R = dR - consts.gamma * (I + A)

    res = torch.stack([r_S, r_E, r_I, r_A, r_R], dim=-1)  # (B, P, A, 5)
    # Normalise to fractions of N so each compartment residual is O(1).
    return res / consts.N.unsqueeze(0).unsqueeze(-1)


def nation_weekly_incidence(
    consts: PhysicsConstants,
    E_week: torch.Tensor,       # (n_weeks, n_nodes, P, A) exposed at collocation nodes
    tau_nodes: torch.Tensor,    # (n_weeks, n_nodes) local-time nodes used for the week
    membership: torch.Tensor,   # (R, P)
    nation_population: torch.Tensor,  # (R,)
    alpha: torch.Tensor,        # scalar ascertainment
    observation_scale: float,
) -> torch.Tensor:
    """Model-predicted weekly ILI *incidence* per 100k, per nation.

    Observed ILI is ASCERTAINED SYMPTOMATIC incidence. New symptomatic infections in
    week k = f * integral_0^1 zeta * E(tau) dtau, and only alpha of those are
    ascertained, so the observed model is:

        alpha * f * integral zeta * E,

    summed over patch & age, aggregated to nations, converted to a per-100k rate.

    E_week and tau_nodes are expected pre-shaped as (n_weeks, n_nodes, P, A) and
    (n_weeks, n_nodes); the trapezoid integral is taken over the node axis.
    """
    # f * zeta * E gives the instantaneous new-SYMPTOMATIC-infection flux per day.
    flux = consts.f_sym * consts.zeta * E_week           # (weeks, nodes, P, A)
    per_patch_flux = flux.sum(dim=-1)                    # (weeks, nodes, P): sum over age

    # Integrate over the week in DAYS: dt_day = days_per_week * dtau.
    weekly = torch.trapezoid(per_patch_flux, x=tau_nodes.unsqueeze(-1), dim=1)
    weekly = weekly * consts.days_per_week               # (weeks, P) new sympt./week

    # Aggregate patches -> nations, scale by ascertainment, convert to per-100k rate.
    nation_counts = torch.matmul(weekly, membership.t())  # (weeks, R)
    rate = (
        alpha
        * observation_scale
        * nation_counts
        / nation_population.unsqueeze(0)
    )                                                    # (weeks, R)
    return rate.t()                                      # (R, weeks)