"""Training driver: builds the four loss terms and jointly optimises theta + parameters.

Loss = w_phys * L_phys + w_junction * L_junction + w_data * L_data + w_ic * L_ic

  * L_phys      SEIR + mean-field residual, over sampled schedules and all weeks.
  * L_junction  overlapping-strip state-continuity across week boundaries.
  * L_data      weekly ILI incidence aggregated to nations, only the true calendar.
  * L_ic        week-1 initial condition (Falkirk seed, S ~ N elsewhere).

The network output already enforces S+E+I+R = N exactly (softmax head), so there is
no conservation penalty. Time-derivatives are obtained by autodiff of the network
w.r.t. the local-time coordinate tau.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .config import ModelConfig, TrainConfig
from .data import EpiData
from .network import SEIRPINN
from .physics import (
    PhysicsConstants,
    beta_per_patch,
    nation_weekly_incidence,
    seir_residuals,
)
from .schedules import ScheduleSampler, DailyCalendar


def _junction_weight(tau_rel: torch.Tensor, kind: str, delta: float) -> torch.Tensor:
    """Weight over the overlap strip, as a function of distance from the boundary.

    tau_rel is the signed distance (in weeks) from the boundary t_{k+1}.
    """
    a = tau_rel.abs()
    if kind == "uniform":
        return torch.ones_like(a)
    if kind == "triangle":
        return torch.clamp(1.0 - a / delta, min=0.0)
    if kind == "gaussian":
        sigma = delta / 2.0
        return torch.exp(-(tau_rel ** 2) / (2.0 * sigma ** 2))
    raise ValueError(f"Unknown junction weight '{kind}'.")


class PINNTrainer:
    """Owns the network, trainable parameters, fixed physics tensors, and the loop."""

    def __init__(self, data: EpiData, mcfg: ModelConfig, tcfg: TrainConfig) -> None:
        self.data = data
        self.mcfg = mcfg
        self.tcfg = tcfg

        self.device = torch.device(
            tcfg.device if (tcfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self.dtype = getattr(torch, tcfg.dtype)
        torch.manual_seed(tcfg.seed)
        np.random.seed(tcfg.seed)

        P, A = data.n_patches, mcfg.n_age_groups

        def T(arr):
            return torch.as_tensor(arr, dtype=self.dtype, device=self.device)

        # ---- fixed physics tensors ---------------------------------------- #
        self.consts = PhysicsConstants(
            N=T(data.N),
            cms_school=T(data.cms_school),
            cms_holiday=T(data.cms_holiday),
            M_AA_school=T(data.M_AA_school),
            flux_Tij=T(data.flux_Tij),
            ngm_radius=T(data.ngm_radius),
            zeta=mcfg.zeta,
            gamma=mcfg.gamma,
            adult_index=mcfg.adult_index,
            days_per_week=mcfg.days_per_week,
        )
        self.membership = T(data.nation_membership)          # (R, P)
        self.nation_population = T(data.nation_population)    # (R,)
        self.y_obs = T(data.y_obs)                            # (R, T)
        self.obs_week_index = torch.as_tensor(
            data.obs_week_index, dtype=torch.int64, device=self.device
        )
        # Characteristic scale for the data loss: mean-square of the observations.
        # Dividing the data MSE by this makes it dimensionless and O(1), so the
        # loss weights in TrainConfig are comparable across the four terms.
        self._data_scale = float((data.y_obs ** 2).mean()) + 1e-8

        # ---- network ------------------------------------------------------ #
        self.net = SEIRPINN(
            n_patches=P,
            n_age_groups=A,
            n_weeks=mcfg.n_weeks,
            N=T(data.N),
            cfg=tcfg,
            use_budget=False,
        ).to(self.device)

        # ---- trainable physics parameters (unconstrained -> transformed) -- #
        self.raw_r0 = self._make_param(mcfg.r0_init, mcfg.train_r0, positive=True)
        self.raw_mu = self._make_param(mcfg.mu_init, mcfg.train_mu, unit=True)
        self.raw_kappa = self._make_param(mcfg.kappa_init, mcfg.train_kappa, positive=True)
        self.raw_alpha = self._make_param(mcfg.alpha_init, mcfg.train_alpha, unit=True)

        # ---- schedule sampler (weekly POLICY) + fixed daily calendar ------- #
        self.sampler = ScheduleSampler(
            n_weeks=mcfg.n_weeks,
            n_patches=P,
            budget_weeks=tcfg.budget_weeks,
            include_all_open=tcfg.include_all_open,
            include_all_closed=tcfg.include_all_closed,
            seed=tcfg.seed,
        )
        # Fixed historical school calendar at daily resolution, PER PATCH.
        self.calendar = DailyCalendar(
            epidemic_start=mcfg.epidemic_start,
            holiday_ranges_by_nation=mcfg.holiday_ranges_by_nation,
            district_nations=data.district_nations,
            n_weeks=mcfg.n_weeks,
            days_per_week=int(mcfg.days_per_week),
        )
        # Device tensor of the per-patch term-time table: (P, n_weeks, days_per_week)
        self.calendar_table = torch.as_tensor(
            self.calendar.table, dtype=self.dtype, device=self.device
        )
        self._build_ic_targets()

    def _term_time(self, tau: torch.Tensor, week: torch.Tensor) -> torch.Tensor:
        """Per-patch term-time (1/0) for each (tau, week); returns (B, P).

        Day within week = floor(tau * days_per_week); indexes the precomputed
        per-patch table (P, n_weeks, days_per_week).
        """
        dpw = self.calendar_table.shape[-1]
        day = torch.clamp((tau.squeeze(-1) * dpw).floor().long(), 0, dpw - 1)  # (B,)
        term = self.calendar_table[:, week, day]                              # (P, B)
        return term.t()                                                        # (B, P)

    # ------------------------------------------------------------------ #
    # parameter transforms
    # ------------------------------------------------------------------ #
    def _make_param(self, init: float, trainable: bool, positive=False, unit=False):
        """Create a raw (unconstrained) parameter; softplus->positive, sigmoid->(0,1)."""
        if positive:
            raw = float(np.log(np.expm1(max(init, 1e-4))))  # inverse softplus
        elif unit:
            init = min(max(init, 1e-4), 1 - 1e-4)
            raw = float(np.log(init / (1 - init)))          # inverse sigmoid
        else:
            raw = float(init)
        t = torch.tensor(raw, dtype=self.dtype, device=self.device, requires_grad=trainable)
        return t

    @property
    def r0(self):
        return torch.nn.functional.softplus(self.raw_r0)

    @property
    def mu(self):
        return torch.sigmoid(self.raw_mu)

    @property
    def kappa(self):
        return torch.nn.functional.softplus(self.raw_kappa)

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    def trainable_parameters(self):
        params = list(self.net.parameters())
        for raw in (self.raw_r0, self.raw_mu, self.raw_kappa, self.raw_alpha):
            if raw.requires_grad:
                params.append(raw)
        return params

    # ------------------------------------------------------------------ #
    # initial condition targets (week 1)
    # ------------------------------------------------------------------ #
    def _build_ic_targets(self) -> None:
        """S ~ N everywhere, a small seed of exposed adults in the seed district, I=R=0."""
        P, A = self.data.n_patches, self.mcfg.n_age_groups
        N = self.data.N
        S0 = N.copy().astype(np.float64)
        E0 = np.zeros_like(S0)
        I0 = np.zeros_like(S0)
        adult = self.mcfg.adult_index
        seed = self.data.seed_district_index
        E0[seed, adult] = self.mcfg.seed_exposed_count
        S0[seed, adult] = max(N[seed, adult] - self.mcfg.seed_exposed_count, 0.0)

        def T(a):
            return torch.as_tensor(a, dtype=self.dtype, device=self.device)

        self.ic_S = T(S0)
        self.ic_E = T(E0)
        self.ic_I = T(I0)

    # ------------------------------------------------------------------ #
    # forward helper: evaluate net + d/dtau at given (tau, week, closure)
    # ------------------------------------------------------------------ #
    def _forward_with_dt(
        self, tau: torch.Tensor, week: torch.Tensor, closure: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (state, d state / d(week)), both (B, P, A, 4)."""
        # Reverse-mode fallback: one grad call per output column. Correct but O(P*A*4)
        # backward passes; only used if forward-mode AD is unavailable.
        tau = tau.clone().requires_grad_(True)
        state = self.net(tau, week, closure)                 # (B, P, A, 4)
        B = state.shape[0]
        flat = state.reshape(B, -1)                          # (B, P*A*4)
        grads = torch.zeros_like(flat)
        ones = torch.ones(B, device=self.device, dtype=self.dtype)
        for j in range(flat.shape[1]):
            g = torch.autograd.grad(
                flat[:, j], tau, grad_outputs=ones, retain_graph=True, create_graph=True
            )[0]                                             # (B, 1)
            grads[:, j] = g.squeeze(1)
        dstate = grads.reshape(B, self.data.n_patches, self.mcfg.n_age_groups, 4)
        return state, dstate

    def _forward_with_dt_fast(
        self, tau: torch.Tensor, week: torch.Tensor, closure: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """d state / d tau via forward-mode AD (jvp).

        One jvp with a unit tangent on tau yields the full time-derivative in a single
        pass whose cost is independent of output width -- essential at 378 districts.
        """
        from torch.func import jvp  # available in recent torch

        def f(tau_in):
            return self.net(tau_in, week, closure)

        tangent = torch.ones_like(tau)
        state, dstate = jvp(f, (tau,), (tangent,))
        return state, dstate

    def _time_derivative(self, tau, week, closure):
        """Prefer fast forward-mode AD; fall back to the loop if unavailable."""
        try:
            return self._forward_with_dt_fast(tau, week, closure)
        except Exception:
            return self._forward_with_dt(tau, week, closure)

    # ------------------------------------------------------------------ #
    # loss terms
    # ------------------------------------------------------------------ #
    def loss_physics(self, schedules: torch.Tensor) -> torch.Tensor:
        """Mean squared SEIR residual over sampled schedules, all weeks, collocation nodes."""
        Ns, n_weeks, P = schedules.shape
        A = self.mcfg.n_age_groups
        nc = self.tcfg.n_collocation

        beta_p = beta_per_patch(self.consts, self.r0)
        total = torch.zeros((), device=self.device, dtype=self.dtype)

        for s in range(Ns):
            # collocation local-times shared across weeks this step
            tau = torch.rand(n_weeks * nc, 1, device=self.device, dtype=self.dtype)
            weeks = torch.repeat_interleave(
                torch.arange(n_weeks, device=self.device), nc
            )
            policy = schedules[s][weeks]                     # (n_weeks*nc, P) weekly policy
            # Network is conditioned on the POLICY closure (the RL action).
            state, dstate = self._time_derivative(tau, weeks, policy)
            # Effective open = term-time(day) AND not policy-closed = product of {0,1}s.
            term = self._term_time(tau, weeks)              # (n_weeks*nc, 1)
            effective_open = term * policy                  # (n_weeks*nc, P)
            res = seir_residuals(
                self.consts, state, dstate, beta_p, self.mu, self.kappa, effective_open
            )
            total = total + (res ** 2).mean()
        return total / Ns

    def loss_junction(self, schedules: torch.Tensor) -> torch.Tensor:
        """Overlapping-strip continuity of the state across week boundaries."""
        Ns, n_weeks, P = schedules.shape
        delta = self.tcfg.overlap_delta
        kind = self.tcfg.junction_weight
        nq = self.tcfg.n_junction

        total = torch.zeros((), device=self.device, dtype=self.dtype)
        count = 0
        for s in range(Ns):
            for k in range(n_weeks - 1):
                # sample strip points around the boundary t_{k+1}, in signed offset.
                off = (torch.rand(nq, 1, device=self.device, dtype=self.dtype) * 2 - 1) * delta
                # week-k side: local tau near 1 (end of week k) => tau = 1 + off
                tau_k = (1.0 + off).clamp(0.0, 1.0)
                # week-(k+1) side: local tau near 0 (start of week k+1) => tau = off
                tau_k1 = (off).clamp(0.0, 1.0)
                wk = torch.full((nq,), k, device=self.device, dtype=torch.int64)
                wk1 = torch.full((nq,), k + 1, device=self.device, dtype=torch.int64)
                ck = schedules[s, k].unsqueeze(0).expand(nq, P)
                ck1 = schedules[s, k + 1].unsqueeze(0).expand(nq, P)

                u_k = self.net(tau_k, wk, ck)                # (nq, P, A, 4)
                u_k1 = self.net(tau_k1, wk1, ck1)
                w = _junction_weight(off, kind, delta)       # (nq, 1)
                # continuity in FRACTIONS of N, matching the physics/IC normalisation
                N = self.consts.N.unsqueeze(0).unsqueeze(-1)  # (1, P, A, 1)
                diff = ((u_k - u_k1) / N).reshape(nq, -1)
                total = total + (w * (diff ** 2)).mean()
                count += 1
        return total / max(count, 1)

    def loss_data(self) -> torch.Tensor:
        """Weekly ILI incidence per nation under the true (all-open) policy vs. observations.

        The network is conditioned on the true historical POLICY (all-open in 2009); the
        fixed daily calendar shapes the dynamics through the physics loss, so the learned
        E already reflects term/holiday mixing. Incidence = alpha * integral of zeta*E.
        """
        P, A = self.data.n_patches, self.mcfg.n_age_groups
        n_weeks = self.mcfg.n_weeks
        nq = self.tcfg.n_collocation

        true_policy = torch.as_tensor(
            self.sampler.true, dtype=self.dtype, device=self.device
        )  # (n_weeks, P), all-open

        # Evaluate E on a fixed quadrature grid within each week to integrate incidence.
        tau_nodes = torch.linspace(0, 1, nq, device=self.device, dtype=self.dtype)
        tau_grid = tau_nodes.repeat(n_weeks).unsqueeze(1)    # (n_weeks*nq, 1)
        weeks = torch.repeat_interleave(
            torch.arange(n_weeks, device=self.device), nq
        )
        policy = true_policy[weeks]                          # (n_weeks*nq, P)
        state = self.net(tau_grid, weeks, policy)            # (n_weeks*nq, P, A, 4)
        E = state[..., 1].reshape(n_weeks, nq, P, A)
        tau_week = tau_nodes.unsqueeze(0).expand(n_weeks, nq)

        pred = nation_weekly_incidence(
            self.consts, E, tau_week, self.membership,
            self.nation_population, self.alpha, self.mcfg.observation_scale,
        )                                                    # (R, n_weeks)

        # obs_week_index holds the MODEL week each observation maps to (by date);
        # y_obs is aligned to that same ordering.
        pred_obs = pred[:, self.obs_week_index]              # (R, n_obs)
        return ((pred_obs - self.y_obs) ** 2).mean() / self._data_scale

    def loss_ic(self) -> torch.Tensor:
        """Week-1 (index 0) initial condition at tau=0 under the true (all-open) policy."""
        P, A = self.data.n_patches, self.mcfg.n_age_groups
        tau0 = torch.zeros(1, 1, device=self.device, dtype=self.dtype)
        week0 = torch.zeros(1, device=self.device, dtype=torch.int64)
        policy0 = torch.as_tensor(
            self.sampler.true[0], dtype=self.dtype, device=self.device
        ).unsqueeze(0)
        state = self.net(tau0, week0, policy0)[0]            # (P, A, 4)
        S, E, I = state[..., 0], state[..., 1], state[..., 2]
        # normalise IC residual by N as well
        N = self.consts.N
        loss = (
            ((S - self.ic_S) / N) ** 2
            + ((E - self.ic_E) / N) ** 2
            + ((I - self.ic_I) / N) ** 2
        ).mean()
        return loss

    # ------------------------------------------------------------------ #
    # full objective
    # ------------------------------------------------------------------ #
    def total_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        sched_np = self.sampler.sample(self.tcfg.n_schedules)
        schedules = torch.as_tensor(sched_np, dtype=self.dtype, device=self.device)

        l_phys = self.loss_physics(schedules)
        l_junc = self.loss_junction(schedules)
        l_data = self.loss_data()
        l_ic = self.loss_ic()

        loss = (
            self.tcfg.w_phys * l_phys
            + self.tcfg.w_junction * l_junc
            + self.tcfg.w_data * l_data
            + self.tcfg.w_ic * l_ic
        )
        logs = {
            "loss": float(loss.detach()),
            "phys": float(l_phys.detach()),
            "junction": float(l_junc.detach()),
            "data": float(l_data.detach()),
            "ic": float(l_ic.detach()),
            "R0": float(self.r0.detach()),
            "mu": float(self.mu.detach()),
            "kappa": float(self.kappa.detach()),
            "alpha": float(self.alpha.detach()),
        }
        return loss, logs

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def train(self) -> Dict[str, float]:
        params = self.trainable_parameters()
        opt = torch.optim.Adam(params, lr=self.tcfg.adam_lr)
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.tcfg.lr_decay_step, gamma=self.tcfg.lr_decay_gamma
        )

        t0 = time.time()
        last_logs: Dict[str, float] = {}
        pbar = tqdm(range(1, self.tcfg.adam_iters + 1), desc="Adam", unit="it")
        for it in pbar:
            opt.zero_grad(set_to_none=True)
            loss, logs = self.total_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            opt.step()
            sched.step()
            last_logs = logs
            pbar.set_postfix(
                loss=f"{logs['loss']:.3e}",
                data=f"{logs['data']:.3e}",
                R0=f"{logs['R0']:.3f}",
                mu=f"{logs['mu']:.3f}",
            )
            if it % self.tcfg.log_every == 0 or it == 1:
                dt = time.time() - t0
                tqdm.write(
                    f"[{it:>6}] loss={logs['loss']:.4e} "
                    f"phys={logs['phys']:.3e} junc={logs['junction']:.3e} "
                    f"data={logs['data']:.3e} ic={logs['ic']:.3e} | "
                    f"R0={logs['R0']:.3f} mu={logs['mu']:.3f} "
                    f"kappa={logs['kappa']:.3f} alpha={logs['alpha']:.3f} "
                    f"({dt:.0f}s)"
                )

        if self.tcfg.lbfgs_iters > 0:
            self._polish_lbfgs(params)
            _, last_logs = self.total_loss()

        return last_logs

    def _polish_lbfgs(self, params) -> None:
        opt = torch.optim.LBFGS(
            params,
            max_iter=self.tcfg.lbfgs_iters,
            line_search_fn="strong_wolfe",
            history_size=50,
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            loss, _ = self.total_loss()
            loss.backward()
            return loss

        print("L-BFGS polish...")
        opt.step(closure)

    # ------------------------------------------------------------------ #
    # prediction / export
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_nation_incidence(self, policy: np.ndarray = None) -> np.ndarray:
        """Predicted weekly ILI-per-100k for each nation under a given POLICY.

        policy: (n_weeks, P) in {0,1}; defaults to the true 2009 policy (all-open).
        Returns (R, n_weeks).
        """
        if policy is None:
            policy = self.sampler.true
        n_weeks, P = policy.shape
        nq = max(self.tcfg.n_collocation, 16)
        pol = torch.as_tensor(policy, dtype=self.dtype, device=self.device)
        tau_nodes = torch.linspace(0, 1, nq, device=self.device, dtype=self.dtype)
        tau_grid = tau_nodes.repeat(n_weeks).unsqueeze(1)
        weeks = torch.repeat_interleave(torch.arange(n_weeks, device=self.device), nq)
        pol_in = pol[weeks]
        state = self.net(tau_grid, weeks, pol_in)
        E = state[..., 1].reshape(n_weeks, nq, P, self.mcfg.n_age_groups)
        tau_week = tau_nodes.unsqueeze(0).expand(n_weeks, nq)
        pred = nation_weekly_incidence(
            self.consts, E, tau_week, self.membership,
            self.nation_population, self.alpha, self.mcfg.observation_scale,
        )
        return pred.cpu().numpy()

    @torch.no_grad()
    def predict_nation_incidence_daily(
        self, policy: np.ndarray = None, samples_per_day: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicted DAILY ILI incidence rate per nation, as a continuous curve.

        Evaluates the network at ``samples_per_day`` points within each day of every
        week and reports the daily incidence rate per 100k:
            rate_day = alpha * observation_scale * (zeta * sum_{p in r, age} E) / N_r,
        i.e. new symptomatic infections per 100k per DAY. Integrating this over the 7
        days of a week recovers the weekly incidence used in the data loss.

        Returns
        -------
        t_days : (n_days,) time in WEEK units (day d -> d/7), for plotting on the same
                 axis as the weekly observations.
        rate   : (R, n_days) daily incidence rate per 100k per day for each nation.
        """
        if policy is None:
            policy = self.sampler.true
        n_weeks, P = policy.shape
        dpw = int(self.mcfg.days_per_week)
        A = self.mcfg.n_age_groups
        spd = max(1, samples_per_day)

        pol = torch.as_tensor(policy, dtype=self.dtype, device=self.device)

        # Build the per-day sample grid: for each week, dpw*spd points across [0,1).
        # Place samples at day-centres (offset by (j+0.5)/spd within each day).
        day_idx = torch.arange(dpw, device=self.device)
        sub = (torch.arange(spd, device=self.device) + 0.5) / spd
        # tau for each (day, sub): (day + sub)/dpw
        tau_day = ((day_idx.unsqueeze(1) + sub.unsqueeze(0)) / dpw).reshape(-1)  # (dpw*spd,)
        n_per_week = tau_day.shape[0]

        tau_grid = tau_day.repeat(n_weeks).unsqueeze(1)                          # (n_weeks*npw,1)
        weeks = torch.repeat_interleave(torch.arange(n_weeks, device=self.device), n_per_week)
        pol_in = pol[weeks]
        state = self.net(tau_grid, weeks, pol_in)
        E = state[..., 1]                                                        # (n_weeks*npw, P, A)

        # Aggregate to nations: new infections/day = zeta * sum_age E, sum over patch->nation.
        flux = self.consts.zeta * E.sum(dim=-1)                                  # (rows, P)
        nation_counts = torch.matmul(flux, self.membership.t())                  # (rows, R)
        rate = (
            self.alpha * self.mcfg.observation_scale
            * nation_counts / self.nation_population.unsqueeze(0)
        )                                                                        # (rows, R)

        rate = rate.reshape(n_weeks, n_per_week, -1)                             # (n_weeks, npw, R)
        rate = rate.permute(2, 0, 1).reshape(rate.shape[2], -1)                  # (R, n_weeks*npw)

        # Absolute time in WEEK units for each sample.
        t_days = (
            torch.arange(n_weeks, device=self.device).unsqueeze(1) + tau_day.unsqueeze(0)
        ).reshape(-1)                                                            # (n_weeks*npw,)
        return t_days.cpu().numpy(), rate.cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(
            {
                "net": self.net.state_dict(),
                "raw_r0": self.raw_r0.detach(),
                "raw_mu": self.raw_mu.detach(),
                "raw_kappa": self.raw_kappa.detach(),
                "raw_alpha": self.raw_alpha.detach(),
                "district_names": self.data.district_names,
                "nation_names": self.data.nation_names,
                # architecture needed to rebuild an identical network on reload
                "arch": {
                    "hidden_layers": self.tcfg.hidden_layers,
                    "hidden_width": self.tcfg.hidden_width,
                    "activation": self.tcfg.activation,
                    "initializer": self.tcfg.initializer,
                    "week_embed_dim": self.tcfg.week_embed_dim,
                },
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Restore network weights and fitted parameters from a saved checkpoint.

        The fixed data arrays are not stored in the checkpoint; rebuild the trainer
        from the same ModelConfig/data files first, then call this to load weights.
        If the checkpoint records a different architecture than this trainer was built
        with, the network is rebuilt to match before loading.
        """
        ckpt = torch.load(path, map_location=self.device)

        arch = ckpt.get("arch")
        if arch is not None:
            mismatched = any(
                getattr(self.tcfg, k) != v for k, v in arch.items()
            )
            if mismatched:
                print(
                    "Rebuilding network to match checkpoint architecture "
                    f"({arch['hidden_layers']}x{arch['hidden_width']}, "
                    f"embed={arch['week_embed_dim']})."
                )
                for k, v in arch.items():
                    setattr(self.tcfg, k, v)
                self.net = SEIRPINN(
                    n_patches=self.data.n_patches,
                    n_age_groups=self.mcfg.n_age_groups,
                    n_weeks=self.mcfg.n_weeks,
                    N=torch.as_tensor(self.data.N, dtype=self.dtype, device=self.device),
                    cfg=self.tcfg,
                    use_budget=False,
                ).to(self.device)

        self.net.load_state_dict(ckpt["net"])
        with torch.no_grad():
            for name in ("raw_r0", "raw_mu", "raw_kappa", "raw_alpha"):
                getattr(self, name).copy_(ckpt[name].to(self.device))
        if ckpt.get("district_names") != self.data.district_names:
            print(
                "WARNING: checkpoint district ordering differs from the rebuilt data; "
                "predictions may be misaligned. Ensure the same data files and subset."
            )