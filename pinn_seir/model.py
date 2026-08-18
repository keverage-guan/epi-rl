"""Training driver: builds the loss terms and jointly optimises theta + parameters.

Loss = w_phys * L_phys + w_junction * L_junction + w_data * L_data + w_ic * L_ic
       [+ alpha_prior_weight * (alpha - alpha_prior_mean)^2 when alpha is trained]

  * L_phys      SEIAR + mean-field residual, over sampled schedules and all weeks.
  * L_junction  overlapping-strip state-continuity across week boundaries.
  * L_data      weekly ILI SYMPTOMATIC incidence aggregated to nations, true calendar.
  * L_ic        week-1 initial condition (Falkirk seed, S ~ N elsewhere, I=A=R=0).

The network output already enforces S+E+I+A+R = N exactly (softmax head), so there is
no conservation penalty. Time-derivatives are obtained by autodiff of the network
w.r.t. the local-time coordinate tau.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

from .config import ModelConfig, TrainConfig
from .data import EpiData
from .network import SEIRPINN, N_COMPARTMENTS
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
            f_sym=mcfg.f_sym,
            r_asym=mcfg.r_asym,
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
        # Characteristic scale for the data loss. Pinned via TrainConfig.data_scale
        # when comparing across runs that load different observation files.
        if tcfg.data_scale is not None:
            self._data_scale = float(tcfg.data_scale)
        else:
            self._data_scale = float((data.y_obs ** 2).mean()) + 1e-8

        # ---- held-out dev observations (evaluation only, never in a loss) -- #
        self.has_dev = data.y_obs_dev is not None
        if self.has_dev:
            self.y_obs_dev = T(data.y_obs_dev)
            self.obs_week_index_dev = torch.as_tensor(
                data.obs_week_index_dev, dtype=torch.int64, device=self.device
            )

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

        # Per-nation holiday mask over model weeks, used by every held-out metric
        # (dev now, test later). Derived from the daily calendar, NOT from the
        # is_holiday column in the split files, so the two can be cross-checked.
        term_frac = self.calendar_table.mean(dim=2)              # (P, n_weeks)
        memb = self.membership                                    # (R, P)
        per_nation = (memb @ term_frac) / memb.sum(dim=1, keepdim=True)
        self._week_is_holiday = (per_nation < 0.5)                # (R, n_weeks)

        # ---- regime switch (containment -> treatment) --------------------- #
        if mcfg.regime_switch_date is None:
            self.switch_day = None
        else:
            d0 = datetime.strptime(mcfg.epidemic_start, "%Y-%m-%d").date()
            ds = datetime.strptime(mcfg.regime_switch_date, "%Y-%m-%d").date()
            self.switch_day = (ds - d0).days
            horizon = int(mcfg.n_weeks * mcfg.days_per_week)
            if not (0 < self.switch_day < horizon):
                raise ValueError(
                    f"regime_switch_date {mcfg.regime_switch_date} maps to model day "
                    f"{self.switch_day}, outside the fitted horizon [1, {horizon - 1}] "
                    f"(epidemic_start={mcfg.epidemic_start}, n_weeks={mcfg.n_weeks}). "
                    f"A switch at or before day 0 makes the pre-switch parameters "
                    f"unidentifiable; one past the horizon makes the post-switch "
                    f"parameters unidentifiable. Either would train to completion and "
                    f"report meaningless values."
                )
            print(
                f"  regime switch: {mcfg.regime_switch_date} = model day "
                f"{self.switch_day} (week {self.switch_day // int(mcfg.days_per_week)}, "
                f"day {self.switch_day % int(mcfg.days_per_week)} of that week)"
            )

        self.raw_r0_post    = self._make_param(mcfg.r0_post_init,    mcfg.train_r0_post,    positive=True)
        self.raw_gamma      = self._make_param(mcfg.gamma,           mcfg.train_gamma,      positive=True)
        self.raw_gamma_post = self._make_param(mcfg.gamma_post_init, mcfg.train_gamma_post, positive=True)

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

    def _regime(self, tau, week):
        """1.0 for post-switch collocation points, 0.0 for pre. Returns (B,1).

        Uses the same floor-to-day convention as _term_time, so the switch is
        resolved at daily resolution inside week 9 -- no weekly threshold.
        """
        if self.switch_day is None:
            return torch.zeros_like(tau)
        dpw = int(self.mcfg.days_per_week)
        day = week.to(torch.int64) * dpw + (tau.squeeze(-1) * dpw).floor().long()
        return (day >= self.switch_day).to(tau.dtype).unsqueeze(-1)

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

    @property
    def r0_post(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_r0_post)

    @property
    def gamma_pre(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_gamma)

    @property
    def gamma_post(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_gamma_post)

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
        """S ~ N everywhere, a small seed of exposed adults in the seed district, I=A=R=0."""
        P, A = self.data.n_patches, self.mcfg.n_age_groups
        N = self.data.N
        S0 = N.copy().astype(np.float64)
        E0 = np.zeros_like(S0)
        I0 = np.zeros_like(S0)
        A0 = np.zeros_like(S0)
        adult = self.mcfg.adult_index
        seed = self.data.seed_district_index
        E0[seed, adult] = self.mcfg.seed_exposed_count
        S0[seed, adult] = max(N[seed, adult] - self.mcfg.seed_exposed_count, 0.0)

        def T(a):
            return torch.as_tensor(a, dtype=self.dtype, device=self.device)

        self.ic_S = T(S0)
        self.ic_E = T(E0)
        self.ic_I = T(I0)
        self.ic_A = T(A0)

    # ------------------------------------------------------------------ #
    # forward helper: evaluate net + d/dtau at given (tau, week, closure)
    # ------------------------------------------------------------------ #
    def _forward_with_dt(
        self, tau: torch.Tensor, week: torch.Tensor, closure: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (state, d state / d(week)), both (B, P, A, 5)."""
        # Reverse-mode fallback: one grad call per output column. Correct but
        # O(P*A*5) backward passes; only used if forward-mode AD is unavailable.
        tau = tau.clone().requires_grad_(True)
        state = self.net(tau, week, closure)                 # (B, P, A, 5)
        B = state.shape[0]
        flat = state.reshape(B, -1)                          # (B, P*A*5)
        grads = torch.zeros_like(flat)
        ones = torch.ones(B, device=self.device, dtype=self.dtype)
        for j in range(flat.shape[1]):
            g = torch.autograd.grad(
                flat[:, j], tau, grad_outputs=ones, retain_graph=True, create_graph=True
            )[0]                                             # (B, 1)
            grads[:, j] = g.squeeze(1)
        dstate = grads.reshape(
            B, self.data.n_patches, self.mcfg.n_age_groups, N_COMPARTMENTS
        )
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
    def loss_physics(self, schedules):
        Ns, n_weeks, P = schedules.shape
        nc = self.tcfg.n_collocation
        total = torch.zeros((), device=self.device, dtype=self.dtype)

        for s in range(Ns):
            tau = torch.rand(n_weeks * nc, 1, device=self.device, dtype=self.dtype)
            weeks = torch.repeat_interleave(torch.arange(n_weeks, device=self.device), nc)
            policy = schedules[s][weeks]

            state, dstate = self._time_derivative(tau, weeks, policy)
            term = self._term_time(tau, weeks)
            effective_open = term * policy

            # --- regime-dependent parameters, per collocation point ---
            g = self._regime(tau, weeks)                       # (B,1)
            r0_eff    = (1.0 - g) * self.r0        + g * self.r0_post
            gamma_eff = (1.0 - g) * self.gamma_pre + g * self.gamma_post
            beta_p = beta_per_patch(self.consts, r0_eff, gamma_eff)   # (B,P)

            res = seir_residuals(
                self.consts, state, dstate, beta_p, gamma_eff,
                self.mu, self.kappa, effective_open,
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

                u_k = self.net(tau_k, wk, ck)                # (nq, P, A, 5)
                u_k1 = self.net(tau_k1, wk1, ck1)
                w = _junction_weight(off, kind, delta)       # (nq, 1)
                # continuity in FRACTIONS of N, matching the physics/IC normalisation
                N = self.consts.N.unsqueeze(0).unsqueeze(-1)  # (1, P, A, 1)
                diff = ((u_k - u_k1) / N).reshape(nq, -1)
                total = total + (w * (diff ** 2)).mean()
                count += 1
        return total / max(count, 1)

    def _predict_nation_weekly(self) -> torch.Tensor:
        """(R, n_weeks) ascertained symptomatic incidence under the true policy.

        Shared by the training data loss and the dev evaluation so the two can
        never drift apart.
        """
        P, A = self.data.n_patches, self.mcfg.n_age_groups
        n_weeks = self.mcfg.n_weeks
        nq = self.tcfg.n_collocation

        true_policy = torch.as_tensor(
            self.sampler.true, dtype=self.dtype, device=self.device
        )

        tau_nodes = torch.linspace(0, 1, nq, device=self.device, dtype=self.dtype)
        tau_grid = tau_nodes.repeat(n_weeks).unsqueeze(1)
        weeks = torch.repeat_interleave(
            torch.arange(n_weeks, device=self.device), nq
        )
        policy = true_policy[weeks]
        state = self.net(tau_grid, weeks, policy)
        E = state[..., 1].reshape(n_weeks, nq, P, A)
        tau_week = tau_nodes.unsqueeze(0).expand(n_weeks, nq)

        return nation_weekly_incidence(
            self.consts, E, tau_week, self.membership,
            self.nation_population, self.alpha, self.mcfg.observation_scale,
        )

    def loss_data(self) -> torch.Tensor:
        """Weekly ILI incidence per nation vs. the TRAINING observations."""
        pred = self._predict_nation_weekly()
        pred_obs = pred[:, self.obs_week_index]
        return ((pred_obs - self.y_obs) ** 2).mean() / self._data_scale

    @torch.no_grad()
    def evaluate_on(
        self, y_obs, week_index, prefix: str = "dev"
    ) -> Dict[str, float]:
        """Held-out metrics on any observation set. Toggles dropout off and back.

        `<prefix>_data` uses the same normalisation as the training data loss, so the
        two curves are directly comparable. `<prefix>_rmse` is in per-100k units and is
        independent of _data_scale, so it stays comparable across runs that pinned
        different scales.
        """
        was_training = self.net.training
        self.net.eval()
        try:
            pred = self._predict_nation_weekly()
            pred_obs = pred[:, week_index]                        # (R, n_obs)
            err2 = (pred_obs - y_obs) ** 2

            out = {
                f"{prefix}_data": float(err2.mean() / self._data_scale),
                f"{prefix}_rmse": float(err2.mean().sqrt()),
            }

            # Per-nation, so one nation blowing up is visible.
            for r, name in enumerate(self.data.nation_names):
                out[f"{prefix}_rmse_{name.lower()}"] = float(err2[r].mean().sqrt())

            # Holiday / term-time breakdown. With ~2 holiday weeks in a 15% slice, a
            # single bad week dominates the aggregate; always read these two together.
            hol = self._week_is_holiday[:, week_index]            # (R, n_obs)
            for label, mask in (("holiday", hol), ("term", ~hol)):
                n = int(mask.sum())
                out[f"{prefix}_rmse_{label}"] = (
                    float(err2[mask].mean().sqrt()) if n else float("nan")
                )
                out[f"{prefix}_n_{label}"] = n
            return out
        finally:
            if was_training:
                self.net.train()

    def evaluate_dev(self) -> Dict[str, float]:
        return self.evaluate_on(self.y_obs_dev, self.obs_week_index_dev, "dev")

    def loss_ic(self) -> torch.Tensor:
        """Week-1 (index 0) initial condition at tau=0 under the true (all-open) policy."""
        P, A = self.data.n_patches, self.mcfg.n_age_groups
        tau0 = torch.zeros(1, 1, device=self.device, dtype=self.dtype)
        week0 = torch.zeros(1, device=self.device, dtype=torch.int64)
        policy0 = torch.as_tensor(
            self.sampler.true[0], dtype=self.dtype, device=self.device
        ).unsqueeze(0)
        state = self.net(tau0, week0, policy0)[0]            # (P, A, 5)
        S, E, I, Asym = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
        # normalise IC residual by N as well
        N = self.consts.N
        loss = (
            ((S - self.ic_S) / N) ** 2
            + ((E - self.ic_E) / N) ** 2
            + ((I - self.ic_I) / N) ** 2
            + ((Asym - self.ic_A) / N) ** 2
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
        l_phys = self.loss_physics(schedules)
        l_junc = self.loss_junction(schedules)
        if self.tcfg.w_data == 0.0:
            l_data = torch.zeros((), device=self.device, dtype=self.dtype)
        else:
            l_data = self.loss_data()
        l_ic = self.loss_ic()

        loss = (
            self.tcfg.w_phys * l_phys
            + self.tcfg.w_junction * l_junc
            + self.tcfg.w_data * l_data
            + self.tcfg.w_ic * l_ic
        )

        # Soft prior on alpha to break the alpha/R0 scale degeneracy: only active when
        # alpha is trainable and the prior weight is positive.
        l_alpha_prior = torch.zeros((), device=self.device, dtype=self.dtype)
        if self.raw_alpha.requires_grad and self.tcfg.alpha_prior_weight > 0.0:
            l_alpha_prior = (self.alpha - self.tcfg.alpha_prior_mean) ** 2
            loss = loss + self.tcfg.alpha_prior_weight * l_alpha_prior

        logs = {
            "loss": float(loss.detach()),
            "phys": float(l_phys.detach()),
            "junction": float(l_junc.detach()),
            "data": float(l_data.detach()),
            "ic": float(l_ic.detach()),
            "alpha_prior": float(l_alpha_prior.detach()),
            "R0": float(self.r0.detach()),
            "mu": float(self.mu.detach()),
            "kappa": float(self.kappa.detach()),
            "alpha": float(self.alpha.detach()),
            "R0_post": float(self.r0_post.detach()),
            "gamma": float(self.gamma_pre.detach()),
            "gamma_post": float(self.gamma_post.detach()),
        }
        return loss, logs

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def train(self, best_checkpoint_path: Optional[str] = None) -> Dict[str, float]:
        params = self.trainable_parameters()
        opt = torch.optim.Adam(params, lr=self.tcfg.adam_lr)
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.tcfg.lr_decay_step, gamma=self.tcfg.lr_decay_gamma
        )

        t0 = time.time()
        last_logs: Dict[str, float] = {}
        self.best_iter: Optional[int] = None
        self.best_dev: float = float("inf")
        self.best_dev_logs: Dict[str, float] = {}
        stale_evals = 0
        # Dropout active during training (regularisation + MC-consistent physics).
        self.net.train()
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
                R0p=f"{logs['R0_post']:.3f}",
            )
            if it % self.tcfg.log_every == 0 or it == 1:
                dt = time.time() - t0
                dev_str = ""
                if self.has_dev:
                    dev = self.evaluate_dev()
                    dev_str = (f"dev={dev['dev_data']:.3e} "
                               f"dev_rmse={dev['dev_rmse']:.3e} "
                               f"dev_rmse_hol={dev['dev_rmse_holiday']:.3e} "
                               f"dev_rmse_term={dev['dev_rmse_term']:.3e} ")
                    if dev["dev_data"] < self.best_dev:
                        self.best_dev = dev["dev_data"]
                        self.best_iter = it
                        self.best_dev_logs = dict(dev)
                        stale_evals = 0
                        if best_checkpoint_path is not None:
                            self.save(best_checkpoint_path)
                    else:
                        stale_evals += 1
                tqdm.write(
                    f"[{it:>6}] loss={logs['loss']:.4e} "
                    f"phys={logs['phys']:.3e} junc={logs['junction']:.3e} "
                    f"data={logs['data']:.3e} ic={logs['ic']:.3e} "
                    f"{dev_str}| "
                    f"R0={logs['R0']:.3f} mu={logs['mu']:.3f} "
                    f"kappa={logs['kappa']:.3f} alpha={logs['alpha']:.3f} "
                    f"R0_post={logs['R0_post']:.3f} "
                    f"gamma={logs['gamma']:.4f} gamma_post={logs['gamma_post']:.4f} "
                    f"({dt:.0f}s)"
                )
                if (self.tcfg.early_stop_patience > 0
                        and stale_evals >= self.tcfg.early_stop_patience):
                    tqdm.write(
                        f"Early stop at {it}: {stale_evals} evaluations without a "
                        f"dev improvement (best {self.best_dev:.4e} @ {self.best_iter})."
                    )
                    break

        if self.tcfg.lbfgs_iters > 0:
            self._polish_lbfgs(params)
            _, last_logs = self.total_loss()
            # The polish runs after the Adam loop and was never dev-checked, so
            # checkpoint.pt could be worse than checkpoint_best.pt. Check once.
            if self.has_dev:
                dev = self.evaluate_dev()
                tqdm.write(
                    f"[post-LBFGS] dev={dev['dev_data']:.4e} "
                    f"(best during Adam {self.best_dev:.4e} @ {self.best_iter})"
                )
                if dev["dev_data"] < self.best_dev:
                    self.best_dev = dev["dev_data"]
                    self.best_iter = -1          # -1 = post-polish
                    self.best_dev_logs = dict(dev)
                    if best_checkpoint_path is not None:
                        self.save(best_checkpoint_path)

        # Resting state after training is deterministic: dropout OFF. MC samplers turn
        # it back on temporarily via net.mc_dropout(); plain predictors stay stochastic-
        # free. load_checkpoint also leaves the net in eval mode.
        self.net.eval()

        if self.has_dev:
            last_logs = dict(last_logs)
            last_logs["best_iter"] = self.best_iter
            last_logs["best_dev_data"] = self.best_dev
            last_logs.update({f"best_{k}": v for k, v in self.best_dev_logs.items()})

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
        Returns (R, n_weeks). This is ASCERTAINED SYMPTOMATIC incidence
        (alpha * f_sym * integral zeta*E).
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
            rate_day = alpha * f_sym * observation_scale * (zeta * sum_{p in r, age} E) / N_r,
        i.e. new ASCERTAINED SYMPTOMATIC infections per 100k per DAY. Integrating this
        over the 7 days of a week recovers the weekly incidence used in the data loss.

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

        # Aggregate to nations: new SYMPTOMATIC infections/day = f_sym * zeta * sum_age E,
        # sum over patch -> nation; only alpha of those are ascertained.
        flux = self.consts.f_sym * self.consts.zeta * E.sum(dim=-1)             # (rows, P)
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

    # ------------------------------------------------------------------ #
    # Monte-Carlo dropout sampling
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_nation_incidence(
        self,
        n_samples: int,
        policy: np.ndarray = None,
        daily: bool = False,
        samples_per_day: int = 2,
    ) -> np.ndarray:
        """Draw MC-dropout samples of the per-nation incidence curve.

        Keeps the trunk's dropout layers active and repeats the forward pass
        ``n_samples`` times, yielding a distribution over incidence trajectories that
        approximates the network's epistemic uncertainty. Requires the model to have
        been trained with ``dropout_rate > 0`` (otherwise every sample is identical and
        a warning is issued).

        Parameters
        ----------
        n_samples : number of stochastic forward passes.
        policy    : (n_weeks, P) closure policy; defaults to the true 2009 all-open.
        daily     : if True sample the daily-resolution curve, else weekly incidence.
        samples_per_day : passed through to the daily predictor when ``daily``.

        Returns
        -------
        np.ndarray
            weekly:  (n_samples, R, n_weeks)
            daily :  (n_samples, R, n_days)   (the shared time axis is available from
                     :meth:`predict_nation_incidence_daily`).
        """
        if self.net.dropout_rate <= 0.0:
            print(
                "WARNING: dropout_rate is 0, so MC-dropout samples are all identical. "
                "Retrain with TrainConfig.dropout_rate > 0 (e.g. --dropout-rate 0.05)."
            )

        samples = []
        with self.net.mc_dropout():          # dropout layers -> stochastic (train mode)
            for _ in range(n_samples):
                if daily:
                    _, rate = self.predict_nation_incidence_daily(
                        policy=policy, samples_per_day=samples_per_day
                    )
                else:
                    rate = self.predict_nation_incidence(policy=policy)
                samples.append(rate)
        return np.stack(samples, axis=0)

    @torch.no_grad()
    def sample_state_trajectories(
        self,
        n_samples: int,
        policy: np.ndarray = None,
        samples_per_day: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """MC-dropout samples of the full SEIAR compartment trajectories, per nation.

        Evaluates the network on a daily grid with dropout active and aggregates the
        five compartments (S,E,I,A,R) from districts up to nations for each sample.

        Returns
        -------
        t_days : (n_days,) time in WEEK units (day d -> d/7).
        traj   : (n_samples, R, n_days, 5) nation-level compartment counts per sample,
                 in the order S, E, I, A, R.
        """
        if self.net.dropout_rate <= 0.0:
            print(
                "WARNING: dropout_rate is 0, so MC-dropout samples are all identical. "
                "Retrain with TrainConfig.dropout_rate > 0."
            )

        if policy is None:
            policy = self.sampler.true
        n_weeks, P = policy.shape
        dpw = int(self.mcfg.days_per_week)
        A = self.mcfg.n_age_groups
        spd = max(1, samples_per_day)

        pol = torch.as_tensor(policy, dtype=self.dtype, device=self.device)

        day_idx = torch.arange(dpw, device=self.device)
        sub = (torch.arange(spd, device=self.device) + 0.5) / spd
        tau_day = ((day_idx.unsqueeze(1) + sub.unsqueeze(0)) / dpw).reshape(-1)
        n_per_week = tau_day.shape[0]
        tau_grid = tau_day.repeat(n_weeks).unsqueeze(1)
        weeks = torch.repeat_interleave(
            torch.arange(n_weeks, device=self.device), n_per_week
        )
        pol_in = pol[weeks]

        t_days = (
            torch.arange(n_weeks, device=self.device).unsqueeze(1) + tau_day.unsqueeze(0)
        ).reshape(-1).cpu().numpy()

        # (R, P) membership for aggregating district counts to nations.
        membership = self.membership                          # (R, P)

        out = []
        with self.net.mc_dropout():
            for _ in range(n_samples):
                state = self.net(tau_grid, weeks, pol_in)     # (rows, P, A, 5)
                # sum over age, aggregate patches -> nations, per compartment
                by_patch = state.sum(dim=2)                    # (rows, P, 5)
                # (R, P) @ (rows, P, 5) -> (rows, R, 5)
                nat = torch.einsum("rp,bpc->brc", membership, by_patch)  # (rows, R, 5)
                out.append(nat.cpu().numpy())
        traj = np.stack(out, axis=0)                          # (n_samples, rows, R, 5)
        traj = np.transpose(traj, (0, 2, 1, 3))               # (n_samples, R, rows, 5)
        return t_days, traj

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
                    "dropout_rate": self.tcfg.dropout_rate,
                },
                # compartment count, so a reload can detect an incompatible (4-comp)
                # checkpoint from before the asymptomatic split.
                "n_compartments": N_COMPARTMENTS,
                # fixed structural constants that change the observation/physics meaning
                "obs_params": {
                    "f_sym": self.mcfg.f_sym,
                    "r_asym": self.mcfg.r_asym,
                },
            },
            path,
        )

    def load_checkpoint(self, path: str, strict_arch: bool = False) -> None:
        """Restore network weights and fitted parameters from a saved checkpoint.

        The fixed data arrays are not stored in the checkpoint; rebuild the trainer
        from the same ModelConfig/data files first, then call this to load weights.
        If the checkpoint records a different architecture than this trainer was built
        with, the network is rebuilt to match before loading.
        """
        ckpt = torch.load(path, map_location=self.device)

        ckpt_nc = ckpt.get("n_compartments", 4)
        if ckpt_nc != N_COMPARTMENTS:
            raise ValueError(
                f"Checkpoint has {ckpt_nc} compartments but this model uses "
                f"{N_COMPARTMENTS} (S,E,I,A,R). This checkpoint predates the "
                "asymptomatic split and cannot be loaded; retrain from scratch."
            )

        arch = ckpt.get("arch")
        if arch is not None:
            mismatched = any(
                getattr(self.tcfg, k) != v for k, v in arch.items()
            )
            if mismatched and strict_arch:
                diff = {k: (getattr(self.tcfg, k), v)
                        for k, v in arch.items() if getattr(self.tcfg, k) != v}
                raise ValueError(
                    "Architecture mismatch under strict loading (current, checkpoint): "
                    f"{diff}. A warm start must use an identical network; rerun with "
                    "matching --hidden-layers/--hidden-width/--dropout-rate."
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

        # Warn if the checkpoint was fit with different fixed observation constants,
        # since alpha/attack-rate interpretation depends on f_sym.
        obs = ckpt.get("obs_params")
        if obs is not None:
            if abs(obs.get("f_sym", self.mcfg.f_sym) - self.mcfg.f_sym) > 1e-9 or \
               abs(obs.get("r_asym", self.mcfg.r_asym) - self.mcfg.r_asym) > 1e-9:
                print(
                    "WARNING: checkpoint f_sym/r_asym differ from the current config "
                    f"(ckpt f_sym={obs.get('f_sym')}, r_asym={obs.get('r_asym')} vs "
                    f"config f_sym={self.mcfg.f_sym}, r_asym={self.mcfg.r_asym}). "
                    "Predicted incidence and alpha interpretation will not match the fit."
                )

        self.net.load_state_dict(ckpt["net"])
        # Deterministic by default after reload; MC samplers re-enable dropout locally.
        self.net.eval()
        with torch.no_grad():
            for name in ("raw_r0", "raw_mu", "raw_kappa", "raw_alpha"):
                getattr(self, name).copy_(ckpt[name].to(self.device))
        if ckpt.get("district_names") != self.data.district_names:
            print(
                "WARNING: checkpoint district ordering differs from the rebuilt data; "
                "predictions may be misaligned. Ensure the same data files and subset."
            )