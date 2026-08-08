from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

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
        self.membership = T(data.nation_membership)
        self.nation_population = T(data.nation_population)
        self.y_obs = T(data.y_obs)
        self.obs_week_index = torch.as_tensor(
            data.obs_week_index, dtype=torch.int64, device=self.device
        )
        self._data_scale = float((data.y_obs ** 2).mean()) + 1e-8

        obs_per_week = np.zeros(mcfg.n_weeks, dtype=np.float64)
        obs_per_week[data.obs_week_index] = np.clip(data.y_obs.sum(axis=0), 0.0, None)
        peak = obs_per_week.max()
        obs_per_week = obs_per_week / peak if peak > 0 else obs_per_week
        self.week_activity_prior = T(obs_per_week)

        total_symptomatic = float(
            (data.y_obs.sum(axis=1) * data.nation_population).sum()
        ) / (mcfg.alpha_init * mcfg.observation_scale)
        total_infections = total_symptomatic / mcfg.f_sym
        self._target_cumulative_R = T(total_infections)
        self._last_obs_week = int(data.obs_week_index.max())

        self.net = SEIRPINN(
            n_patches=P,
            n_age_groups=A,
            n_weeks=mcfg.n_weeks,
            N=T(data.N),
            cfg=tcfg,
            use_budget=False,
        ).to(self.device)

        self.raw_r0 = self._make_param(mcfg.r0_init, mcfg.train_r0, positive=True)
        self.raw_mu = self._make_param(mcfg.mu_init, mcfg.train_mu, unit=True)
        self.raw_kappa = self._make_param(mcfg.kappa_init, mcfg.train_kappa, positive=True)
        self.raw_alpha = self._make_param(mcfg.alpha_init, mcfg.train_alpha, unit=True)

        self.sampler = ScheduleSampler(
            n_weeks=mcfg.n_weeks,
            n_patches=P,
            budget_weeks=tcfg.budget_weeks,
            include_all_open=tcfg.include_all_open,
            include_all_closed=tcfg.include_all_closed,
            seed=tcfg.seed,
        )
        self.calendar = DailyCalendar(
            epidemic_start=mcfg.epidemic_start,
            holiday_ranges_by_nation=mcfg.holiday_ranges_by_nation,
            district_nations=data.district_nations,
            n_weeks=mcfg.n_weeks,
            days_per_week=int(mcfg.days_per_week),
        )
        self.calendar_table = torch.as_tensor(
            self.calendar.table, dtype=self.dtype, device=self.device
        )
        self._build_ic_targets()

    def _term_time(self, tau: torch.Tensor, week: torch.Tensor) -> torch.Tensor:
        dpw = self.calendar_table.shape[-1]
        day = torch.clamp((tau.squeeze(-1) * dpw).floor().long(), 0, dpw - 1)
        term = self.calendar_table[:, week, day]
        return term.t()

    def _make_param(self, init: float, trainable: bool, positive=False, unit=False):
        if positive:
            raw = float(np.log(np.expm1(max(init, 1e-4))))
        elif unit:
            init = min(max(init, 1e-4), 1 - 1e-4)
            raw = float(np.log(init / (1 - init)))
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

    def _build_ic_targets(self) -> None:
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

    def _forward_with_dt(
        self, tau: torch.Tensor, week: torch.Tensor, closure: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tau = tau.clone().requires_grad_(True)
        state = self.net(tau, week, closure)
        B = state.shape[0]
        flat = state.reshape(B, -1)
        grads = torch.zeros_like(flat)
        ones = torch.ones(B, device=self.device, dtype=self.dtype)
        for j in range(flat.shape[1]):
            g = torch.autograd.grad(
                flat[:, j], tau, grad_outputs=ones, retain_graph=True, create_graph=True
            )[0]
            grads[:, j] = g.squeeze(1)
        dstate = grads.reshape(
            B, self.data.n_patches, self.mcfg.n_age_groups, N_COMPARTMENTS
        )
        return state, dstate

    def _forward_with_dt_fast(
        self, tau: torch.Tensor, week: torch.Tensor, closure: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from torch.func import jvp

        def f(tau_in):
            return self.net(tau_in, week, closure)

        tangent = torch.ones_like(tau)
        state, dstate = jvp(f, (tau,), (tangent,))
        return state, dstate

    def _time_derivative(self, tau, week, closure):
        try:
            return self._forward_with_dt_fast(tau, week, closure)
        except Exception:
            return self._forward_with_dt(tau, week, closure)

    def loss_physics(self, schedules: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Ns, n_weeks, P = schedules.shape
        A = self.mcfg.n_age_groups
        nc = self.tcfg.n_collocation

        beta_p = beta_per_patch(self.consts, self.r0)
        floor = self.tcfg.phys_activity_floor
        week_weight_table = self.week_activity_prior + floor
        week_weight_table = week_weight_table / week_weight_table.mean()
        total = torch.zeros((), device=self.device, dtype=self.dtype)
        total_mono = torch.zeros((), device=self.device, dtype=self.dtype)

        for s in range(Ns):
            tau = torch.rand(n_weeks * nc, 1, device=self.device, dtype=self.dtype)
            weeks = torch.repeat_interleave(
                torch.arange(n_weeks, device=self.device), nc
            )
            policy = schedules[s][weeks]
            state, dstate = self._time_derivative(tau, weeks, policy)
            term = self._term_time(tau, weeks)
            effective_open = term * policy
            res = seir_residuals(
                self.consts, state, dstate, beta_p, self.mu, self.kappa, effective_open
            )
            weight = week_weight_table[weeks].view(-1, 1, 1, 1)
            total = total + (weight * res ** 2).mean()

            N = self.consts.N.unsqueeze(0)
            dS_frac = dstate[..., 0] / N
            dR_frac = dstate[..., 4] / N
            mono = torch.relu(dS_frac).pow(2) + torch.relu(-dR_frac).pow(2)
            w2 = week_weight_table[weeks].view(-1, 1, 1)
            total_mono = total_mono + (w2 * mono).mean()
        return total / Ns, total_mono / Ns

    def loss_junction(self, schedules: torch.Tensor) -> torch.Tensor:
        Ns, n_weeks, P = schedules.shape
        delta = self.tcfg.overlap_delta
        kind = self.tcfg.junction_weight
        nq = self.tcfg.n_junction

        total = torch.zeros((), device=self.device, dtype=self.dtype)
        count = 0
        for s in range(Ns):
            for k in range(n_weeks - 1):
                off = (torch.rand(nq, 1, device=self.device, dtype=self.dtype) * 2 - 1) * delta
                tau_k = (1.0 + off).clamp(0.0, 1.0)
                tau_k1 = (off).clamp(0.0, 1.0)
                wk = torch.full((nq,), k, device=self.device, dtype=torch.int64)
                wk1 = torch.full((nq,), k + 1, device=self.device, dtype=torch.int64)
                ck = schedules[s, k].unsqueeze(0).expand(nq, P)
                ck1 = schedules[s, k + 1].unsqueeze(0).expand(nq, P)

                u_k = self.net(tau_k, wk, ck)
                u_k1 = self.net(tau_k1, wk1, ck1)
                w = _junction_weight(off, kind, delta)
                N = self.consts.N.unsqueeze(0).unsqueeze(-1)
                diff = ((u_k - u_k1) / N).reshape(nq, -1)
                total = total + (w * (diff ** 2)).mean()
                count += 1
        return total / max(count, 1)

    def loss_data(self) -> torch.Tensor:
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

        pred = nation_weekly_incidence(
            self.consts, E, tau_week, self.membership,
            self.nation_population, self.alpha, self.mcfg.observation_scale,
        )

        pred_obs = pred[:, self.obs_week_index]
        return ((pred_obs - self.y_obs) ** 2).mean() / self._data_scale

    def loss_ic(self) -> torch.Tensor:
        P, A = self.data.n_patches, self.mcfg.n_age_groups
        tau0 = torch.zeros(1, 1, device=self.device, dtype=self.dtype)
        week0 = torch.zeros(1, device=self.device, dtype=torch.int64)
        policy0 = torch.as_tensor(
            self.sampler.true[0], dtype=self.dtype, device=self.device
        ).unsqueeze(0)
        state = self.net(tau0, week0, policy0)[0]
        S, E, I, Asym = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
        N = self.consts.N
        loss = (
            ((S - self.ic_S) / N) ** 2
            + ((E - self.ic_E) / N) ** 2
            + ((I - self.ic_I) / N) ** 2
            + ((Asym - self.ic_A) / N) ** 2
        ).mean()
        return loss

    def loss_cumulative(self) -> torch.Tensor:
        true_policy = torch.as_tensor(
            self.sampler.true, dtype=self.dtype, device=self.device
        )
        tau = torch.full((1, 1), 0.99, device=self.device, dtype=self.dtype)
        week_t = torch.tensor([self._last_obs_week], dtype=torch.int64, device=self.device)
        closure_t = true_policy[self._last_obs_week].unsqueeze(0)
        state = self.net(tau, week_t, closure_t)
        R_pred = state[..., 4].sum()
        target = self._target_cumulative_R
        return ((R_pred - target) / target.clamp_min(1.0)) ** 2

    def total_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        sched_np = self.sampler.sample(self.tcfg.n_schedules)
        schedules = torch.as_tensor(sched_np, dtype=self.dtype, device=self.device)

        l_phys, l_mono = self.loss_physics(schedules)
        l_junc = self.loss_junction(schedules)
        l_data = self.loss_data()
        l_ic = self.loss_ic()
        l_cumulative = self.loss_cumulative()

        loss = (
            self.tcfg.w_phys * l_phys
            + self.tcfg.w_junction * l_junc
            + self.tcfg.w_data * l_data
            + self.tcfg.w_ic * l_ic
            + self.tcfg.w_monotonic * l_mono
            + self.tcfg.w_cumulative * l_cumulative
        )

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
            "monotonic": float(l_mono.detach()),
            "cumulative": float(l_cumulative.detach()),
            "alpha_prior": float(l_alpha_prior.detach()),
            "R0": float(self.r0.detach()),
            "mu": float(self.mu.detach()),
            "kappa": float(self.kappa.detach()),
            "alpha": float(self.alpha.detach()),
        }
        return loss, logs

    def train(self, checkpoint_dir: str = None, checkpoint_every: int = None) -> Dict[str, float]:
        params = self.trainable_parameters()
        opt = torch.optim.Adam(params, lr=self.tcfg.adam_lr)
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.tcfg.lr_decay_step, gamma=self.tcfg.lr_decay_gamma
        )

        t0 = time.time()
        last_logs: Dict[str, float] = {}
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
                alpha=f"{logs['alpha']:.3f}",
            )
            if it % self.tcfg.log_every == 0 or it == 1:
                dt = time.time() - t0
                tqdm.write(
                    f"[{it:>6}] loss={logs['loss']:.4e} "
                    f"phys={logs['phys']:.3e} junc={logs['junction']:.3e} "
                    f"data={logs['data']:.3e} ic={logs['ic']:.3e} "
                    f"mono={logs['monotonic']:.3e} cum={logs['cumulative']:.3e} | "
                    f"R0={logs['R0']:.3f} mu={logs['mu']:.3f} "
                    f"kappa={logs['kappa']:.3f} alpha={logs['alpha']:.3f} "
                    f"({dt:.0f}s)"
                )
            if checkpoint_dir and checkpoint_every and it % checkpoint_every == 0:
                was_training = self.net.training
                self.net.eval()
                ckpt_path = Path(checkpoint_dir) / f"checkpoint_iter{it:06d}.pt"
                self.save(str(ckpt_path))
                tqdm.write(f"  [checkpoint] saved {ckpt_path}")
                self.net.train(was_training)

        if self.tcfg.lbfgs_iters > 0:
            self._polish_lbfgs(params)
            _, last_logs = self.total_loss()

        self.net.eval()
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

    @torch.no_grad()
    def predict_nation_incidence(self, policy: np.ndarray = None) -> np.ndarray:
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
        weeks = torch.repeat_interleave(torch.arange(n_weeks, device=self.device), n_per_week)
        pol_in = pol[weeks]
        state = self.net(tau_grid, weeks, pol_in)
        E = state[..., 1]

        flux = self.consts.f_sym * self.consts.zeta * E.sum(dim=-1)
        nation_counts = torch.matmul(flux, self.membership.t())
        rate = (
            self.alpha * self.mcfg.observation_scale
            * nation_counts / self.nation_population.unsqueeze(0)
        )

        rate = rate.reshape(n_weeks, n_per_week, -1)
        rate = rate.permute(2, 0, 1).reshape(rate.shape[2], -1)

        t_days = (
            torch.arange(n_weeks, device=self.device).unsqueeze(1) + tau_day.unsqueeze(0)
        ).reshape(-1)
        return t_days.cpu().numpy(), rate.cpu().numpy()

    @torch.no_grad()
    def sample_nation_incidence(
        self,
        n_samples: int,
        policy: np.ndarray = None,
        daily: bool = False,
        samples_per_day: int = 2,
    ) -> np.ndarray:
        if self.net.dropout_rate <= 0.0:
            print(
                "WARNING: dropout_rate is 0, so MC-dropout samples are all identical. "
                "Retrain with TrainConfig.dropout_rate > 0 (e.g. --dropout-rate 0.05)."
            )

        samples = []
        with self.net.mc_dropout():
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

        membership = self.membership

        out = []
        with self.net.mc_dropout():
            for _ in range(n_samples):
                state = self.net(tau_grid, weeks, pol_in)
                by_patch = state.sum(dim=2)
                nat = torch.einsum("rp,bpc->brc", membership, by_patch)
                out.append(nat.cpu().numpy())
        traj = np.stack(out, axis=0)
        traj = np.transpose(traj, (0, 2, 1, 3))
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
                "arch": {
                    "hidden_layers": self.tcfg.hidden_layers,
                    "hidden_width": self.tcfg.hidden_width,
                    "activation": self.tcfg.activation,
                    "initializer": self.tcfg.initializer,
                    "week_embed_dim": self.tcfg.week_embed_dim,
                    "dropout_rate": self.tcfg.dropout_rate,
                },
                "n_compartments": N_COMPARTMENTS,
                "obs_params": {
                    "f_sym": self.mcfg.f_sym,
                    "r_asym": self.mcfg.r_asym,
                },
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
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
        self.net.eval()
        with torch.no_grad():
            for name in ("raw_r0", "raw_mu", "raw_kappa", "raw_alpha"):
                getattr(self, name).copy_(ckpt[name].to(self.device))
        if ckpt.get("district_names") != self.data.district_names:
            print(
                "WARNING: checkpoint district ordering differs from the rebuilt data; "
                "predictions may be misaligned. Ensure the same data files and subset."
            )
