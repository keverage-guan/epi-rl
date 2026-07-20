"""The policy-conditioned SEIR(A) network.

A single continuous network represents the piecewise (per-week) solution by carrying
sub-domain labels as extra inputs -- the augmented-coordinate / discontinuity-capturing
pattern. Inputs are:

    tau            local time within the week, in [0, 1]           (1)
    week embedding a learned vector for the integer week index k    (E)
    closure state  c_k in {0,1}^P                                   (P)
    budget         remaining closure budget per patch (optional)    (P)

The output is the stacked S,E,I,A,R state for every (patch, age) pair. Population
conservation S+E+I+A+R = N is enforced *exactly* by a softmax over the FIVE compartment
logits, scaled by the known N[p,i]; residuals downstream are written in fractions of N.

Dropout / MC-dropout
--------------------
The trunk is a plain PyTorch MLP (not ``dde.nn.FNN``) because the DeepXDE PyTorch
backend's FNN does not support dropout. Building the trunk here lets us insert
``nn.Dropout`` after every hidden activation, which serves two purposes:

  * regularisation during training (standard dropout), and
  * epistemic-uncertainty sampling at inference via **Monte-Carlo dropout**: keep the
    dropout layers active (in "train" mode) while everything else is frozen, and draw
    repeated stochastic forward passes to obtain a distribution over trajectories.

Use :meth:`SEIRPINN.mc_dropout` (context manager) or :meth:`set_mc_dropout` to turn the
dropout layers on for sampling without enabling training-mode behaviour elsewhere.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn as nn
import deepxde as dde

from .config import TrainConfig

# Number of compartments in the (age-structured) SEIAR model:
#   0=S  1=E  2=I (symptomatic)  3=A (asymptomatic)  4=R
N_COMPARTMENTS = 5


def _glorot_normal_(weight: torch.Tensor) -> None:
    """Glorot/Xavier normal init, matching DeepXDE's default 'Glorot normal'."""
    nn.init.xavier_normal_(weight)


class DropoutMLP(nn.Module):
    """Fully-connected trunk with Glorot-normal init and dropout after each hidden act.

    Mirrors the layer sizes / activation that ``dde.nn.FNN`` would have used, but adds
    an ``nn.Dropout`` after every hidden-layer activation so the network supports both
    ordinary dropout regularisation and Monte-Carlo dropout at inference.
    """

    def __init__(
        self,
        layer_sizes,
        activation: str = "tanh",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if activation != "tanh":
            # The rest of the code assumes tanh; keep a clear signal if that changes.
            act_map = {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}
            if activation not in act_map:
                raise ValueError(f"Unsupported activation '{activation}'.")
            self._act = act_map[activation]
        else:
            self._act = torch.tanh

        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            lin = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            _glorot_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.linears.append(lin)
            # one dropout per hidden layer (i.e. not after the final linear output)
            is_hidden = i < len(layer_sizes) - 1
            self.dropouts.append(
                nn.Dropout(p=dropout_rate) if (is_hidden and dropout_rate > 0.0)
                else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for j, lin in enumerate(self.linears):
            x = lin(x)
            if j < len(self.linears) - 1:      # hidden layer -> activate + dropout
                x = self._act(x)
                x = self.dropouts[j](x)
        return x


class SEIRPINN(nn.Module):
    """Maps (tau, week, closure[, budget]) -> SEIAR compartment counts (B, P, A, 5)."""

    def __init__(
        self,
        n_patches: int,
        n_age_groups: int,
        n_weeks: int,
        N: torch.Tensor,           # (P, A) population, on the target device/dtype
        cfg: TrainConfig,
        use_budget: bool = False,
    ) -> None:
        super().__init__()
        self.P = n_patches
        self.A = n_age_groups
        self.n_weeks = n_weeks
        self.use_budget = use_budget
        self.n_compartments = N_COMPARTMENTS
        self.dropout_rate = float(getattr(cfg, "dropout_rate", 0.0))

        # Known population, registered as a buffer so it moves with .to(device).
        self.register_buffer("N", N)  # (P, A)

        # Learned embedding for the discrete week index (sub-domain label).
        self.week_embed = nn.Embedding(n_weeks, cfg.week_embed_dim)

        in_dim = 1 + cfg.week_embed_dim + self.P + (self.P if use_budget else 0)
        out_dim = self.P * self.A * N_COMPARTMENTS

        # Custom dropout-capable trunk (replaces dde.nn.FNN, whose PyTorch backend has
        # no dropout support). Same layer sizes / init as before.
        self.trunk = DropoutMLP(
            [in_dim] + [cfg.hidden_width] * cfg.hidden_layers + [out_dim],
            activation=cfg.activation,
            dropout_rate=self.dropout_rate,
        )

    # ------------------------------------------------------------------ #
    # Monte-Carlo dropout controls
    # ------------------------------------------------------------------ #
    def set_mc_dropout(self, enabled: bool) -> None:
        """Force just the dropout layers into train/eval mode for MC sampling.

        Leaves every other module (embeddings, linears) in their current mode, so this
        is safe to call inside a ``torch.no_grad()`` inference block: only the dropout
        masks become stochastic.
        """
        for m in self.trunk.dropouts:
            if isinstance(m, nn.Dropout):
                m.train(enabled)

    @contextlib.contextmanager
    def mc_dropout(self):
        """Context manager enabling MC dropout for the duration of a sampling block."""
        was_training = self.training
        try:
            self.set_mc_dropout(True)
            yield
        finally:
            self.set_mc_dropout(False)
            # restore the overall module training flag
            self.train(was_training)

    def forward(
        self,
        tau: torch.Tensor,          # (B, 1) local time within week, in [0, 1]
        week: torch.Tensor,         # (B,) int64 week index
        closure: torch.Tensor,      # (B, P) float in {0,1}
        budget: torch.Tensor = None # (B, P) float, optional
    ) -> torch.Tensor:
        emb = self.week_embed(week)                     # (B, E)
        features = [tau, emb, closure]
        if self.use_budget:
            if budget is None:
                raise ValueError("Model built with use_budget=True but no budget passed.")
            features.append(budget)
        x = torch.cat(features, dim=1)                  # (B, in_dim)

        logits = self.trunk(x)                          # (B, P*A*5)
        logits = logits.view(-1, self.P, self.A, N_COMPARTMENTS)   # (B, P, A, 5)

        # Softmax over the 5 compartments -> fractions summing to 1, then scale by N.
        frac = torch.softmax(logits, dim=-1)            # (B, P, A, 5)
        counts = frac * self.N.unsqueeze(0).unsqueeze(-1)  # (B, P, A, 5)
        return counts

    @staticmethod
    def split(state: torch.Tensor):
        """Split a (B, P, A, 5) state into S, E, I, A, R, each (B, P, A)."""
        return (
            state[..., 0],
            state[..., 1],
            state[..., 2],
            state[..., 3],
            state[..., 4],
        )