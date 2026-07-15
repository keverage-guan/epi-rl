"""The policy-conditioned SEIR network.

A single continuous network represents the piecewise (per-week) solution by carrying
sub-domain labels as extra inputs -- the augmented-coordinate / discontinuity-capturing
pattern. Inputs are:

    tau            local time within the week, in [0, 1]           (1)
    week embedding a learned vector for the integer week index k    (E)
    closure state  c_k in {0,1}^P                                   (P)
    budget         remaining closure budget per patch (optional)    (P)

The output is the stacked SEIR state for every (patch, age) pair. Population
conservation S+E+I+R = N is enforced *exactly* by a softmax over the four compartment
logits, scaled by the known N[p,i]; residuals downstream are written in fractions of N.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import deepxde as dde

from .config import TrainConfig


class SEIRPINN(nn.Module):
    """Maps (tau, week, closure[, budget]) -> SEIR compartment counts of shape (B, P, A, 4)."""

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

        # Known population, registered as a buffer so it moves with .to(device).
        self.register_buffer("N", N)  # (P, A)

        # Learned embedding for the discrete week index (sub-domain label).
        self.week_embed = nn.Embedding(n_weeks, cfg.week_embed_dim)

        in_dim = 1 + cfg.week_embed_dim + self.P + (self.P if use_budget else 0)
        out_dim = self.P * self.A * 4

        # DeepXDE FNN is a standard torch Module; we reuse it for the trunk.
        self.trunk = dde.nn.FNN(
            [in_dim] + [cfg.hidden_width] * cfg.hidden_layers + [out_dim],
            cfg.activation,
            cfg.initializer,
        )

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

        logits = self.trunk(x)                          # (B, P*A*4)
        logits = logits.view(-1, self.P, self.A, 4)     # (B, P, A, 4)

        # Softmax over the 4 compartments -> fractions summing to 1, then scale by N.
        frac = torch.softmax(logits, dim=-1)            # (B, P, A, 4)
        counts = frac * self.N.unsqueeze(0).unsqueeze(-1)  # (B, P, A, 4)
        return counts

    @staticmethod
    def split(state: torch.Tensor):
        """Split a (B, P, A, 4) state into S, E, I, R, each (B, P, A)."""
        return state[..., 0], state[..., 1], state[..., 2], state[..., 3]