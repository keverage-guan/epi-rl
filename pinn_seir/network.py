from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn as nn
import deepxde as dde

from .config import TrainConfig

N_COMPARTMENTS = 5

def _glorot_normal_(weight: torch.Tensor) -> None:
    nn.init.xavier_normal_(weight)

class DropoutMLP(nn.Module):

    def __init__(
        self,
        layer_sizes,
        activation: str = "tanh",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if activation != "tanh":
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
            is_hidden = i < len(layer_sizes) - 1
            self.dropouts.append(
                nn.Dropout(p=dropout_rate) if (is_hidden and dropout_rate > 0.0)
                else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for j, lin in enumerate(self.linears):
            x = lin(x)
            if j < len(self.linears) - 1:
                x = self._act(x)
                x = self.dropouts[j](x)
        return x

class SEIRPINN(nn.Module):

    def __init__(
        self,
        n_patches: int,
        n_age_groups: int,
        n_weeks: int,
        N: torch.Tensor,
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

        self.register_buffer("N", N)

        self.week_embed = nn.Embedding(n_weeks, cfg.week_embed_dim)

        in_dim = 1 + cfg.week_embed_dim + self.P + (self.P if use_budget else 0)
        out_dim = self.P * self.A * N_COMPARTMENTS

        self.trunk = DropoutMLP(
            [in_dim] + [cfg.hidden_width] * cfg.hidden_layers + [out_dim],
            activation=cfg.activation,
            dropout_rate=self.dropout_rate,
        )

    def set_mc_dropout(self, enabled: bool) -> None:
        for m in self.trunk.dropouts:
            if isinstance(m, nn.Dropout):
                m.train(enabled)

    @contextlib.contextmanager
    def mc_dropout(self):
        was_training = self.training
        try:
            self.set_mc_dropout(True)
            yield
        finally:
            self.set_mc_dropout(False)
            self.train(was_training)

    def forward(
        self,
        tau: torch.Tensor,
        week: torch.Tensor,
        closure: torch.Tensor,
        budget: torch.Tensor = None
    ) -> torch.Tensor:
        emb = self.week_embed(week)
        features = [tau, emb, closure]
        if self.use_budget:
            if budget is None:
                raise ValueError("Model built with use_budget=True but no budget passed.")
            features.append(budget)
        x = torch.cat(features, dim=1)

        logits = self.trunk(x)
        logits = logits.view(-1, self.P, self.A, N_COMPARTMENTS)

        frac = torch.softmax(logits, dim=-1)
        counts = frac * self.N.unsqueeze(0).unsqueeze(-1)
        return counts

    @staticmethod
    def split(state: torch.Tensor):
        return (
            state[..., 0],
            state[..., 1],
            state[..., 2],
            state[..., 3],
            state[..., 4],
        )
