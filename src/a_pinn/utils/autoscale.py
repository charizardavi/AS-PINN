from __future__ import annotations
import math
import torch
import torch.nn as nn
import deepxde as dde


class AutoScale(nn.Module):
    def __init__(
        self,
        input_dim: int,
        alpha_init: float = 1.0,
        beta_init: float = 0.0,
        reg_lambda: float | None = 1e-3,
    ):
        super().__init__()
        assert alpha_init > 0, "alpha_init must be positive."
        self.log_alpha = nn.Parameter(torch.full((input_dim,), math.log(alpha_init)))
        self.beta = nn.Parameter(torch.full((input_dim,), beta_init))
        self.reg_lambda = reg_lambda
        self.losses: list[torch.Tensor] = []

    def forward(self, x):
        alpha = torch.exp(self.log_alpha)
        if self.reg_lambda is not None and self.reg_lambda > 0:
            self.losses = [self.reg_lambda * (self.log_alpha**2).mean()]
        return x * alpha + self.beta


class ScaledFNN(nn.Module):
    def __init__(
        self,
        layers,
        activation="tanh",
        initializer="Glorot normal",
        reg_lambda: float | None = 1e-3,
    ):
        super().__init__()
        self.scaler = AutoScale(layers[0], reg_lambda=reg_lambda)
        self.fnn = dde.nn.pytorch.FNN(layers, activation, initializer)

    def forward(self, x):
        return self.fnn(self.scaler(x))


def make_scaled_fnn(
    layers,
    activation="tanh",
    initializer="Glorot normal",
    reg_lambda: float | None = 1e-3,
):
    return ScaledFNN(layers, activation, initializer, reg_lambda)
