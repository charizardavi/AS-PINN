from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import deepxde as dde

from a_pinn.nets.fnn import make_fnn
from a_pinn.utils.autoscale import make_scaled_fnn


NU = 0.01 / np.pi

def cole_hopf(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    a = np.exp(-np.pi**2 * NU * t)
    num = -2 * np.pi * a * np.sin(np.pi * x)
    den = 1 + 2 * a * np.cos(np.pi * x) + a**2
    return np.where(t == 0, -np.sin(np.pi * x), num / den)

def rel_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    return np.linalg.norm(pred - truth) / np.linalg.norm(truth)

def rel_linf(pred: np.ndarray, truth: np.ndarray) -> float:
    return np.max(np.abs(pred - truth)) / np.max(np.abs(truth))

def residual_rmse(model: dde.Model, pde_module, n: int = 10_000) -> float:
    rng = np.random.default_rng(42)

    dim = pde_module.input_dim
    pts = rng.uniform(-1, 1, size=(n, dim))
    if dim == 2:
        pts[:, 1] = rng.uniform(0, 1, size=n)
    else:
        pts[:, 2] = rng.uniform(0, 1, size=n)

    res = model.predict(pts, operator=pde_module.residual)
    return float(np.sqrt(np.mean(res**2)))


def _load_model(pde_module, ckpt_path: Path) -> dde.Model:
    geomtime = pde_module.geom_time()
    data = dde.data.TimePDE(
        geomtime,
        pde_module.residual,
        pde_module.conditions(geomtime),
        num_domain=1,
        num_boundary=1,
        num_initial=1,
    )
    
    if "autoscale" in ckpt_path.stem:
        net = make_scaled_fnn(
            [pde_module.input_dim] + [128]*8 + [pde_module.output_size],
            activation="tanh",
            initializer="Glorot normal",
        )
    else:
        net = make_fnn([pde_module.input_dim] + [128]*8 + [pde_module.output_size])
    
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.restore(str(ckpt_path))

    return model


def benchmark_burgers(pde_module, ckpt_path: Path, n: int = 10_000) -> None:
    model = _load_model(pde_module, ckpt_path)

    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(n, 1))
    t = rng.uniform(0, 1, size=(n, 1))
    XT = np.hstack([x, t])
    pred  = model.predict(XT)
    truth = cole_hopf(XT[:, 0:1], XT[:, 1:2])

    l2   = rel_l2(pred, truth)
    linf = rel_linf(pred, truth)
    rmse = residual_rmse(model, pde_module, n)

    print(f"\n[Burgers benchmark on {n} random points]")
    print(f"Analytic L2: {l2:.3e}")
    print(f"Analytic Linf: {linf:.3e}")
    print(f"Residual RMSE: {rmse:.3e}\n")


def benchmark_residual(pde_module, ckpt_path: Path, n: int = 10_000) -> None:
    model = _load_model(pde_module, ckpt_path)
    rmse  = residual_rmse(model, pde_module, n)

    print(f"\n[{pde_module.__name__.split('.')[-1]} residual benchmark]")
    print(f"RMSE on {n} random points: {rmse:.3e}\n")
