import deepxde as dde
import numpy as np

EPSILON = 0.01

output_size = 1


def residual(x, u):
    u_t = dde.grad.jacobian(u, x, i=0, j=2)
    u_xx = dde.grad.hessian(u, x, component=0, i=0)
    u_yy = dde.grad.hessian(u, x, component=0, i=1)
    lap = u_xx + u_yy

    return u_t - EPSILON**2 * lap + u**3 - u


def geom_time():
    geom = dde.geometry.Rectangle([-1, -1], [1, 1])
    time = dde.geometry.TimeDomain(0, 1)
    return dde.geometry.GeometryXTime(geom, time)


def conditions(geomtime):
    ic = dde.IC(
        geomtime,
        lambda x: np.cos(np.pi * x[:, 0:1]) * np.cos(np.pi * x[:, 1:2]),
        lambda _, on_initial: on_initial,
    )

    bc = dde.DirichletBC(
        geomtime,
        lambda X: np.zeros((X.shape[0], output_size)),
        boundary,
    )

    return [ic, bc]


def boundary(x, on_b):
    return on_b
