import deepxde as dde
import numpy as np
import math

NU = 0.01 / math.pi

output_size = 1


def residual(x, u):
    u_t = dde.grad.jacobian(u, x, i=0, j=1)
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    return u_t + u * u_x - NU * u_xx


def geom_time():
    geom = dde.geometry.Interval(-1, 1)
    time = dde.geometry.TimeDomain(0, 1)
    return dde.geometry.GeometryXTime(geom, time)


def conditions(geomtime):
    ic = dde.IC(
        geomtime,
        lambda X: -np.sin(np.pi * X[:, 0:1]),
        lambda _, on_initial: on_initial,
    )

    def on_left(x, on_boundary):
        return on_boundary and np.isclose(x[0], -1.0)

    bc_l = dde.DirichletBC(
        geomtime,
        lambda X: -np.sin(np.pi * X[:, 0:1]),
        on_left,
    )

    def on_right(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1.0)

    bc_r = dde.DirichletBC(
        geomtime,
        lambda X: -np.sin(np.pi * X[:, 0:1]),
        on_right,
    )

    return [ic, bc_l, bc_r]
