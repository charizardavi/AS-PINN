import deepxde as dde
import numpy as np

G = 9.81

output_size = 3

def residual(x, u):

    h = u[:, 0:1]
    hu = u[:, 1:2]
    hv = u[:, 2:3]

    h_t = dde.grad.jacobian(h, x, i=0, j=2)
    hu_t = dde.grad.jacobian(hu, x, i=0, j=2)
    hv_t = dde.grad.jacobian(hv, x, i=0, j=2)

    hu_x = dde.grad.jacobian(hu, x, i=0, j=0)
    hv_y = dde.grad.jacobian(hv, x, i=0, j=1)
    eq1 = h_t + hu_x + hv_y

    flux_hu_x = dde.grad.jacobian(hu**2 / h + 0.5 * G * h**2, x, i=0, j=0)
    flux_hu_y = dde.grad.jacobian(hu * hv / h, x, i=0, j=1)
    eq2 = hu_t + flux_hu_x + flux_hu_y

    flux_hv_x = dde.grad.jacobian(hu * hv / h, x, i=0, j=0)
    flux_hv_y = dde.grad.jacobian(hv**2 / h + 0.5 * G * h**2, x, i=0, j=1)
    eq3 = hv_t + flux_hv_x + flux_hv_y

    return dde.concat([eq1, eq2, eq3], axis=1)


def geom_time():

    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    time = dde.geometry.TimeDomain(0, 0.2)
    return dde.geometry.GeometryXTime(geom, time)


def boundary(x, on_boundary):
    return on_boundary


def conditions(geomtime):
    ic_h = dde.IC(
        geomtime,
        lambda X: 1 + 0.1 * np.exp(-50 * ((X[:, 0:1] - 0.5) ** 2 + (X[:, 1:2] - 0.5) ** 2)),
        lambda _, on_i: on_i,
        component=0,
    )

    ic_hu = dde.IC(
        geomtime,
        lambda X: np.zeros((X.shape[0], 1)),
        lambda _, on_i: on_i,
        component=1,
    )

    ic_hv = dde.IC(
        geomtime,
        lambda X: np.zeros((X.shape[0], 1)),
        lambda _, on_i: on_i,
        component=2,
    )

    bc_h = dde.NeumannBC(
        geomtime,
        lambda X: np.zeros((X.shape[0], 1)),
        boundary,
        component=0,
    )
    bc_hu = dde.NeumannBC(
        geomtime,
        lambda X: np.zeros((X.shape[0], 1)),
        boundary,
        component=1,
    )
    bc_hv = dde.NeumannBC(
        geomtime,
        lambda X: np.zeros((X.shape[0], 1)),
        boundary,
        component=2,
    )
    return [ic_h, ic_hu, ic_hv, bc_h, bc_hu, bc_hv]
