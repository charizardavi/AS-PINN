import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear
from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork as GatePointsNetwork
import deepxde as dde
from deepxde.geometry.geometry_2d import Rectangle, Disk
import numpy as np

# —— 1) Force PyTorch backend & 64-bit precision ——
dde.config.set_default_backend("pytorch")
dde.config.pytorch_default_dtype = torch.float64

NUM_DOMAIN = 100
NUM_BOUNDARY = 10
NUM_COLL_POINTS = NUM_DOMAIN + NUM_BOUNDARY


class GE_PINN(nn.Module):
    regularizer = None

    def __init__(self, dim=2, r_max=0.3, num_neighbors=20):
        super().__init__()
        self.dim = dim
        self.r_max = r_max
        self.num_neighbors = num_neighbors

        # for input normalization: map [0,2]→[-1,1], [0,1]→[-1,1]
        self.register_buffer("x_offset", torch.tensor([1.0, 0.5]))  
        self.register_buffer("x_scale",  torch.tensor([1.0, 0.5]))  

        irreps_in = Irreps("0e")
        irreps_out = Irreps("0e + 1o")

        self.network = GatePointsNetwork(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            max_radius=r_max,
            num_neighbors=num_neighbors,
            num_nodes=NUM_COLL_POINTS,
            mul=16,
            layers=3,
            lmax=1,
            pool_nodes=False,
        )
        hidden_irreps = irreps_out
        self.to_vec = Linear(irreps_in=hidden_irreps, irreps_out=Irreps("1o"))
        self.to_p   = Linear(irreps_in=hidden_irreps, irreps_out=Irreps("0e"))

    def forward(self, x):
        # —— 2) normalize to [-1,1]^2 ——
        # x is (N,2) in [0,2]x[0,1]
        x = (x - self.x_offset) / self.x_scale

        device = x.device
        if self.dim == 2:
            zeros = torch.zeros(x.shape[0], 1, device=device, dtype=x.dtype)
            coords = torch.cat([x, zeros], dim=1)
        else:
            coords = x

        node_inputs = torch.ones(
            coords.shape[0], self.network.irreps_in.dim, device=device, dtype=x.dtype
        )
        data_dict = {"pos": coords, "x": node_inputs}
        latent = self.network(data_dict)
        vec    = self.to_vec(latent)
        p      = self.to_p(latent)
        uv     = vec[:, :2] if self.dim == 2 else vec
        return torch.cat([uv, p], dim=1).to(x.dtype)  # ensure consistency


def navier_stokes(x, y):
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    grads = {}
    for comp in [0, 1, 2]:
        for axis in [0, 1]:
            grads[f"d{comp}_{axis}"] = dde.grad.jacobian(y, x, i=comp, j=axis)
    grads.update({
        f"d{comp}_{axis}_{axis}": dde.grad.hessian(y, x, i=comp, j=axis)
        for comp in [0, 1] for axis in [0, 1]
    })

    # Debug: spot any remaining NaNs
    for name, g in grads.items():
        if torch.isnan(g).any():
            print(f"⚠️  grads['{name}'] has NaNs")

    Re = 100.0
    lap_u = grads["d0_0_0"] + grads["d0_1_1"]
    lap_v = grads["d1_0_0"] + grads["d1_1_1"]

    eq1 = u * grads["d0_0"] + v * grads["d0_1"] + grads["d2_0"] - (1 / Re) * lap_u
    eq2 = u * grads["d1_0"] + v * grads["d1_1"] + grads["d2_1"] - (1 / Re) * lap_v
    eq3 = grads["d0_0"] + grads["d1_1"]

    return eq1, eq2, eq3



rect = Rectangle([0, 0], [2, 1])
disk = Disk([0.2, 0.5], 0.1)
domain = rect - disk


def inflow(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)


def outflow(x, on_boundary):
    return on_boundary and np.isclose(x[0], 2.0)


def walls(x, on_boundary):
    return on_boundary and (np.isclose(x[1], 0.0) or np.isclose(x[1], 1.0))


def cylinder(x, on_boundary):
    r = np.linalg.norm(x - np.array([0.2, 0.5]))
    return on_boundary and np.isclose(r, 0.1, atol=1e-3)


bcs = [
    dde.DirichletBC(domain, lambda x: 1.0, inflow, component=0),
    dde.DirichletBC(domain, lambda x: 0.0, inflow, component=1),
    dde.DirichletBC(domain, lambda x: 0.0, walls, component=0),
    dde.DirichletBC(domain, lambda x: 0.0, walls, component=1),
    dde.DirichletBC(domain, lambda x: 0.0, cylinder, component=0),
    dde.DirichletBC(domain, lambda x: 0.0, cylinder, component=1),
    dde.DirichletBC(domain, lambda x: 0.0, outflow, component=2),
]

data = dde.data.PDE(
    domain,
    navier_stokes,
    bcs,
    num_domain=NUM_DOMAIN,
    num_boundary=NUM_BOUNDARY,
)

net = GE_PINN(dim=2, r_max=0.3, num_neighbors=20)
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=5000)

model.compile("L-BFGS-B")
losshistory, train_state = model.train()
