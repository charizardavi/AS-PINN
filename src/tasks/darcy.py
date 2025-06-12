import torch

class DarcyTask:
    def __init__(self, k_range=(0.1,5.0), grid_size=64, device="cuda", bc_weight=1.0):
        self.device    = device
        self.k_min     = k_range[0]
        self.k_max     = k_range[1]
        self.G         = grid_size
        self.dx        = 1.0 / (grid_size - 1)
        self.bc_weight = bc_weight

    def sample_batch(self, batch_size):
        k = torch.rand(batch_size, 1, self.G, self.G, device=self.device)
        k = k * (self.k_max - self.k_min) + self.k_min
        return k

    def loss(self, preds, k):
        u = preds.squeeze(1)
        k = k.squeeze(1)
        dx = self.dx

        u_x = (u[:,:,2:] - u[:,:,:-2]) / (2*dx)
        flux_x = k[:,:,1:-1] * u_x
        div_x = (flux_x[:,:,2:] - flux_x[:,:,:-2]) / (2*dx)
        div_x = div_x[:,2:-2,:]

        u_y = (u[:,2:,:] - u[:,:-2,:]) / (2*dx)
        flux_y = k[:,1:-1,:] * u_y
        div_y = (flux_y[:,2:,:] - flux_y[:,:-2,:]) / (2*dx)
        div_y = div_y[:,:,2:-2]

        pde_loss = (div_x + div_y).pow(2).mean()

        u_full = preds.squeeze(1)
        bc_edges = torch.cat([
            u_full[:,:,0], u_full[:,:,-1],
            u_full[:,0,:], u_full[:,-1,:]
        ], dim=-1)
        bc_loss = bc_edges.pow(2).mean()

        return pde_loss + self.bc_weight * bc_loss
