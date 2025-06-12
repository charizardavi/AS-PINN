from neuralop.models.fno import FNO2d
import torch.nn as nn
import torch
from torch.serialization import safe_globals, add_safe_globals
import neuralop.layers.spectral_convolution as sc


class FNOBackbone(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        width: int = 64,
        depth: int = 5,
        n_modes_height: int = 16,
        n_modes_width: int = 16,
        embed_dim: int = 2,  # ← new parameter
        ckpt_path: str | None = None,
    ):
        super().__init__()
        self.net = FNO2d(
            in_channels=in_ch,
            out_channels=out_ch,
            n_modes_height=n_modes_height,
            n_modes_width=n_modes_width,
            hidden_channels=width,
            n_layers=depth,
            embed_dim=embed_dim,  # ← pass it here
        )
        if ckpt_path:
            # disable PyTorch 2.6 safe mode if needed...
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.load_state_dict(state["model"], strict=False)

    def forward(self, x):
        return self.net(x)

    def load_backbone(self, path: str):
        add_safe_globals([torch._C._nn.gelu, sc.SpectralConv])

        with safe_globals([torch._C._nn.gelu, sc.SpectralConv]):
            state = torch.load(path, map_location="cpu")

        self.load_state_dict(state["model"], strict=False)
