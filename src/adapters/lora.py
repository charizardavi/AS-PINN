from src.adapters.base import AdapterBase
import torch.nn as nn, types
import loralib as lora
from neuralop.layers.spectral_convolution import SpectralConv

class LoRAAdapter(AdapterBase):
    def __init__(self, rank=4, layers="0", alpha=1.0):
        super().__init__()
        self.rank   = rank
        self.alpha  = alpha
        self.layers = [int(i) for i in layers.split(",")]

    def attach(self, backbone: nn.Module):
        try:
            convs = backbone.net.fno_blocks.convs
        except AttributeError:
            raise RuntimeError("LoRAAdapter: cannot find backbone.net.fno_blocks.convs")

        for idx in self.layers:
            conv: SpectralConv = convs[idx]
            if not isinstance(conv, SpectralConv):
                raise TypeError(f"Expected SpectralConv, got {type(conv)}")

            C = conv.out_channels

            lora_lin = lora.Linear(
                in_features = C,
                out_features= C,
                r           = self.rank,
                lora_alpha  = self.alpha,
                bias        = False
            )
            conv.add_module(f"lora_linear_{idx}", lora_lin)

            orig_transform = conv.transform
            def make_transform(orig_fn, lin_mod, C):
                def new_transform(self, x_ft, output_shape=None):
                    y_base = orig_fn(x_ft, output_shape)
                    B, C_ = y_base.shape[:2]
                    spatial = y_base.shape[2:]
                    y_flat  = y_base.permute(0, *range(2, y_base.ndim), 1)
                    y_flat  = y_flat.reshape(-1, C_)
                    delta   = lin_mod(y_flat).view(B, *spatial, C_)
                    delta   = delta.permute(0, -1, *range(1, len(spatial)+1))
                    return y_base + delta
                return new_transform

            conv.transform = types.MethodType(
                make_transform(orig_transform, lora_lin, C),
                conv
            )

        for p in backbone.parameters():
            p.requires_grad = False
        
        for m in backbone.modules():
            if isinstance(m, lora.Linear):
                for p in m.parameters():
                    p.requires_grad = True

        return backbone
