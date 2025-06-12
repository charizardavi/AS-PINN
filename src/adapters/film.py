from src.adapters.base import AdapterBase
import torch.nn as nn
from neuralop.layers.spectral_convolution import SpectralConv
import types
import torch
from typing import Optional, Sequence

__all__: Sequence[str] = [
    "FiLMBlock",
    "MultiRankFiLMBlock",
    "FiLMAdapter",
]


class FiLMBlock(nn.Module):
    def __init__(self, channels: int, init_gamma: float = 1.0, init_beta: float = 0.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((channels,), init_gamma))
        self.beta = nn.Parameter(torch.full((channels,), init_beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1, -1] + [1] * (x.ndim - 2)
        return self.gamma.view(*shape) * x + self.beta.view(*shape)


class MultiRankFiLMBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        rank: int = 4,
        init_gamma: float = 1.0,
        init_beta: float = 0.0,
    ):
        super().__init__()
        assert rank >= 1, "Rank must be >= 1"
        self.rank = rank
        self.gamma = nn.Parameter(torch.full((rank, channels), init_gamma))
        self.beta = nn.Parameter(torch.full((rank, channels), init_beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1, -1] + [1] * (x.ndim - 2)
        mods = [
            self.gamma[k].view(*shape) * x + self.beta[k].view(*shape)
            for k in range(self.rank)
        ]
        return torch.stack(mods, dim=0).mean(dim=0)


class FiLMAdapter(AdapterBase):
    def __init__(
        self,
        layers: str = "all",
        rank: int = 1,
        init_gamma: float = 1.0,
        init_beta: float = 0.0,
    ):
        super().__init__()
        self.layers: Optional[Sequence[int]] = None if layers == "all" else [
            int(i) for i in layers.split(",") if i.strip() != ""
        ]
        self.rank = rank
        self.init_g = init_gamma
        self.init_b = init_beta

    def attach(self, backbone: nn.Module) -> nn.Module:

        try:
            conv_list = backbone.net.fno_blocks.convs 
        except AttributeError as err:
            raise AttributeError(
                "FiLMAdapter: expected backbone.net.fno_blocks.convs to exist"
            ) from err

        targets = range(len(conv_list)) if self.layers is None else self.layers

        for idx in targets:
            conv = conv_list[idx]
            if not isinstance(conv, SpectralConv):
                raise TypeError(
                    f"FiLMAdapter: layer {idx} is {type(conv).__name__}, expected SpectralConv"
                )
            if hasattr(conv, "film"):
                continue

            film_block: nn.Module
            if self.rank is None or self.rank <= 1:
                film_block = FiLMBlock(conv.out_channels, self.init_g, self.init_b)
            else:
                film_block = MultiRankFiLMBlock(
                    conv.out_channels, self.rank, self.init_g, self.init_b
                )

            conv.add_module("film", film_block)

            orig_transform = conv.transform

            def patched_transform(self_conv: SpectralConv, x, output_shape=None):
                y = orig_transform(x, output_shape=output_shape)
                return self_conv.film(y)

            conv.transform = types.MethodType(patched_transform, conv)

        return backbone
