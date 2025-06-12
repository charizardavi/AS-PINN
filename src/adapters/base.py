import torch
import torch.nn as nn


class AdapterBase(nn.Module):
    def attach(self, backbone: nn.Module):
        raise NotImplementedError
