from functools import partial
import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_act

class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)

# Modifica per accedere a cfg.mem.inplace in modo sicuro
default_inplace_value = False # Valore di default se cfg.mem.inplace non è accessibile

# Per SWISH
swish_inplace = default_inplace_value
if cfg is not None and hasattr(cfg, 'mem') and cfg.mem is not None and hasattr(cfg.mem, 'inplace'):
    swish_inplace = cfg.mem.inplace
register_act('swish', partial(SWISH, inplace=swish_inplace))

# Per LeakyReLU
lrelu_inplace = default_inplace_value
if cfg is not None and hasattr(cfg, 'mem') and cfg.mem is not None and hasattr(cfg.mem, 'inplace'):
    lrelu_inplace = cfg.mem.inplace
register_act('lrelu_03', partial(nn.LeakyReLU, 0.3, inplace=lrelu_inplace))

# Add Gaussian Error Linear Unit (GELU).
register_act('gelu', nn.GELU)