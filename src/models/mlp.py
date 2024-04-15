"""
    MLP constructor
    author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
    date: February 2023
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
from .configs import MLPConfig
from src.utils.printarr import printarr
from src.models.simnorm import SimNorm

def get_act(activation):
    actlayer = None
    if activation == 'relu':
        actlayer = nn.ReLU()
    elif activation == 'sigmoid':
        actlayer = nn.Sigmoid()
    elif activation == 'tanh':
        actlayer = nn.Tanh()
    elif activation == 'leaky_relu':
        actlayer = nn.LeakyReLU()
    elif activation == 'elu':
        actlayer = nn.ELU()
    elif activation == 'selu':
        actlayer = nn.SELU()
    elif activation == 'gelu':
        actlayer = nn.GELU()
    elif activation == 'glu':
        actlayer = nn.GLU()
    elif activation == 'silu':
        actlayer = nn.SiLU()
    elif activation == 'mish':
        actlayer = nn.Mish()
    elif activation == 'simnorm':
        actlayer = SimNorm()
    elif activation == 'none':
        actlayer == nn.Identity()
    else:
        raise ValueError(f'Invalid activation function {activation}')
    
    return actlayer

def _MLP(input_dim: int, hidden_dim: List[int], output_dim: int = None, activation: str = 'relu', normalize=False, outact='none', init_fn=lambda l: l, out_init_fn=None):
    layers = []
    n = 1 if activation != 'glu' else 2
    out_init_fn = init_fn if out_init_fn is None else out_init_fn
    for i in range(len(hidden_dim)):
        if i == 0:
            layers.append(init_fn(nn.Linear(input_dim, hidden_dim[i] * n)))
        else:
            layers.append(init_fn(nn.Linear(hidden_dim[i-1], hidden_dim[i] * n)))
        
        if normalize:
            layers.append(nn.LayerNorm(hidden_dim[i] * n))

        layers.append(get_act(activation=activation))

    if output_dim is not None:
        layers.append(out_init_fn(nn.Linear(hidden_dim[-1], output_dim) if len(hidden_dim)  > 0 else nn.Linear(input_dim, output_dim)))
    if outact != 'none':
        layers.append(get_act(outact))

    return nn.Sequential(*layers)

class ResidualMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.mlp = _MLP(cfg.input_dim, cfg.hidden_dims, cfg.output_dim, cfg.activation)
        self.residual = nn.Linear(cfg.input_dim, cfg.output_dim) if cfg.input_dim != cfg.output_dim else nn.Identity()
    def forward(self, x):
        return self.mlp(x) + self.residual(x)

def MLP(cfg: MLPConfig, init_fn=lambda l: l, out_init_fn=None):
    normalize = cfg.normalize if 'normalize' in cfg else False
    outact = cfg.outact if 'outact' in cfg else 'none'
    return _MLP(cfg.input_dim, cfg.hidden_dims, cfg.output_dim, cfg.activation, normalize=normalize, outact=outact, init_fn=init_fn, out_init_fn=None)

def DynamicsMLP(cfg):
    return _MLP(cfg.latent_dim + cfg.n_options, cfg.hidden_dims, output_dim=cfg.latent_dim, activation=cfg.activation)
    # return ResidualMLP(cfg)
def RewardMLP(cfg):
    return _MLP(2*cfg.latent_dim + cfg.n_options, cfg.hidden_dims, 1, cfg.activation)


class _MLPCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = _MLP(cfg.input_dim + cfg.latent_dim, cfg.hidden_dims, cfg.output_dim, cfg.activation)
    def forward(self, x, z):
        return self.mlp(torch.cat([x, z], dim=-1))


def MLPCritic(cfg):
    return _MLPCritic(cfg)