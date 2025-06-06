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

def _MLP(input_dim: int, hidden_dim: List[int], output_dim: int = None, activation: str = 'relu'):
    layers = []
    n = 1 if activation != 'glu' else 2
    for i in range(len(hidden_dim)):
        if i == 0:
            layers.append(nn.Linear(input_dim, hidden_dim[i] * n))
        else:
            layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i] * n))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU())
        elif activation == 'elu':
            layers.append(nn.ELU())
        elif activation == 'selu':
            layers.append(nn.SELU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'glu':
            layers.append(nn.GLU())
        elif activation == 'silu':
            layers.append(nn.SiLU())
        elif activation == 'none':
            pass
        else:
            raise ValueError('Invalid activation function')
    if output_dim is not None:
        layers.append(nn.Linear(hidden_dim[-1], output_dim) if len(hidden_dim)  > 0 else nn.Linear(input_dim, output_dim)) 
    return nn.Sequential(*layers)

class ResidualMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.mlp = _MLP(cfg.input_dim, cfg.hidden_dims, cfg.output_dim, cfg.activation)
        self.residual = nn.Linear(cfg.input_dim, cfg.output_dim) if cfg.input_dim != cfg.output_dim else nn.Identity()
    def forward(self, x):
        return self.mlp(x) + self.residual(x)

def MLP(cfg: MLPConfig):
    return _MLP(cfg.input_dim, cfg.hidden_dims, cfg.output_dim, cfg.activation)

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