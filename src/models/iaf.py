'''
    Inverse Autoregressive Flow (IAF) implementation
    from 'Improved Variational Inference with Inverse Autoregressive Flow' by Kingma et al.
    author: Rafael Rodriguez-Sanchez
    date: February 2023
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .made import MADE

from functools import partial

class IAFBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, cond_dim, reverse=True, sample=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.reverse = reverse
        self.order = torch.arange(1, input_dim + 1) if not reverse else torch.arange(input_dim, 0, -1)
        self.mean = MADE(input_dim, hidden_dims, order=self.order, sample=sample)
        self.scale = MADE(input_dim, hidden_dims, order=self.order, sample=sample)
        self.cond = nn.Linear(cond_dim, self.input_dim)
    def forward(self, x, h=None):
        _x = x + self.cond(h) if h is not None else x
        mean, scale = self.mean(_x), torch.sigmoid(self.scale(_x))
        z = x * scale + mean * (1-scale)
        return z, mean, scale


class IAF(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dims, num_blocks, sample=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([IAFBlock(out_dim, hidden_dims, cond_dim=out_dim, reverse=(i%2==0), sample=sample) for i in range(num_blocks)])

    def forward(self, x, h=None):
        log_det = 0
        _x = x
        for block in self.blocks:
            _x, _, scale = block(_x, h)
            log_det += torch.log(scale)
        return _x, log_det


class IAFDist(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        self.feats = features
        self.cfg = config
        self.iaf = IAF(config.input_dim, config.output_dim, config.hidden_dims, config.num_blocks)
    
    def forward(self, z, h):
        _h = self.feats(h)
        return self.iaf(z, _h)

    def sample(self, h, n_samples=1):
        z = torch.randn(n_samples, self.cfg.output_dim)
        z, _ = self.forward(z, h)
        return z

    def sample_n_dist(self, h, n_samples=1):
        epsilon = torch.randn(n_samples * h.shape[0], self.cfg.output_dim).to(h.device)
        z, log_det = self.forward(epsilon, h)
        return z, log_det.sum(-1), None

def IAFDistribution(config):
    return partial(IAFDist, config=config)