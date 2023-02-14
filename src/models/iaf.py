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

class IAFBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, reverse=True, sample=False):
        super().__init__(self)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.reverse = reverse
        self.order = torch.arange(1, input_dim + 1) if not reverse else torch.arange(input_dim, 0, -1)
        self.mean = MADE(input_dim, hidden_dims, order=self.order, sample=sample)
        self.scale = MADE(input_dim, hidden_dims, order=self.order, sample=sample)

    def forward(self, x):
        mean, scale = self.mean(x), F.sigmoid(self.scale(x))
        z = x * scale + mean * (1-scale)
        return z, mean, scale


class IAF(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_blocks, sample=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([IAFBlock(input_dim, hidden_dims, reverse=(i%2==0), sample=sample) for i in range(num_blocks)])

    def forward(self, x):
        log_det = 0
        _x = x
        for block in self.blocks:
            _x, _, scale = block(_x)
            log_det += torch.log(scale).sum(1)
        return _x, log_det