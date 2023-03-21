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
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.reverse = reverse
        self.order = torch.arange(1, input_dim + 1) if not reverse else torch.arange(input_dim, 0, -1)
        self.mean = MADE(input_dim, hidden_dims, order=self.order, sample=sample)
        self.scale = MADE(input_dim, hidden_dims, order=self.order, sample=sample)

    def forward(self, x):
        m, s = self.mean(x), torch.clamp(torch.exp(self.scale(x)), min=1e-5)
        z = x * s + m
        return z, m, s
    
    def inverse(self, z):
        with torch.no_grad():
            x = torch.zeros_like(z)
            for j in range(self.input_dim):
                i = self.order[j]
                s = torch.clamp(torch.exp(self.scale(x)), min=1e-5)
                # x[:, i-1] = (z[:, i-1] - self.mean(x)[:, i-1] * (1-s[:, i-1])) / s[:, i-1]
                x[:, i-1] = (z[:, i-1] - self.mean(x)[:, i-1]) / s[:, i-1]
        return x


class IAF(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_blocks, sample=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([IAFBlock(input_dim, hidden_dims, reverse=False, sample=sample) for i in range(num_blocks)])

    def forward(self, x):
        '''
            x: sample from initial distribution
        '''
        log_det = 0
        _x = x
        for block in self.blocks:
            _x, mean, scale = block(_x)
            log_det += torch.log(scale)
        return _x, log_det.sum(-1)
    

    def inverse(self, x):
        '''
            x: sample from final distribution
        '''
        _x = x
        for block in reversed(self.blocks):
            _x = block.inverse(_x)
        return _x    
