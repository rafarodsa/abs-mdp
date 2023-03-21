'''
    Masked Autoregressive Flow for Density Estimation
    based on 'Masked Autoregressive Flow for Density Estimation' by Papamakarios et al.
    author: Rafael Rodriguez-Sanchez
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .made import MADE
from src.utils.printarr import printarr

class MAFBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, reverse=True, sample=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.reverse = reverse
        self.order = torch.arange(1, input_dim + 1) if not reverse else torch.arange(input_dim, 0, -1)
        self.mean = MADE(input_dim, hidden_dims, order=self.order, sample=sample)
        self.scale = MADE(input_dim, hidden_dims, order=self.order, sample=sample)

    def forward(self, x):
        '''
            x: sample from desired distribution
        '''
        m, log_s = self.mean(x), self.scale(x)
        s = torch.exp(log_s)
        u = (x - m) / s
        return u, m, log_s
    
    def inverse(self, u):
        '''
            u: sample from initial distribution
        '''
        x = torch.zeros_like(u)
        printarr(x, u)
        for j in range(self.input_dim):
            i = self.order[j]
            # log_s = torch.clamp(self.scale(x), min=-7)
            log_s = self.scale(x)
            s = torch.exp(log_s)
            x[:, i-1] += u[:, i-1] * s[:, i-1] + self.mean(x)[:, i-1]
        return x, log_s.sum(-1)
    

class MAF(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_blocks, sample=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([MAFBlock(input_dim, hidden_dims, reverse=(i%2==0), sample=sample) for i in range(num_blocks)])

    def forward(self, x):
        '''
            x: sample from desired distribution
        '''
        log_det = 0
        _x = x
        for block in reversed(self.blocks):
            _x, mean, log_s = block(_x)
            log_det += log_s
        return _x, log_det.sum(-1)
    
    def inverse(self, u):
        '''
            u: sample from base distribution
        '''
        _u = u
        log_s = 0
        for block in self.blocks:
            _u, _log_s = block.inverse(_u)
            log_s += _log_s
        return _u, log_s