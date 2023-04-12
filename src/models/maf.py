'''
    Masked Autoregressive Flow for Density Estimation
    based on 'Masked Autoregressive Flow for Density Estimation' by Papamakarios et al.
    author: Rafael Rodriguez-Sanchez
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .made import MADE, ConditionalMADE
from src.utils.printarr import printarr
from functools import partial

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
    

class ConditionalMAFBlock(MAFBlock):
    def __init__(self, input_dim, hidden_dims, cond_dim, reverse=True, sample=False):
        super().__init__(input_dim, hidden_dims, reverse=reverse, sample=sample)
        self.mean = ConditionalMADE(input_dim, hidden_dims, cond_dim, order=self.order, sample=sample)
        self.scale = ConditionalMADE(input_dim, hidden_dims, cond_dim, order=self.order, sample=sample)
    
    def forward(self, x, cond):
        '''
            x: sample from desired distribution
            cond: conditioning variable
        '''
        m, log_s = self.mean(x, cond), self.scale(x, cond)
        s = torch.exp(log_s)
        u = (x - m) / s
        return u, m, log_s
    
    def inverse(self, u, cond):
        '''
            u: sample from initial distribution
            cond: conditioning variable
        '''
        x = torch.zeros_like(u)
        for j in range(self.input_dim):
            i = self.order[j]
            # log_s = torch.clamp(self.scale(x, cond), min=-7)
            log_s = self.scale(x, cond)
            s = torch.exp(log_s)
            x[:, i-1] += u[:, i-1] * s[:, i-1] + self.mean(x, cond)[:, i-1]
        return x, log_s.sum(-1)

class ConditionalMAF(MAF):
    def __init__(self, input_dim, hidden_dims, num_blocks, cond_dim, sample=False):
        super().__init__(input_dim, hidden_dims, num_blocks, sample=sample)
        self.blocks = nn.ModuleList([ConditionalMAFBlock(input_dim, hidden_dims, cond_dim, reverse=(i%2==0), sample=sample) for i in range(num_blocks)])
    
    def forward(self, x, cond):
        '''
            x: sample from desired distribution
            cond: conditioning variable
        '''
        log_det = 0
        _x = x
        for block in reversed(self.blocks):
            _x, mean, log_s = block(_x, cond)
            log_det += log_s
        return _x, log_det.sum(-1)
    
    def inverse(self, u, cond):
        '''
            u: sample from base distribution
            cond: conditioning variable
        '''
        _u = u
        log_s = 0
        for block in self.blocks:
            _u, _log_s = block.inverse(_u, cond)
            log_s += _log_s
        return _u, log_s

class MAFDist:
    def __init__(self, maf, cond):
        self.cond = cond
        self.maf = maf
    def log_prob(self, x):
        return self.maf.log_prob(x, self.cond)
    def freeze(self):
        for p in self.maf.parameters():
            p.requires_grad_ = False
        return self
    def sample(self, n=1):
        printarr(self.cond)
        h = self.maf.features(self.cond)
        cond = h.repeat_interleave(n, dim=0)
        u = torch.randn(cond.shape[0], self.maf.input_dim)
        printarr(u, cond)
        samples = self.maf.inverse(u, cond)[0].reshape(self.cond.shape[0], n, self.maf.input_dim).permute(0, 2, 1)
        printarr(samples)
        return samples

class MAFDistribution(ConditionalMAF):

    def __init__(self, features, cfg):
        super().__init__(cfg.output_dim, cfg.hidden_dims, cfg.num_blocks, cfg.input_dim)
        self.features = features
        self.cfg = cfg
        
    def forward(self, x, cond):
        c = self.features(cond)
        return super().forward(x, c)

    def log_prob(self, x, cond):
        u, log_det = self.forward(x, cond)
        log_prob = -0.5 * (u**2).sum(-1) - 0.5 * np.log(2 * np.pi) * self.cfg.output_dim - log_det
        return log_prob
    
    def distribution(self, cond):
        return MAFDist(self, cond)
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad_ = False
        return self
    
def CondMAFFactory(cfg):
    return partial(MAFDistribution, cfg=cfg)