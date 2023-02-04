import torch
from torch import nn
from torch.nn import functional as F

from .configs import DiagGaussianConfig

from functools import partial

class DiagonalGaussianModule(nn.Module):

    def __init__(self, features, config: DiagGaussianConfig):
        super().__init__()
        self.feats = features
        self.output_dim = config.output_dim
        self.min_var = torch.tensor(config.min_var)
        self.max_var = torch.tensor(config.max_var) # TODO: should this be in log scale or not?

    def forward(self, input):
        feats = self.feats(input)
        mean, log_var = feats[..., :self.output_dim], feats[..., self.output_dim:]
        
        #softly constrain the variance
        log_var = self.min_var + F.softplus(log_var - self.min_var)
        log_var = self.max_var - F.softplus(self.max_var - log_var)
    
        return mean, log_var

    def sample(self, input, n_samples=1):
        mean, log_var = self.forward(input)
        std = torch.exp(log_var / 2)
        std_normal = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        q = torch.distributions.Normal(mean, std)
        q_no_grad = torch.distributions.Normal(mean.detach(), std.detach())
        z = q.rsample(torch.zeros(n_samples).size())
        return z, q, std_normal, q_no_grad

class FixedVarGaussian(DiagonalGaussianModule):
    def __init__(self, features, config: DiagGaussianConfig):
        super().__init__(features, config)
        self.log_var = torch.log(torch.tensor(config.var))
    
    def forward(self, input):
        feats = self.feats(input)
        mean = feats[..., :self.output_dim]
        log_var = self.log_var * torch.ones_like(mean)
        return mean, log_var

def DiagonalGaussian(config: DiagGaussianConfig):
    return partial(DiagonalGaussianModule, config=config)

def Deterministic(config: DiagGaussianConfig):
    return lambda features: features