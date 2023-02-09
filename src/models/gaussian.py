import torch
from torch import nn
from torch.nn import functional as F

from .configs import DiagGaussianConfig

from functools import partial
from torch.distributions import Normal, register_kl


class DiagonalNormal(torch.distributions.Distribution):
    def __init__(self, mean, log_var):
        self._mean = mean
        self._log_var = log_var
        self.dist = Normal(mean, torch.exp(log_var / 2))

    @property
    def mean(self):
        return self._mean
    @property
    def var(self):
        return torch.exp(self._log_var)

    def sample(self, n_samples=1):
        return self.dist.rsample(torch.zeros(n_samples).size())
    
    def log_prob(self, x):
        return self.dist.log_prob(x).sum(-1)
    
    def entropy(self):
        return self.dist.entropy().sum(-1)

    def detach(self):
        return DiagonalNormal(self.mean.detach(), self._log_var.detach())

@register_kl(DiagonalNormal, DiagonalNormal)
def diag_normal_kl(p: DiagonalNormal, q: DiagonalNormal):
    return torch.distributions.kl_divergence(p.dist, q.dist).sum(-1)

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

    def sample_n_dist(self, input, n_samples=1):
        mean, log_var = self.forward(input)
        std = torch.exp(log_var / 2)
        std_normal = DiagonalNormal(torch.zeros_like(mean), torch.ones_like(std))
        q = DiagonalNormal(mean, std)
        z = q.sample(n_samples)
        return z, q, std_normal
    
    def sample(self, input, n_samples=1):
        mean, log_var = self.forward(input)
        std = torch.exp(log_var / 2)
        q = DiagonalNormal(mean, std)
        z = q.sample(n_samples)
        return z
    
    def distribution(self, input):
        mean, log_var = self.forward(input)
        std = torch.exp(log_var / 2)
        q = DiagonalNormal(mean, std)
        return q

class FixedVarGaussian(DiagonalGaussianModule):
    def __init__(self, features, config: DiagGaussianConfig):
        nn.Module.__init__(self)
        self.feats = features
        self._log_var = torch.log(torch.tensor(config.var))
        self.register_buffer('log_var', self._log_var)
    
    def forward(self, input):
        mean = self.feats(input)
        return mean

    def sample_n_dist(self, input, n_samples=1):
        mean = self.forward(input)
        std = torch.exp(self.log_var / 2)
        std_normal = DiagonalNormal(torch.zeros_like(mean), torch.ones_like(std))
        q = DiagonalNormal(mean, std)
        z = q.sample(n_samples)
        return z, q, std_normal

    def sample(self, input, n_samples):
        mean = self.forward(input)
        std = torch.exp(self.log_var / 2)
        q = DiagonalNormal(mean, std)
        z = q.sample(n_samples)
        return z
    
    def distribution(self, input):
        mean = self.forward(input)
        std = torch.exp(self.log_var / 2)
        q = DiagonalNormal(mean, std)
        return q

def DiagonalGaussian(config: DiagGaussianConfig):
    return partial(DiagonalGaussianModule, config=config)

def Deterministic(config: DiagGaussianConfig):
    config.var = 1.
    return partial(FixedVarGaussian, config=config)