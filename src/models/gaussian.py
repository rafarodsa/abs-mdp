import torch
from torch import nn
from torch.nn import functional as F

from .configs import DiagGaussianConfig

from functools import partial
from torch.distributions import Normal, register_kl, MultivariateNormal, kl_divergence

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
    
    @property
    def std(self):
        return torch.exp(self._log_var/2)

    def sample(self, n_samples=1):
        return self.dist.rsample(torch.zeros(n_samples).size())
    
    def log_prob(self, x):
        logp = self.dist.log_prob(x).sum(-1)
        p_m = MultivariateNormal(self.mean, covariance_matrix=torch.diag_embed(self.var))
        assert torch.allclose(p_m.log_prob(x), logp)
        return logp
    
    def entropy(self):
        return self.dist.entropy().sum(-1)

    def detach(self):
        return DiagonalNormal(self.mean.detach(), self._log_var.detach())

@register_kl(DiagonalNormal, DiagonalNormal)
def diag_normal_kl(p: DiagonalNormal, q: DiagonalNormal): 
    kl = torch.distributions.kl_divergence(p.dist, q.dist).sum(-1)

    # TEST
    if True:
        cov= torch.diag_embed(p.var)
        assert torch.allclose(torch.diagonal(cov, dim1=1, dim2=2), p.var)
        p_m = MultivariateNormal(p.mean, covariance_matrix=torch.diag_embed(p.var))
        q_m = MultivariateNormal(q.mean, covariance_matrix=torch.diag_embed(q.var))
        kl_test = kl_divergence(p_m, q_m)
        assert torch.all(~torch.isnan(kl)), f'KL has NaNs'
        assert torch.all(~torch.isnan(kl_test)), f'KL_test has NaNs'
        assert kl.shape == kl_test.shape
        # assert torch.allclose(kl_test, kl, rtol=1e-4), f'Failed with avg abs diference: {(kl_test-kl).abs().mean()}. min: {(kl_test-kl).abs().min()}. max: {(kl_test-kl).abs().max()}'

    return kl

class DiagonalGaussianModule(nn.Module):

    def __init__(self, features, config: DiagGaussianConfig):
        super().__init__()
        self.feats = features
        self.output_dim = config.output_dim

        self.mean = nn.Linear(config.input_dim, config.output_dim)
        self.log_var = nn.Linear(config.input_dim, config.output_dim)
        self.min_var = torch.log(torch.tensor(config.min_std)) * 2
        self.max_var = torch.log(torch.tensor(config.max_std)) * 2 # TODO: should this be in log scale or not?

    def forward(self, input):
        feats = self.feats(input)
        mean, log_var = self.mean(F.relu(feats)), self.log_var(F.relu(feats))
        
        #softly constrain the variance
        log_var = self.max_var - F.softplus(self.max_var - log_var)
        log_var = self.min_var + F.softplus(log_var - self.min_var)
       
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
        # self.mean = nn.Linear(config.input_dim, config.output_dim)
    
    def forward(self, input):
        mean = self.feats(input)
        return mean

    def sample_n_dist(self, input, n_samples=1):
        mean = self.forward(input)
        std = torch.exp(self.log_var / 2) * torch.ones_like(mean)
        std_normal = DiagonalNormal(torch.zeros_like(mean), torch.ones_like(std))
        q = DiagonalNormal(mean, std)
        z = q.sample(n_samples)
        return z, q, std_normal

    def sample(self, input, n_samples):
        mean = self.forward(input)
        std = torch.exp(self.log_var / 2) * torch.ones_like(mean)
        q = DiagonalNormal(mean, std)
        z = q.sample(n_samples)
        return z
    
    def distribution(self, input):
        mean = self.forward(input)
        std = torch.exp(self.log_var / 2) * torch.ones_like(mean)
        q = DiagonalNormal(mean, std)
        return q

def DiagonalGaussian(config: DiagGaussianConfig):
    return partial(DiagonalGaussianModule, config=config)

def Deterministic(config: DiagGaussianConfig):
    config.var = 1e-10
    return partial(FixedVarGaussian, config=config)

