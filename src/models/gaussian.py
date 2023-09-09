import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal, Categorical
import numpy as np
from .configs import DiagGaussianConfig

from functools import partial
from torch.distributions import Normal, register_kl, MultivariateNormal, kl_divergence
from src.utils.printarr import printarr


class DiagonalNormal(torch.distributions.Distribution):
    def __init__(self, mean, std):
        self._mean = mean
        self._log_var = torch.log(std) * 2
        self.dist = Normal(mean, std)

    @property
    def mean(self):
        return self._mean
    @property
    def var(self):
        return torch.exp(self._log_var)
    
    def mode(self):
        return self.mean
    
    @property
    def std(self):
        return torch.exp(self._log_var/2)

    def sample(self, n_samples=1):
        return self.dist.rsample(torch.zeros(n_samples).size())
    
    def log_prob(self, x, batched=False):
        
        if batched:
            batch_size = x.shape[0] // self.mean.shape[0]
            _mean = self.mean.repeat(batch_size, 1, 1)
            _var = self.var.repeat(batch_size, 1, 1)
            _log_prob = -(x - _mean) ** 2 / _var - 0.5 * (torch.log(_var) + float(np.log(2 * np.pi)))
            _log_prob = _log_prob.sum(-1)
        else:
            p_m = MultivariateNormal(self.mean, covariance_matrix=torch.diag_embed(self.var))
            _log_prob = p_m.log_prob(x)

        return _log_prob
    
    def entropy(self):
        h = self.dist.entropy().sum(-1)
        return h

    def detach(self):
        return DiagonalNormal(self.mean.detach(), self.std.detach())

@register_kl(DiagonalNormal, DiagonalNormal)
def diag_normal_kl(p: DiagonalNormal, q: DiagonalNormal): 
    kl = torch.distributions.kl_divergence(p.dist, q.dist).sum(-1)

    # TEST
    if False:
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

def softplus(x, beta=1, threshold=20):
    return torch.log(1 + torch.exp(beta * x)) * (x * beta <= 20) / beta + x * (x * beta > 20) 


class DiagonalGaussianModule(nn.Module):

    def __init__(self, features, config: DiagGaussianConfig):
        super().__init__()
        self.feats = features
        self.output_dim = config.output_dim
        
        self.mean = nn.Linear(config.input_dim, config.output_dim)
        self.log_var = nn.Linear(config.input_dim, config.output_dim)
        self.min_var = torch.log(torch.tensor(config.min_std)) * 2
        self.max_var = torch.log(torch.tensor(config.max_std)) * 2 

    def forward(self, input):
        feats = self.feats(input)
        # mean, log_var = self.mean(F.leaky_relu(feats)), self.log_var(F.leaky_relu(feats))
        mean, log_var = self.mean(F.elu(feats)), self.log_var(F.elu(feats))
        
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
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad_ = False
        return self
    
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_ = True
        return self


class SphericalGaussianModule(DiagonalGaussianModule):
    def __init__(self, features, config):
        nn.Module.__init__(self)
        self.feats = features
        self.output_dim = config.output_dim

        self.mean = nn.Linear(config.input_dim, config.output_dim)
        self.log_var = nn.Linear(config.input_dim, 1)
        self.min_var = torch.log(torch.tensor(config.min_std)) * 2
        self.max_var = torch.log(torch.tensor(config.max_std)) * 2 

    def forward(self, input):
        feats = self.feats(input)
        mean, log_var = self.mean(F.relu(feats)), self.log_var(F.relu(feats))
        
        # softly constrain the variance
        log_var = self.max_var - softplus(self.max_var - log_var)
        log_var = self.min_var + softplus(log_var - self.min_var)
       
        return mean, log_var * torch.ones_like(mean)
    
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

def SphericalGaussian(config: DiagGaussianConfig):
    return partial(SphericalGaussianModule, config=config)

def Deterministic(config: DiagGaussianConfig):
    config.var = 1e-15
    return partial(FixedVarGaussian, config=config)



### Gaussian Mixture 
class MixtureDiagonalNormal(torch.distributions.Distribution):

    def __init__(self, means, stds, pis):
        super(MixtureDiagonalNormal, self).__init__()
        self._means = means # list
        self._stds = stds # list
        self._pis = pis # (..., K)
        
        # Create DiagonalNormal components
        self._components = [DiagonalNormal(mean, std) for mean, std in zip(self._means, self._stds)]

    @property
    def means(self):
        return self._means

    @property
    def vars(self):
        return [std ** 2 for std in self._stds]
    
    @property
    def stds(self):
        return self._stds
    
    @property
    def mean(self):
        weighted_means = [self._pis[..., i:i+1] * self._means[i] for i in range(len(self._components))]
        return sum(weighted_means)

    def mode(self):
        mixture_mode = torch.argmax(self._pis, dim=-1)
        return self._means[mixture_mode]

    @property
    def var(self):
        weighted_vars_plus_means = [self._pis[..., i:i+1] * (self.vars[i] + self._means[i]**2) for i in range(len(self._components))]
        mixture_var = sum(weighted_vars_plus_means) - self.mean**2
        return mixture_var

    @property
    def std(self):
        return torch.sqrt(self.var)

    def sample(self, n_samples=1):
        # Sampling using Categorical Distribution
        # TODO Test
        categorical = Categorical(probs=self._pis)
        indices = categorical.sample((n_samples,))
        samples = []
        for i in range(n_samples):
            sample_from_chosen_component = self._components[indices[i]].sample()[0] # only one sample
            samples.append(sample_from_chosen_component)
        return torch.stack(samples)

    def log_prob(self, x, batched=False):
        # Compute log prob for each component and then combine with log(pis)
        # import ipdb; ipdb.set_trace()
        batch_size = x.shape[0] // self._means[0].shape[0]
        repeats = [batch_size] + [1] * (len(self._means[0].shape) - 1)
        log_probs = torch.stack([component.log_prob(x, batched=batched) for component in self._components], dim=-1)
        _pis = self._pis.repeat(*repeats) if batched else self._pis
        return torch.logsumexp(log_probs + torch.log(_pis + 1e-10), dim=-1)

    def entropy(self):
        #TODO Test
        # Entropy for each component
        component_entropies = torch.stack([component.entropy() for component in self._components], dim=1)
        
        # Entropy of the mixing coefficients
        pis_entropy = -torch.sum(self._pis * torch.log(self._pis + 1e-10), dim=1)
        # Weighted sum of component entropies
        weighted_component_entropies = torch.sum(self._pis * component_entropies, dim=1)

        return weighted_component_entropies + pis_entropy

    def detach(self):
        return MixtureDiagonalNormal([mean.detach() for mean in self._means], [std.detach() for std in self._stds], self._pis.detach())


class MixtureDiagonalGaussianModule(nn.Module):

    def __init__(self, features, config):
        super().__init__()
        self.feats = features
        self.output_dim = config.output_dim
        self.k = config.n_components

        self.means = nn.ModuleList([nn.Linear(config.input_dim, config.output_dim) for _ in range(self.k)])
        self.log_vars = nn.ModuleList([nn.Linear(config.input_dim, config.output_dim) for _ in range(self.k)])
        self.mixing_coeffs = nn.Linear(config.input_dim, self.k)  # the mixing coefficients
        
        self.min_var = torch.log(torch.tensor(config.min_std)) * 2
        self.max_var = torch.log(torch.tensor(config.max_std)) * 2 

    def forward(self, input):
        feats = self.feats(input)
        means, log_vars = [], []
        for i in range(self.k):
            mean, log_var = self.means[i](F.elu(feats)), self.log_vars[i](F.elu(feats))
            # softly constrain the variance
            log_var = self.max_var - F.softplus(self.max_var - log_var)
            log_var = self.min_var + F.softplus(log_var - self.min_var)
            means.append(mean)
            log_vars.append(log_var)

        pis = F.softmax(self.mixing_coeffs(F.elu(feats)), dim=-1)
        return means, log_vars, pis

    def sample_n_dist(self, input, n_samples=1):
        means, log_vars, pis = self.forward(input)
        stds = [torch.exp(log_var / 2) for log_var in log_vars]
        mixture_dist = MixtureDiagonalNormal(means, stds, pis)
        z = mixture_dist.sample(n_samples)
        std_normal = DiagonalNormal(torch.zeros_like(means[0]), torch.ones_like(stds[0]))
        return z, mixture_dist, std_normal

    def sample(self, input, n_samples=1):
        means, log_vars, pis = self.forward(input)
        stds = [torch.exp(log_var / 2) for log_var in log_vars]
        mixture_dist = MixtureDiagonalNormal(means, stds, pis)
        z = mixture_dist.sample(n_samples)
        return z

    def distribution(self, input):
        means, log_vars, pis = self.forward(input)
        stds = [torch.exp(log_var / 2) for log_var in log_vars]
        mixture_dist = MixtureDiagonalNormal(means, stds, pis)
        return mixture_dist
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad_ = False
        return self
    
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_ = True
        return self

def build_gaussian_mixture(config):
    return partial(MixtureDiagonalGaussianModule, config=config)