import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal, Categorical
import numpy as np
from .configs import DiagGaussianConfig

from functools import partial
from torch.distributions import Normal, register_kl, MultivariateNormal, kl_divergence
from src.utils.printarr import printarr

from src.models.common import get_act

class DiagonalNormal(torch.distributions.Distribution):
    def __init__(self, mean, std, feats):
        self._feats = feats
        self._mean = mean
        self._log_var = torch.log(std) * 2
        self.dist = Normal(mean, std)

    @property
    def feats(self):
        return self._feats

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

    def sample(self, n_samples=1, std_limit=3):
        sample = torch.randn_like(self.mean) * self.std * std_limit / 3  + self.mean
        # return self.dist.rsample(torch.zeros(n_samples).size())
        return sample
    
    def log_prob(self, x, batched=False):
        
        if batched:
            batch_size = x.shape[0] // self.mean.shape[0]
            repeats = [batch_size] + [1] * (len(self.mean.shape) - 1)
            _mean = self.mean.repeat(*repeats)
            _var = self.var.repeat(*repeats)
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
        self.activation = config.activation
        self.split = config.input_dim // 2
        self.mean = nn.Sequential(get_act(self.activation), nn.Linear(config.input_dim // 2, config.output_dim))
        self.log_var = nn.Sequential(get_act(self.activation), nn.Linear(config.input_dim - config.input_dim // 2, config.output_dim))
        self.min_var = torch.log(torch.tensor(config.min_std)) * 2
        self.max_var = torch.log(torch.tensor(config.max_std)) * 2 

    def forward(self, input):
        feats = self.feats(input)
        mean, log_var = self.mean(feats[..., :self.split]), self.log_var(feats[..., self.split:])
        
        #softly constrain the variance
        log_var = self.max_var - F.softplus(self.max_var - log_var)
        log_var = self.min_var + F.softplus(log_var - self.min_var)
       
        return (mean, log_var), feats

    def sample_n_dist(self, input, n_samples=1):
 
        (mean, log_var), feats = self.forward(input)
        std = torch.exp(log_var / 2)
        std_normal = DiagonalNormal(torch.zeros_like(mean), torch.ones_like(std))
        q = DiagonalNormal(mean, std, feats)
        z = q.sample(n_samples)
        return z, q, std_normal
    
    def sample(self, input, n_samples=1):
        (mean, log_var), feats = self.forward(input)
        std = torch.exp(log_var / 2)
        q = DiagonalNormal(mean, std, feats)
        z = q.sample(n_samples)
        return z
    
    def distribution(self, input):
        (mean, log_var), feats = self.forward(input)
        std = torch.exp(log_var / 2)
        q = DiagonalNormal(mean, std, feats)
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
        self.activation = config.activation
        self.split = config.input_dim // 2
        self.mean = nn.Sequential(get_act(self.activation), nn.Linear(config.input_dim // 2, config.output_dim))
        self.log_var = nn.Sequential(get_act(self.activation), nn.Linear(config.input_dim - config.input_dim // 2, config.output_dim))
        self.min_var = torch.log(torch.tensor(config.min_std)) * 2
        self.max_var = torch.log(torch.tensor(config.max_std)) * 2 

    def forward(self, input):
        feats = self.feats(input)
        mean, log_var = self.mean(feats[..., :self.split]), self.log_var(feats[..., self.split:])
        
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
        std_normal = DiagonalNormal(torch.zeros_like(mean), torch.ones_like(std), torch.zeros_like(mean))
        q = DiagonalNormal(mean, std, mean)
        z = q.sample(n_samples)
        return z, q, std_normal

    def sample(self, input, n_samples):
        mean = self.forward(input)
        std = torch.exp(self.log_var / 2) * torch.ones_like(mean)
        q = DiagonalNormal(mean, std, mean)
        z = q.sample(n_samples)
        return z
    
    def distribution(self, input):
        mean = self.forward(input)
        std = torch.exp(self.log_var / 2) * torch.ones_like(mean)
        q = DiagonalNormal(mean, std, mean)
        return q

def DiagonalGaussian(config: DiagGaussianConfig):
    return partial(DiagonalGaussianModule, config=config)

def SphericalGaussian(config: DiagGaussianConfig):
    return partial(SphericalGaussianModule, config=config)

def Deterministic(config: DiagGaussianConfig):
    config.var = 1.
    return partial(FixedVarGaussian, config=config)



### Gaussian Mixture 
class MixtureDiagonalNormal(torch.distributions.Distribution):

    def __init__(self, means, stds, pis, feats):
        super(MixtureDiagonalNormal, self).__init__()
        self._feats = feats
        self._means = means # list
        self._stds = stds # list
        self._pis = pis # (..., K)
        # Create DiagonalNormal components
        self._components = [DiagonalNormal(mean, std, feats) for mean, std in zip(self._means, self._stds)]

    @property
    def feats(self):
        return self._feats

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

    def sample_mode(self):
        categorical = Categorical(probs=self._pis)
        mixture_mode = categorical.sample((1,))[0]
        # mixture_mode = torch.argmax(self._pis, dim=-1)
        return self._means[mixture_mode]

    @property
    def var(self):
        weighted_vars_plus_means = [self._pis[..., i:i+1] * (self.vars[i] + self._means[i]**2) for i in range(len(self._components))]
        mixture_var = sum(weighted_vars_plus_means) - self.mean**2
        return mixture_var

    @property
    def std(self):
        return torch.sqrt(self.var)

    # def sample(self, n_samples=1):
    #     # Sampling using Categorical Distribution
    #     # TODO Test
    #     event_shape = self._pis.shape[:-1]
    #     categorical = Categorical(probs=self._pis)
    #     indices = categorical.sample((n_samples,))
    #     samples = []
    #     for i in range(n_samples):
    #         sample_from_chosen_component = self._components[indices[i]].sample()[0] # only one sample
    #         samples.append(sample_from_chosen_component)
    #     return torch.stack(samples)

    def sample(self, n_samples=1, std_limit=3):
        # Sampling using Categorical Distribution
        event_shape = self._pis.shape[:-1]
        categorical = Categorical(probs=self._pis)
        indices = categorical.sample((n_samples,)) # n_samples x event_size
        samples = torch.stack([normal.sample(n_samples, std_limit=std_limit) for normal in self._components]) # n_components x n_samples x event_size x dim_sample
        indices = indices.reshape(-1)
        x = samples.reshape(len(self._components), -1, samples.shape[-1])
        x = x[indices, range(len(indices))]
        return x.reshape(*event_shape, -1)


    def log_prob(self, x, batched=False):
        # Compute log prob for each component and then combine with log(pis)
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
        self.input_dim = config.input_dim
        self.activation = config.activation

        self.per_mod_input = self.input_dim // self.k

        self.means = nn.ModuleList([nn.Sequential(get_act(self.activation), nn.Linear(self.per_mod_input//2, config.output_dim)) for _ in range(self.k)])
        self.log_vars = nn.ModuleList([nn.Sequential(get_act(self.activation), nn.Linear(self.per_mod_input-self.per_mod_input//2, config.output_dim)) for _ in range(self.k)])
        self.mixing_coeffs = nn.Sequential(get_act(self.activation), nn.Linear(config.input_dim, self.k))  # the mixing coefficients
        
        self.min_var = torch.log(torch.tensor(config.min_std)) * 2
        self.max_var = torch.log(torch.tensor(config.max_std)) * 2 


    def forward(self, input):
        feats = self.feats(input)
        means, log_vars = [], []
        for i in range(self.k):
            _feats = feats[..., i*self.per_mod_input:(i+1)*self.per_mod_input]
            mean, log_var = self.means[i](_feats[..., :self.per_mod_input//2]), self.log_vars[i](_feats[..., self.per_mod_input//2:])
            # softly constrain the variance
            log_var = self.max_var - F.softplus(self.max_var - log_var)
            log_var = self.min_var + F.softplus(log_var - self.min_var)
            means.append(mean)
            log_vars.append(log_var)

        pis = F.softmax(self.mixing_coeffs(feats), dim=-1)
        return (means, log_vars, pis), feats

    def sample_n_dist(self, input, n_samples=1):
        (means, log_vars, pis), feats = self.forward(input)
        stds = [torch.exp(log_var / 2) for log_var in log_vars]
        mixture_dist = MixtureDiagonalNormal(means, stds, pis, feats)
        z = mixture_dist.sample(n_samples)
        std_normal = DiagonalNormal(torch.zeros_like(means[0]), torch.ones_like(stds[0]))
        return z, mixture_dist, std_normal

    def sample(self, input, n_samples=1):
        (means, log_vars, pis), feats = self.forward(input)
        stds = [torch.exp(log_var / 2) for log_var in log_vars]
        mixture_dist = MixtureDiagonalNormal(means, stds, pis, feats)
        z = mixture_dist.sample(n_samples)
        return z

    def distribution(self, input):
        (means, log_vars, pis), feats = self.forward(input)
        stds = [torch.exp(log_var / 2) for log_var in log_vars]
        mixture_dist = MixtureDiagonalNormal(means, stds, pis, feats)
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