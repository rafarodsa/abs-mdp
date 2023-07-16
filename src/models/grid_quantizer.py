import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as L
from src.utils.printarr import printarr


class GridQuantizerST(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, codebook):
        """
        Args:
            inputs: Tensor, shape [B, M, C]
            codebook: Tensor, shape [M, N, C]
        Returns:
            quantized: Tensor, shape [B, M, C]
            quantized_indices: LongTensor, shape [B, M]
        """
        B, M, C = inputs.shape
        M, N, _ = codebook.shape
        idx_offset = (torch.arange(M) * N).unsqueeze(0).to(inputs.device) # 1 x M


        inputs = inputs.unsqueeze(2) # B x M x 1 x C

        distance = torch.sum((inputs - codebook.unsqueeze(0))**2, dim=-1) # B x M x N
        quantized_indices = torch.argmin(distance, dim=-1) # B x M 
        quantized_indices = quantized_indices + idx_offset # B x M

        
        codes = torch.index_select(codebook.reshape(-1, C), 0, quantized_indices.flatten()).reshape(B, M, C) # B x M x C
        ctx.mark_non_differentiable(quantized_indices)
        ctx.save_for_backward(inputs, codebook, quantized_indices)
        
        return (codes, quantized_indices-idx_offset)

    @staticmethod
    def backward(ctx, grad_codes, grad_indices):
        inputs, codebook, indices = ctx.saved_tensors
        B, M, _ = grad_codes.shape
        M, N, C = codebook.shape
        idx_offset = (torch.arange(M) * N).unsqueeze(0).to(inputs.device) # 1 x M
        grad_inputs = grad_codebook = None
        if ctx.needs_input_grad[0]:
            # Straight-through gradient
            grad_inputs = grad_codes.clone()

        if ctx.needs_input_grad[1]:
            # gradient of the codebook
            embedding_size = C
            # grad_codes = grad_codes.permute(0, 2, 3, 1).contiguous()
            grad_output_flatten = (grad_codes.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook).reshape(-1, C) # M x N x C
            # printarr(indices, codebook)
            grad_codebook.index_add_(0, indices.flatten(), grad_output_flatten)
            grad_codebook = grad_codebook.reshape(M, N, C)
            # printarr(grad_codebook)


        return (grad_inputs, grad_codebook)
    
class GridQuantizer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, codebook):
        """
        Args:
            inputs: Tensor, shape [B, M, C]
            codebook: Tensor, shape [M, N, C]
        Returns:
            quantized: Tensor, shape [B, M, C]
            quantized_indices: LongTensor, shape [B, M]
        """
        B, M, C = inputs.shape
        M, N, _ = codebook.shape
        idx_offset = (torch.arange(M) * N).unsqueeze(0).to(inputs.device) # 1 x M


        inputs = inputs.unsqueeze(2) # B x M x 1 x C

        distance = torch.sum((inputs - codebook.unsqueeze(0))**2, dim=-1) # B x M x N
        quantized_indices = torch.argmin(distance, dim=-1) # B x M 
        quantized_indices = quantized_indices + idx_offset # B x M

        
        codes = torch.index_select(codebook.reshape(-1, C), 0, quantized_indices.flatten()).reshape(B, M, C) # B x M x C
        ctx.mark_non_differentiable(quantized_indices)
        ctx.save_for_backward(inputs, codebook, quantized_indices)
        print('VQ not ST')
        printarr(codes)
        return (codes, quantized_indices-idx_offset)

    @staticmethod
    def backward(ctx, grad_codes, grad_indices):
        raise NotImplementedError('VectorQuantizer does not support backward pass')

class FactoredQuantizer(nn.Module):
    def __init__(self, num_factors, num_codes, embedding_size):
        super().__init__()
        self.num_factors = num_factors
        self.num_codes = num_codes
        self.embedding_size = embedding_size
        # self.codebook = nn.Parameter(torch.rand((num_factors, num_codes, embedding_size)))
        codebook = torch.linspace(-0.5, 0.5, num_codes).unsqueeze(0).unsqueeze(-1).repeat(num_factors, 1, embedding_size)
        self.codebook = nn.Parameter(codebook)
        _idx_offset = (torch.arange(num_factors) * num_codes).unsqueeze(0) # 1 x M
        self.register_buffer('idx_offset', _idx_offset)
    def forward(self, inputs):
        """
        Args:
            inputs: Tensor, shape [B, M, C]
        Returns:
            quantized: Tensor, shape [B, M, C]
            quantized_indices: LongTensor, shape [B, M]
        """
        return GridQuantizerST.apply(inputs, self.codebook)
    
    def codes(self, indices):
        """
        Args:
            indices: LongTensor, shape [B, M]
        Returns:
            codes: Tensor, shape [B, M, C]
        """
        
        _idx = indices + self.idx_offset
        return torch.index_select(self.codebook.reshape(-1, self.embedding_size), 0, _idx.flatten()).reshape(*indices.shape, self.embedding_size)

    def codes_per_factor(self, factor):
        '''
            Returns the codebook for a factor
        '''

        return self.codebook[factor]
    

def FactoredQuantizerSTFactory(cfg):
    return FactoredQuantizer(cfg.factors, cfg.codes, cfg.embedding_dim)

class FactoredCategoricalModule(nn.Module):
    def __init__(self, feats, num_factors, num_codes, hidden_dim):
        super().__init__()
        self.feats = feats
        self._codebook = None
        self.logits = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_codes)) for _ in range(num_factors)])

    @property
    def codebook(self):
        return self._codebook
    
    def set_codebook(self, quantizer):
        self._codebook = quantizer

    def forward(self, x):
        _feats = self.feats(x)
        _logits = [logit(F.relu(_feats)) for logit in self.logits]
        return _logits
    
    def distribution(self, x):
        _logits = self.forward(x)
        categoricals = [torch.distributions.Categorical(logits=logit) for logit in _logits]
        return FactoredCategoricalDistribution(categoricals, self.codebook)

class FactoredCategoricalDistribution(torch.distributions.Distribution):
    def __init__(self, categoricals, codebook):
        self.categoricals = categoricals
        self.codebook = codebook
    
    @property
    def mean(self):
        if self.codebook is None:
            raise ValueError('Codebook is not set')
        else:
            _mean = torch.stack([(self.codebook.codes_per_factor(i).unsqueeze(0) * categorical.probs.unsqueeze(-1)).sum(dim=1) for i, categorical in enumerate(self.categoricals)], dim=1)
        return _mean
    
    @property
    def mode(self):
        _mode = torch.stack([categorical.probs.argmax(dim=-1) for categorical in self.categoricals], dim=-1)
        if self.codebook is None:
            raise ValueError('Codebook is not set')
        code = self.codebook.codes(_mode)
        return code

    def sample(self):
        idx = torch.stack([categorical.sample() for categorical in self.categoricals], dim=-1)
        return self.codebook.codes(idx)
    def log_prob(self, value):
        # value B x M
        # printarr(value)
        _log_prob = torch.stack([categorical.log_prob(value[:, i]).unsqueeze(-1) for i, categorical in enumerate(self.categoricals)], dim=-1)
        return _log_prob.sum(-1)
    
    def entropy(self):
        return sum([categorical.entropy() for categorical in self.categoricals])
    
def FactoredCategoricalModuleFactory(cfg):
    def _factory(feats):
        return FactoredCategoricalModule(feats, cfg.factors, cfg.codes, cfg.hidden_dim)
    return _factory

if __name__=='__main__':

    vq_st = GridQuantizerST.apply
    vq = GridQuantizer.apply

    def test_vq_st_gradient1():
        B = 2
        M = 5 # number of latent factors
        N = 10 # number of codes per latent factor
        C = 11 # embedding size
        inputs = torch.rand((B, M, C), dtype=torch.float32, requires_grad=True)
        codebook = torch.rand((M, N, C), dtype=torch.float32, requires_grad=True)
        codes, _ = vq_st(inputs, codebook)
        printarr(codes, codebook, inputs)
        grad_output = torch.rand((B, M, C))
        grad_inputs, = torch.autograd.grad(codes, inputs,
            grad_outputs=[grad_output])

        # Straight-through estimator
        assert grad_inputs.size() == (B, M, C)
        assert np.allclose(grad_output.numpy(), grad_inputs.numpy())

    def test_vq_st_gradient2():
        B = 2
        M = 10 # number of latent factors
        N = 10 # number of codes per latent factor
        C = 5 # embedding size
        inputs = torch.rand((B, M, C), dtype=torch.float32, requires_grad=True)
        codebook = torch.rand((M, N, C), dtype=torch.float32, requires_grad=True)
        codes, indices = vq_st(inputs, codebook)
        # codes = codes.permute(0,2,3,1).contiguous()
        _, indices = vq(inputs, codebook)

        idx_offset = (torch.arange(M) * N).unsqueeze(0)
        
        
        codes_torch = torch.embedding(codebook.reshape(-1, C), (indices+idx_offset).reshape(-1, 1), padding_idx=-1,
            scale_grad_by_freq=False, sparse=False).reshape((B, M, C))
        
        printarr(codes, indices, codes_torch)
        grad_output = torch.rand((B, M, C), dtype=torch.float32)
        grad_codebook, = torch.autograd.grad(codes, codebook,
            grad_outputs=[grad_output])
        grad_codebook_torch, = torch.autograd.grad(codes_torch, codebook,
            grad_outputs=[grad_output])
        printarr(grad_codebook, grad_codebook_torch)

        # Gradient is the same as torch.embedding function
        assert grad_codebook.size() == (M, N, C)
        assert np.allclose(grad_codebook.numpy(), grad_codebook_torch.numpy())

    def test_gridq_selector():
        B = 2
        M = 3 # number of latent factors
        N = 10 # number of codes per latent factor
        C = 5 # embedding size
        inputs = torch.rand((B, M, C), dtype=torch.float32, requires_grad=True)
        codebook = torch.rand((M, N, C), dtype=torch.float32, requires_grad=True)
        codes, indices = vq_st(inputs, codebook)

        # manual selection
        distance = torch.sum((inputs.unsqueeze(2) - codebook.unsqueeze(0))**2, dim=-1) # B x M x N
        quantized_indices = torch.argmin(distance, dim=-1) # B x M

        codes_manual = torch.zeros((B, M, C))

        for b in range(B):
            for m in range(M):
                codes_manual[b, m] = codebook[m, quantized_indices[b, m]]
        
        assert np.allclose(codes_manual.detach().numpy(), codes.detach().numpy())

    test_vq_st_gradient1()
    test_vq_st_gradient2()
    test_gridq_selector()