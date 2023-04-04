import torch
import torch.nn.functional as F

from src.utils.printarr import printarr
class QuantizedLogisticMixture(torch.distributions.Distribution):
    def __init__(self, loc, log_scale, logit_probs, num_bins=255):
        super().__init__()
        self.loc = loc # [batch_size, num_channels, height, width, num_components]
        self.log_scale = torch.clamp(log_scale, min=-7) # [batch_size, num_channels, height, width, num_components]
        self.logit_probs = logit_probs # [batch_size, num_channels, height, width, num_components]
        self.num_bins = num_bins # int

    def log_prob(self, x):
        # x.shape = [batch_size, num_channels, height, width]
        # x is in [-1, 1]
        # compute log prob in the bin
        # assert torch.all(x >= 0) and torch.all(x <= 1)
        assert torch.all(x >= -1) and torch.all(x <= 1)
        x = x.unsqueeze(-1) # [batch_size, num_channels, height, width, 1]
        # x = x.unsqueeze(-1)*2 -1 # rescale to [-1, 1]
        # printarr(x, self.loc, self.log_scale, self.logit_probs)
        centered_x = x - self.loc
        inv_log_scale = torch.exp(-self.log_scale)
        
        max_x = (centered_x + 1/self.num_bins) * inv_log_scale
        cdf_plus = torch.sigmoid(max_x)
        min_x = (centered_x - 1/self.num_bins) * inv_log_scale
        cdf_minus = torch.sigmoid(min_x)
        # left extreme (x=0)
        log_cdf_left = max_x - F.softplus(max_x)
        # right extreme (x=255)
        log_cdf_right = -F.softplus(min_x)
        log_cdf = torch.log(torch.clamp(cdf_plus - cdf_minus, min=1e-12))

        mid_in = inv_log_scale * centered_x
        log_pdf_mid = mid_in - self.log_scale - 2. * F.softplus(mid_in) #- torch.log(torch.tensor(self.num_bins/2))

        cond_ = (cdf_plus-cdf_minus < 1e-5).float()
        log_cdf = cond_ * log_pdf_mid + (1-cond_) * log_cdf
        left_cond = (x < -0.999).float()
        log_probs = left_cond * log_cdf_left + (1-left_cond) * log_cdf
        right_cond = (x > 0.999).float()
        log_probs = right_cond * log_cdf_right + (1-right_cond) * log_probs

        log_probs = log_probs + torch.log_softmax(self.logit_probs, dim=-1)
        return torch.sum(torch.logsumexp(log_probs, dim=-1).reshape(x.shape[0], -1), dim=-1)
        
    def sample(self, c, i, j, n_samples=1):
        '''
            locs: [batch, n_components]
            log_scales: [batch, n_components]
            mix_logits: [batch, n_components]
        '''
        # printarr(locs, log_scales, mix_logits)
        locs, log_scales, mix_logits = self.loc[:, c, i, j], self.log_scale[:, c, i, j], self.logit_probs[:, c, i, j]
        m = locs.shape[0] # batch
        scales = torch.exp(torch.clamp(log_scales, min=-7))
        probs = torch.softmax(mix_logits, dim=-1) # batch x n_components
        mix_idx = torch.multinomial(probs, n_samples)# batch x n_samples
        u = torch.rand(m, n_samples).to(probs.get_device()) # batch x n_samples x 1
        u = (1-2e-5) * u + 1e-5
        _mu = torch.gather(locs, dim=-1, index=mix_idx.long()) # batch x n_samples
        _s = torch.gather(scales, dim=-1, index=mix_idx.long())
        samples = _mu + _s * (torch.log(u) - torch.log(1.-u))
        return torch.clamp(samples, min=-1, max=1) # batch x n_samples






        