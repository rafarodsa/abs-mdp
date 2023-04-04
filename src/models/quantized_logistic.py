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
        

    


        