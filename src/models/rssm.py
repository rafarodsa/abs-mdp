import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import printarr
from src.models.factories import build_model

class RSSM(nn.Module):
    def __init__(self, latent_dim, act_dim, hidden_dim, mlp_cfg):
        super().__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.init_state = nn.Parameter(torch.zeros(1,hidden_dim))
        self.gru = nn.GRU(latent_dim + act_dim, hidden_dim, num_layers=1, batch_first=True, dropout=1.)
        # self.gru = nn.LSTM(latent_dim + act_dim, hidden_dim, num_layers=1, batch_first=True, dropout=1.)
        mlp_cfg.input_dim = hidden_dim
        self.transition = build_model(mlp_cfg)
        self.training = True
        # self.register_buffer('last_hidden', torch.zeros(1,hidden_dim), persistent=False)
    
    def forward(self, input, hidden=None):

        if len(input.shape) == 3:
            batched = True
            batch_size = input.shape[0]
        elif len(input.shape) == 1:
            batched = False
            batch_size = 1
            input = input.unsqueeze(0)
        else: 
            batched = False
            batch_size = 1

        if hidden is None:
            if self.training or batched:

                device = input.get_device()
                hidden = self._init_state(batch_size=batch_size, batched=batched, device=device)
                # hidden = hidden.to('cpu' if device < 0 else f'cuda:{device}')
            else:
                hidden = self.last_hidden
        else:
            assert len(hidden.shape) == 2
            assert hidden.shape[-1] == self.hidden_dim
        h, last_hidden = self.gru(input, hidden)
        self.last_hidden = last_hidden
        z = self.transition(h)
        return z

    def _init_state(self, batch_size=1, batched=False, device=None):
        if batched:
            # init_ = torch.zeros(1, batch_size, self.hidden_dim).to(device)
            init_ = self.init_state.unsqueeze(1).repeat(1, batch_size, 1)
            # init_ = (init_, torch.zeros(1, batch_size, self.hidden_dim).to(device))
        else:
            # init_ = torch.zeros(1, self.hidden_dim).to(device)
            init_ = self.init_state
            # init_ = (init_, torch.zeros(1, self.hidden_dim).to(device))
        return init_

    def reset(self, device=None):
        # print('resetting')
        self.last_hidden = torch.Tensor(self._init_state(device=device))
    
    def train(self, mode=True):
        self.training = mode
        super().train(mode)
    
    def eval(self):
        # print('eval mode')
        self.train(mode=False)
    
    def to(self, device):
        self.device = device
        super().to(device)


def RSSMFactory(cfg):
    return RSSM(
        latent_dim=cfg.latent_dim,
        act_dim=cfg.n_options,
        hidden_dim=cfg.hidden_dim,
        mlp_cfg=cfg.mlp
    )

        
        