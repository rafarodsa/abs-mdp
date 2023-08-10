import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSM(nn.Module):
    def __init__(self, latent_dim, act_dim, hidden_dim):
        super().__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.last_hidden = self._init_state()
        self.gru = nn.GRU(latent_dim + act_dim, hidden_dim, num_layers=1, batch_first=True)
        self.transition = nn.Linear(hidden_dim, hidden_dim)
        self.training = True
    
    def forward(self, input, hidden=None):
        if len(input.shape) == 3:
            batched = True
            batch_size = input.shape[0]
        else: 
            batched = False
            batch_size = 1

        if hidden is None:
            if self.training or batched:
                hidden = self._init_state(batch_size=batch_size, batched=batched)
                hidden = hidden.to(input.get_device())
            else:
                # print('using last_hidden state')
                hidden = self.last_hidden
        else:
            assert len(hidden.shape) == 2
            assert hidden.shape[-1] == self.hidden_dim
        print(f"training {self.training}")
        h, last_hidden = self.gru(input, hidden)
        self.last_hidden = last_hidden
        z = self.transition(F.relu(h))
        return z
    
    def rollout(self, input, hidden):
        if len(input.shape) == 3:
            batched = True
            batch_size = input.shape[0]
        else: 
            batched = False
            batch_size = 1

        if hidden is None:
            if self.training or batched:
                hidden = self._init_state(batch_size=batch_size, batched=batched)
                hidden = hidden.to(input.get_device())
            else:
                # print('using last_hidden state')
                hidden = self.last_hidden
        else:
            assert len(hidden.shape) == 2
            assert hidden.shape[-1] == self.hidden_dim
        h, last_hidden = self.gru(input, hidden)
        self.last_hidden = last_hidden
        z = self.transition(F.relu(h))
        return z, last_hidden

    def _init_state(self, batch_size=1, batched=False, device=None):
        if batched:
            init_ = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        else:
            init_ = torch.zeros(1, self.hidden_dim).to(device)
        return init_

    def reset(self, device=None):
        # print('resetting')
        self.last_hidden = self._init_state(device=device)
    
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
    )

        
        