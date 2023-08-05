import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSM(nn.Module):
    def __init__(self, latent_dim, act_dim, hidden_dim):
        super().__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.gru = nn.GRU(latent_dim + act_dim, hidden_dim, num_layers=1, batch_first=True)
        self.transition = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, input, hidden=None):
        if len(input.shape) == 3:
            batched = True
            batch_size = input.shape[0]
        else: 
            batched = False
            batch_size = 1

        if hidden is None:
            hidden = self._init_state(batch_size=batch_size, batched=batched)
        else:
            assert len(hidden.shape) == 2
            assert hidden.shape[-1] == self.hidden_dim
        h, _ = self.gru(input, hidden)
        z = self.transition(F.relu(h))
        return z
    
    def _init_state(self, batch_size=1, batched=False):
        if batched:
            init_ = torch.zeros(1, batch_size, self.hidden_dim)
        else:
            init_ = torch.zeros(1, self.hidden_dim)
        return init_

def RSSMFactory(cfg):
    return RSSM(
        latent_dim=cfg.latent_dim,
        act_dim=cfg.n_options,
        hidden_dim=cfg.hidden_dim,
    )

        
        