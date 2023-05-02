'''
    Option Policies
    author: Rafael Rodriguez-Sanchez
'''

import pfrl.policies as policies
import torch
import numpy as np
from src.utils.printarr import printarr



class AbstractPolicyWrapper(torch.nn.Module):
    def __init__(self, policy, encoder):
        super().__init__()
        self.policy = policy
        self.encoder = encoder
    
    def forward(self, s):
        return self.policy(self.encoder(s))

class SoftmaxCategoricalHeadOptions(policies.SoftmaxCategoricalHead):
    def __init__(self, options, **kwargs):
        super().__init__(**kwargs)
        self.options = options

    def _compute_initiation(self, state):
        initiation = np.array([o.initiation(state) for o in self.options])
        return torch.from_numpy(initiation).T

    def forward(self, s, logits):
        initiation = self._compute_initiation(s.numpy())
        logits = torch.where(initiation == 1, logits, torch.full_like(logits, -1e20))
        return super().forward(logits)
    
class OptionPolicy(torch.nn.Module):
    def __init__(self, features, policy_head, value_head, options):
        super().__init__()
        self.features = features
        self.policy_head = policy_head
        self.value_head = value_head
        self.options = options
        self.head = SoftmaxCategoricalHeadOptions(options)

    def forward(self, s):
        h = self.features(s)
        return self.head(s, self.policy_head(h)), self.value_head(h)
    
