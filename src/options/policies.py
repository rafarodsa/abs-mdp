'''
    Option Policies
    author: Rafael Rodriguez-Sanchez
'''

import pfrl.policies as policies
import torch
import numpy as np


def initiation_from_options(options):
    def initiation(state):
        return torch.from_numpy(np.array([o.initiation(state) for o in options]).T)
    return initiation

class SoftmaxCategoricalHeadInitiation(policies.SoftmaxCategoricalHead):
    def __init__(self, initiation_set_fn, **kwargs):
        super().__init__(**kwargs)
        self.initiation_set_fn = initiation_set_fn
    
    def _compute_initiation(self, state):
        return self.initiation_set_fn(state)

    def forward(self, s, logits):
        initiation = self._compute_initiation(s.numpy())
        log_succ = torch.clamp(torch.log(initiation), min=-1e20)
        log_succ = log_succ - torch.logsumexp(log_succ, dim=1, keepdim=True)

        # logits = torch.where(initiation, logits, torch.full_like(logits, -1e20))
        log_prob = torch.log_softmax(logits, dim=1) + log_succ
        dist = torch.distributions.Categorical(logits=log_prob)
        return dist 
    

class SoftmaxCategoricalHeadOptions(SoftmaxCategoricalHeadInitiation):
    def __init__(self, options, **kwargs):
        super().__init__(initiation_from_options(options), **kwargs)
        self.options = options

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
    
class OptionPolicyInit(torch.nn.Module):
    def __init__(self, features, policy_head, value_head, initiation_set_fn):
        super().__init__()
        self.features = features
        self.policy_head = policy_head
        self.value_head = value_head
        self.head = SoftmaxCategoricalHeadInitiation(initiation_set_fn)

    def forward(self, s):
        h = self.features(s)
        return self.head(s, self.policy_head(h)), self.value_head(h)
