"""
    Train an Abstract MDP model
    based on an abstraction function $phi(s)$,
    abstract transition function $T(z' | z, a)$,
    abstract initiation classifiers $I(z)$,
    and grounding function $g(s|z)$.
    
    author: Rafael Rodriguez-Sanchez
    date: 7 March 2023
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.factories import build_model
from src.absmdp.infomax import InfomaxAbstraction
from src.utils.symlog import symlog

import logging
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('medium')


"""
    Abstract MDP
    author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
    date: 1 December 2022
"""
import numpy as np
import torch

from argparse import Namespace
from src.utils.symlog import symexp

class AbstractMDP:
    def __init__(self, mdp_trainer, data):
        self.data = data
        self.trainer = mdp_trainer
        self.n_options = mdp_trainer.n_options
        self.latent_dim = mdp_trainer.latent_dim
        self.obs_dim = mdp_trainer.obs_dim
        self.initial_states = None

        self._prepare_initial_states()
        self.prepare_models()

    def plan(self, initial_state, goal):
        pass
    
    def get_actions(self):
        return list(range(self.n_options)) 

    def get_initial_states(self, n_samples=1):
        if self.initial_states is None:
             self._prepare_initial_states()
        # sample
        sample = np.random.choice(len(self.data), n_samples, replace=True)
        sample = self.initial_states[sample]
        return sample
    
    def action_to_one_hot(self, action):
        return torch.nn.functional.one_hot(action, self.n_options)

    def transition(self, state, action):
        if len(state.size()) > 1:
            input = torch.cat([state, self.action_to_one_hot(action)], dim=-1)
        else: 
            input = torch.cat([state, self.action_to_one_hot(action)], dim=0)
        return self.trainer.transition(input)
    
    def reward(self, state, action, next_state):
        r_in = torch.cat([state, self.action_to_one_hot(action), next_state], dim=-1)
        return symexp(self.reward_fn(r_in))

    def initiation_set(self, state):
        return self.init_classifier(state)

    def encoder(self, ground_state):
        return self.encoder(ground_state)
            
    def ground(self, abstract_state):
        return self.trainer.distribution(abstract_state)

    def gamma(self, state, action):
        return self.gamma ** self.tau(state, action)

    @property
    def n_options(self):
        return self.n_options

    def save(self, path):
        data = [self.data[i] for i in range(len(self.data))]
        torch.save([data, self.trainer], path)
    
    @staticmethod
    def load(path):
        data, trainer = torch.load(path)
        return AbstractMDP(trainer, data)
    
    def _prepare_initial_states(self):
        self.initial_states = torch.stack([d.obs for d in self.data if d.p0 == 1])
    
    def prepare_models(self):
        mdp_trainer = self.mdp_trainer
        self.encoder = mdp_trainer.phi.encoder
        self.encoder.requires_grad_(False)
        self.grounding = mdp_trainer.phi.grounding
        self.grounding.requires_grad_(False)
        self.transition = mdp_trainer.phi.transition
        self.transition.requires_grad_(False)
        self.reward_fn = mdp_trainer.reward_fn
        self.reward_fn.requires_grad_(False)
        self.gamma = mdp_trainer.gamma
        self.gamma.requires_grad_(False)
        self.init_classifier = mdp_trainer.phi.init_classifier
        self.init_classifier.requires_grad_(False)
        self.tau = mdp_trainer.tau
        self.tau.requires_grad_(False)
