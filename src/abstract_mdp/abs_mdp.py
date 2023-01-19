"""
    Abstract MDP
    author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
    date: 1 December 2022
"""
import numpy as np
import torch

from argparse import Namespace



class AbstractMDP:
    def __init__(self, mdp_trainer, data):
        self.data = data
        self.trainer = Namespace(**{
            "encoder": mdp_trainer.encoder,
            "decoder": mdp_trainer.decoder,
            "transition": mdp_trainer.transition,
            "reward_fn": mdp_trainer.reward_fn,
            "init_classifier": mdp_trainer.init_classifier,
            "n_options": mdp_trainer.n_options,
            "latent_dim": mdp_trainer.latent_dim,
            "obs_dim": mdp_trainer.obs_dim
        })
        self.initial_states = None

    def plan(self, goal):
        pass
    
    def get_actions(self):
        return list(range(self.trainer.n_options)) 

    def get_initial_states(self, n_samples=1):
        if not self.initial_states:
             self._prepare_initial_states()
        # sample
        sample = np.random.choice(len(self.data), n_samples, replace=True)
        sample = self.initial_states[sample]
        return sample

    def _prepare_initial_states(self):
        self.initial_states = np.array([d[0] for d in self.data])

    def transition(self, state, action, executed):
        
        if len(state.size()) > 1:
            actions_ = torch.nn.functional.one_hot(action, self.trainer.n_options)
            executed_ = executed.unsqueeze(1)
            input = torch.cat([state, actions_, executed_], dim=-1)
        else: 
            input = torch.cat([state, torch.Tensor(action)], self.trainer.n_options)
        with torch.no_grad():
            return self.trainer.transition(input)
    
    def reward(self, state, action, next_state):
        pass

    def initiation_set(self, state):
        pass

    def encoder(self, ground_state):
        with torch.no_grad():
            return self.trainer.encoder(ground_state)[0].squeeze()

    def ground(self, abstract_state):
        with torch.no_grad():
            return self.trainer.decoder(abstract_state)[0].squeeze()

    def gamma(self, state, action, next_state):
        pass

    @property
    def n_options(self):
        return self.trainer.n_options

    def save(self, path):
        data = [self.data[i] for i in range(len(self.data))]
        torch.save([data, self.trainer], path)
    
    @staticmethod
    def load(path):
        data, trainer = torch.load(path)
        return AbstractMDP(trainer, data)