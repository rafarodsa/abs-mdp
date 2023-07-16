from src.absmdp.mdp import AbstractMDPCritic

import torch
import gym
import numpy as np

from src.utils.printarr import printarr

class DiscreteAbstractMDP(AbstractMDPCritic):
    def __init__(
                 self, 
                 encoder,
                 quantizer,
                 grounding,
                 transition,
                 reward,
                 tau,
                 initial_states,
                 n_options,
                 latent_dim,
                 obs_dim,
                 init_classifier,
                 gamma=0.99,
                ):
        
        self.encoder = encoder
        self.quantizer = quantizer
        self.transition_fn = transition
        self.grounding = grounding
        self.reward_fn = reward
        self.tau_fn = tau
        self.initial_states = initial_states
        self._n_options = n_options
        self._latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.init_classifier = init_classifier
        self._gamma = gamma

        # define action and observation space
        self.action_space = gym.spaces.Discrete(n_options)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32)

        self.reset()
    
    
    def load(model, data):
        mdp_elems = DiscreteAbstractMDP._prepare_models(model)
        initial_states = torch.stack([d.obs for d in data if d.p0 == 1])
        mdp_elems['initial_states'] = initial_states
        return DiscreteAbstractMDP(**mdp_elems)
    
    @staticmethod
    def _prepare_models(mdp_trainer):
        encoder = mdp_trainer.encoder
        encoder.requires_grad_(False)
        grounding = mdp_trainer.grounding
        grounding.requires_grad_(False)
        transition = mdp_trainer.transition
        transition.requires_grad_(False)
        reward_fn = mdp_trainer.reward_fn
        reward_fn.requires_grad_(False)
        init_classifier = mdp_trainer.initsets
        init_classifier.requires_grad_(False)
        tau = mdp_trainer.tau
        tau.requires_grad_(False)
        
        # adding quantizer
        quantizer = mdp_trainer.quantizer
        quantizer.requires_grad_(False)
        
        mdp_elems = {
                     'encoder': encoder, 
                     'quantizer': quantizer,
                     'grounding': grounding, 
                     'transition': transition, 
                     'reward': reward_fn, 
                     'init_classifier': init_classifier, 
                     'tau': tau,
                     'n_options': mdp_trainer.n_options,
                     'latent_dim': mdp_trainer.latent_dim,
                     'obs_dim': mdp_trainer.obs_dim,
                     'gamma': mdp_trainer.cfg.data.gamma
                    }
        return mdp_elems
    
    def encode(self, ground_state):
        # print('encoding')
        b_dims = ground_state.shape[:-2]
        z = super().encode(ground_state)
        
        z_q, _ =  self.quantizer(z.reshape(-1, self.quantizer.num_factors, self.quantizer.embedding_size))
        return z_q.reshape(*b_dims, -1)
    
    def transition(self, state, action):
        b_dims = state.shape[:-1]
        state = state.reshape(*b_dims, -1)
        # printarr(state, action)
        if len(state.size()) == 1:
            input = torch.cat([state, self._action_to_one_hot(action)], dim=-1)
        else: 
            input = torch.cat([state, self._action_to_one_hot(action)], dim=0)
        t = self.transition_fn.distribution(input)
        # print('sampling')
        # return t.sample()[0]
        # print('mode')
        # next_s = t.mode
        next_s = t.sample()
        return next_s.reshape(*b_dims, -1)
    
    # def initiation_set(self, state):
    #     # b_dims = state.shape[:-2]
    #     # state = state.reshape(*b_dims, -1)
    #     return self.init_classifier(state)
    
    # def reward(self, state, action, next_state):
    #     # b_dims = state.shape[:-2]
    #     # state, next_state = state.reshape(*b_dims, -1), next_state.reshape(*b_dims, -1)
    #     return super().reward(state, action, next_state)
    
    def tau(self, state, action):
        # b_dims = state.shape[:-2]
        # state = state.reshape(*b_dims, -1)
        _t = super().tau(state, action)
        return torch.tensor(1)#_t.abs()
