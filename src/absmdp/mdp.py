"""
    Build gym environment from learned Abstract MDP
    
    author: Rafael Rodriguez-Sanchez
    date: 1 May 2023
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import zipfile
import gym
from src.absmdp.infomax import InfomaxAbstraction
from src.utils.symlog import symlog, symexp

from src.absmdp.infomax_attn import AbstractMDPTrainer
from src.utils.printarr import printarr

import logging
logger = logging.getLogger(__name__)


class AbstractMDP(gym.Env):
    def __init__(
                 self, 
                 encoder,
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
        self.transition_fn = transition
        self.grounding_fn = grounding
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
    
    def reset(self, state=None):
        # sample initial state
        self._state = self.get_initial_states() if state is None else self.encode(torch.from_numpy(state)).squeeze(0)
        return self._state.numpy()

    def render(self):
        pass

    def step(self, action):
        next_s = self.transition(self.state, action)
        r = self.reward(self.state, action, next_s).item()
        done = False
        tau =  self.tau(self.state, action)
        info = {'expected_length': tau.item()}
        self._state = next_s
        return next_s.numpy(), r, done, info
    
    # def grounding(self, s, z):
        

    @property
    def state(self):
        return self._state

    def get_actions(self):
        return list(range(self.n_options)) 

    def get_initial_states(self, n_samples=1):
        if self.initial_states is None:
             self._prepare_initial_states()
        # sample
        sample = np.random.choice(len(self.initial_states), n_samples, replace=True)
        sample = self.initial_states[sample]
        return self.encode(sample).squeeze(0)
    
    def _action_to_one_hot(self, action):
        if isinstance(action, (int, np.int64)):
            action = torch.tensor(action)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        return torch.nn.functional.one_hot(action, self.n_options)

    def transition(self, state, action):
        batch = state.shape[0] if len(state.size()) > 1 else 1
        
        if len(state.size()) == 1:
            input = torch.cat([state, self._action_to_one_hot(action)], dim=-1)
        else: 
            input = torch.cat([state, self._action_to_one_hot(action)], dim=0)
        t = self.transition_fn.distribution(input)
        # print('sampling')
        return t.sample()[0] + state
        # return t.mean + state
    
    def reward(self, state, action, next_state):
        a_ = self._action_to_one_hot(action)
        r_in = torch.cat([state, a_, next_state], dim=-1)
        return torch.clamp(symexp(self.reward_fn(r_in)), max=1e10, min=-1e10)
    
    def tau(self, state, action):
        in_ = torch.cat([state, self._action_to_one_hot(action)], dim=-1)
       
        return symexp(self.tau_fn(in_))

    def initiation_set(self, state):
        return self.init_classifier(state)

    def encode(self, ground_state):
        return self.encoder(ground_state)
    
    def ground(self, state):
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        _in = torch.cat([state, torch.zeros(state.shape[0], self._n_options)], dim=-1)
        return self.grounding_fn.distribution(_in)
        
    def gamma(self, state, action):
        return self._gamma ** self.tau(state, action)

    @property
    def n_options(self):
        return self._n_options
    
    @property
    def latent_dim(self):
        return self._latent_dim
    
    @staticmethod
    def load(model, data):
        mdp_elems = AbstractMDP._prepare_models(model)
        initial_states = torch.stack([d.obs for d in data if d.p0 == 1])
        mdp_elems['initial_states'] = initial_states
        return AbstractMDP(**mdp_elems)
    
    
    def _prepare_initial_states(self):
        self.initial_states = torch.stack([d.obs for d in self.data if d.p0 == 1])
    
    @staticmethod
    def _prepare_models(mdp_trainer):
        encoder = mdp_trainer.phi.encoder
        encoder.requires_grad_(False)
        grounding = mdp_trainer.phi.grounding
        grounding.requires_grad_(False)
        transition = mdp_trainer.phi.transition
        transition.requires_grad_(False)
        reward_fn = mdp_trainer.reward
        reward_fn.requires_grad_(False)
        init_classifier = mdp_trainer.initsets
        init_classifier.requires_grad_(False)
        tau = mdp_trainer.tau
        tau.requires_grad_(False)
        mdp_elems = {
                     'encoder': encoder, 
                     'grounding': grounding, 
                     'transition': transition, 
                     'reward': reward_fn, 
                     'init_classifier': init_classifier, 
                     'tau': tau,
                     'n_options': mdp_trainer.n_options,
                     'latent_dim': mdp_trainer.phi.latent_dim,
                     'obs_dim': mdp_trainer.phi.obs_dim,
                     'gamma': mdp_trainer.phi.cfg.data.gamma
                    }
        return mdp_elems


class AbstractMDPCritic(gym.Env):
    def __init__(
                 self, 
                 encoder,
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
                 rssm=False
                ):
        
        self.encoder = encoder
        self.transition_fn = transition
        self.grounding_fn = grounding
        self.reward_fn = reward
        self.tau_fn = tau
        self.initial_states = initial_states
        self._n_options = n_options
        self._latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.init_classifier = init_classifier
        self._gamma = gamma
        self.rssm = rssm
        self.device='cpu'

        self.modules = [encoder, grounding, transition, reward, init_classifier, tau]
        for m in self.modules:
            m.eval()
            m.to(self.device)

        self.transition_fn.feats.eval()
        print(self.transition_fn.feats.training)

        # define action and observation space
        self.action_space = gym.spaces.Discrete(n_options)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32)

        self.reset()
    
    def reset(self, state=None):
        if self.rssm:
            self.transition_fn.feats.reset(self.device) # reset hidden state.
        self._state = self.get_initial_states() if state is None else self.encode(torch.from_numpy(state).to(self.device)).squeeze(0)
        self._state = self._state.to(self.device)
        return self._state.cpu().numpy()

    def render(self):
        pass

    def step(self, action):
        action = action.to(self.device)
        next_s = self.transition(self.state, action)
        r = self.reward(self.state, action, next_s).item()
        done = False
        tau =  self.tau(self.state, action)
        info = {'tau': tau.item()}
        self._state = next_s
        return next_s.cpu().numpy(), r, done, info

    @property
    def state(self):
        return self._state
    
    def grounding(self, s, z, std=0.5):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s)
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        
        _z = self.encoder(s)
        return torch.exp(-(((z - _z)/std) ** 2).sum(-1))

    def get_actions(self):
        return list(range(self.n_options)) 

    def get_initial_states(self, n_samples=1):
        if self.initial_states is None:
             self._prepare_initial_states()
        # sample
        sample = np.random.choice(len(self.initial_states), n_samples, replace=True)
        sample = self.initial_states[sample]
        return self.encode(sample).squeeze(0)
    
    def _action_to_one_hot(self, action):
        if isinstance(action, (int, np.int64)):
            action = torch.tensor(action)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        return torch.nn.functional.one_hot(action, self.n_options)

    def transition(self, state, action):
        batch = state.shape[0] if len(state.size()) > 1 else 1
        if len(state.size()) == 1:
            input = torch.cat([state, self._action_to_one_hot(action)], dim=-1)
        else: 
            input = torch.cat([state, self._action_to_one_hot(action)], dim=0)
        
        t = self.transition_fn.distribution(input.unsqueeze(0))
        # delta_s = t.sample()[0]
        delta_s = t.mean
        next_s = delta_s + state
        return next_s[0]
        # return t.mean + state
    
    def reward(self, state, action, next_state):
        a_ = self._action_to_one_hot(action)
        r_in = torch.cat([state, a_, next_state], dim=-1)
        return torch.clamp(symexp(self.reward_fn(r_in)), max=1e10, min=-1e10)
    
    def tau(self, state, action):
        in_ = torch.cat([state, self._action_to_one_hot(action)], dim=-1)
       
        return symexp(self.tau_fn(in_))

    def initiation_set(self, state):
        return self.init_classifier(state)

    def encode(self, ground_state):
        return self.encoder(ground_state)
    
    def ground(self, state):
        raise NotImplemented
        
    def gamma(self, state, action):
        return self._gamma ** self.tau(state, action)

    @property
    def n_options(self):
        return self._n_options
    
    @property
    def latent_dim(self):
        return self._latent_dim
    
    @staticmethod
    def load(model, data, rssm=False):
        mdp_elems = AbstractMDPCritic._prepare_models(model)
        initial_states = torch.stack([d.obs for d in data if d.p0 == 1]) if not rssm else torch.stack([d.obs[0] for d in data])
        mdp_elems['initial_states'] = initial_states
        return AbstractMDPCritic(**mdp_elems, rssm=rssm)
    
    
    def _prepare_initial_states(self):
        self.initial_states = torch.stack([d.obs for d in self.data if d.p0 == 1])

    def to(self, device):
        self.device = device
        for m in self.modules:
            m.to(device)

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
        mdp_elems = {
                     'encoder': encoder, 
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
