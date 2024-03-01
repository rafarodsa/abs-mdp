import torch
import pfrl
import numpy as np
import jax

from contextlib import contextmanager
import pfrl.agents
from pfrl.utils.recurrent import (
    concatenate_recurrent_states,
    flatten_sequences_time_first,
    get_recurrent_state_at,
    mask_recurrent_state_at,
    one_step_forward,
    pack_and_forward,
)

import itertools

from pfrl.agents.ppo import _add_log_prob_and_value_to_episodes, _limit_sequence_length, _add_log_prob_and_value_to_episodes_recurrent, _compute_explained_variance



def _add_advantage_and_value_target_to_episodes(episodes, gamma, lambd):
    """Add advantage and value target values to a list of episodes."""
    for episode in episodes:
        _add_advantage_and_value_target_to_episode(episode, gamma=gamma, lambd=lambd)

def _add_advantage_and_value_target_to_episode(episode, gamma, lambd):
    """Add advantage and value target values to an episode."""
    adv = 0.0
    for transition in reversed(episode):
        td_err = (
            transition["reward"]
            # + ((gamma**transition["tau"]) * transition["nonterminal"] * transition["next_v_pred"])
            + (gamma * transition["nonterminal"] * transition["next_v_pred"])
            - transition["v_pred"]
        )
        # adv = td_err + ((gamma * lambd))**transition['tau'] * adv
        adv = td_err + ((gamma * lambd)) * adv
        transition["adv"] = adv
        transition["v_teacher"] = adv + transition["v_pred"]

def _make_dataset(
    episodes, model, phi, batch_states, obs_normalizer, gamma, lambd, device
):
    """Make a list of transitions with necessary information."""

    _add_log_prob_and_value_to_episodes(
        episodes=episodes,
        model=model,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer=obs_normalizer,
        device=device,
    )

    _add_advantage_and_value_target_to_episodes(episodes, gamma=gamma, lambd=lambd)

    return list(itertools.chain.from_iterable(episodes))

def _make_dataset_recurrent(
    episodes,
    model,
    phi,
    batch_states,
    obs_normalizer,
    gamma,
    lambd,
    max_recurrent_sequence_len,
    device,
):
    """Make a list of sequences with necessary information."""

    _add_log_prob_and_value_to_episodes_recurrent(
        episodes=episodes,
        model=model,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer=obs_normalizer,
        device=device,
    )

    _add_advantage_and_value_target_to_episodes(episodes, gamma=gamma, lambd=lambd)

    if max_recurrent_sequence_len is not None:
        dataset = _limit_sequence_length(episodes, max_recurrent_sequence_len)
    else:
        dataset = list(episodes)

    return dataset


        
class PPO(pfrl.agents.PPO):
    
    def _update_if_dataset_is_ready(self):
        dataset_size = (
            sum(len(episode) for episode in self.memory)
            + len(self.last_episode)
            + (
                0
                if self.batch_last_episode is None
                else sum(len(episode) for episode in self.batch_last_episode)
            )
        )
        if dataset_size >= self.update_interval:
            self._flush_last_episode()
            if self.recurrent:
                dataset = _make_dataset_recurrent(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    max_recurrent_sequence_len=self.max_recurrent_sequence_len,
                    device=self.device,
                )
                self._update_recurrent(dataset)
            else:
                dataset = _make_dataset(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    device=self.device,
                )
                assert len(dataset) == dataset_size
                self._update(dataset)
            self.explained_variance = _compute_explained_variance(
                list(itertools.chain.from_iterable(self.memory))
            )
            self.memory = []
    def observe(self, batch_obs, batch_reward, batch_done, batch_reset, info=None):
        self.batch_observe([batch_obs], [batch_reward], [batch_done], [batch_reset], [info])
        
    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset, info=None):
        # print(f'Batch Observe PPO. Training {self.training}')
        if self.training:
            self._batch_observe_train(batch_obs, batch_reward, batch_done, batch_reset, info)
        else:
            self._batch_observe_eval(batch_obs, batch_reward, batch_done, batch_reset)

    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset, batch_info=None):
        assert self.training

        for i, (state, action, reward, next_state, done, reset, info) in enumerate(
            zip(
                self.batch_last_state,
                self.batch_last_action,
                batch_reward,
                batch_obs,
                batch_done,
                batch_reset,
                batch_info
            )
        ):

            if state is not None:
                assert action is not None
                transition = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "nonterminal": 0.0 if done else 1.0,
                    "tau": info["tau"]
                }
                if self.recurrent:
                    transition["recurrent_state"] = get_recurrent_state_at(
                        self.train_prev_recurrent_states, i, detach=True
                    )
                    transition["next_recurrent_state"] = get_recurrent_state_at(
                        self.train_recurrent_states, i, detach=True
                    )
                self.batch_last_episode[i].append(transition)
            if done or reset:
                assert self.batch_last_episode[i]
                self.memory.append(self.batch_last_episode[i])
                self.batch_last_episode[i] = []
            self.batch_last_state[i] = None
            self.batch_last_action[i] = None

        self.train_prev_recurrent_states = None

        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i
                for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
                if done or reset
            ]
            if indices_that_ended:
                self.train_recurrent_states = mask_recurrent_state_at(
                    self.train_recurrent_states, indices_that_ended
                )

        self._update_if_dataset_is_ready()


class AbstractPPO(pfrl.agent.Agent):
    def __init__(self, encoder, agent, action_mask=None, device='cpu', recurrent=False):
        self.agent = agent
        self.encoder = encoder
        self.action_mask = action_mask
        self.gamma = agent.gamma
        self.device = device
        self.agent.device = device
        self.recurrent = recurrent
        if self.recurrent:
            assert isinstance(self.encoder, tuple) and len(self.encoder) == 2
            self.encoder, self.transition = encoder
            self._state = None
        print(self.gamma)

    
    def _preprocess(self, obs):
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs.copy()).to(self.device)
        elif isinstance(obs, torch.Tensor):
            return obs.to(self.device)
        return obs

    def preprocess(self, obs):
        return jax.tree_map(self._preprocess, obs)
        
    def act(self, obs, initset=None):
        # obs = jax.tree_map(self.preprocess, obs)
        obs = self.preprocess(obs)
        if not isinstance(obs, list):
            obs = [obs]
        with torch.no_grad():
            if self.recurrent and self._state is None:
                self._state = self.transition.feats._init_state(len(obs), batched=True)
                self._hidden = self._state
            z = jax.tree_map(self.encoder, obs, is_leaf=lambda x: isinstance(x, dict))
            feats = [torch.cat([z, self._state[0, i]], dim=-1) for i, z in enumerate(z)] if self.recurrent else z
            action = self.agent.batch_act(feats)
            self.update_hidden(z, action)
        return action if len(obs) > 1 else action[0]
    
    @torch.no_grad()
    def update_hidden(self, z, action):
        if not self.recurrent:
            return 
        _inp = torch.stack([torch.cat([_z, torch.nn.functional.one_hot(torch.tensor(a), 3).to(_z.device)], dim=-1) for _z, a in zip(z, action)])
        hidden, last_hidden = self.transition.feats.gru(_inp.unsqueeze(1), self._state)
        hidden = self.transition.feats.transition(hidden)
        self._state = hidden.permute(1, 0, 2)
        self._hidden = last_hidden

    @torch.no_grad()
    def batch_reset(self, mask):
        if not self.recurrent:
            return 
        for i in range(len(mask)):
            if not mask[i]:
                self._state[:, i] = torch.Tensor(self.transition.feats._init_state(batched=True)[:, 0])
                self._hidden[:, i] = torch.Tensor(self._state[:, i])


    def load(self, dirname):
        self.agent.load(dirname)
    
    def observe(self, obs, reward, done, reset, info):
        obs = self.preprocess(obs)
        if not isinstance(obs, list):
            obs = [obs]
            reward = [reward]
            done = [done]
            reset = [reset]
            info = [info]
        with torch.no_grad():
            z = jax.tree_map(lambda obs: self.encoder(obs), obs, is_leaf=lambda x: isinstance(x, dict))
        self.agent.batch_observe(z, reward, done, reset, info)
    
    def save(self, dirname):
        self.agent.save(dirname)
    
    def get_statistics(self):
        return self.agent.get_statistics()
    
    @contextmanager
    def eval_mode(self):
        orig_mode = self.agent.training
        try:
            self.agent.training = False
            yield
        finally:
            self.agent.training = orig_mode
    
    def getattr(self, name):
        return getattr(self.agent, name)
    
# from src.agents.agent import Agent
# class PPOAgent(Agent):
