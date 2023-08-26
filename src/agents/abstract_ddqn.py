'''
    Abstract Double DQN Agent
    Extension of Double DQN Agent to support A(s) action availability mask
    author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
    date: 22 June 2023

'''
import torch
import numpy as np

from pfrl.agents import DoubleDQN
from pfrl.utils.contexts import evaluating
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple  

import pfrl
from pfrl import explorers

from src.utils.printarr import printarr

from pfrl.agents.dqn import _batch_reset_recurrent_states_when_episodes_end

from pfrl.utils.recurrent import (
    get_recurrent_state_at,
    recurrent_state_as_numpy,
)

from pfrl.agents import dqn
from pfrl.utils import evaluating
from pfrl.utils.recurrent import pack_and_forward


from pfrl.replay_buffers import PrioritizedReplayBuffer

from pfrl.utils.batch_states import batch_states

def batch_initset(batch_initset, device='cpu'):
    batch_initset = [t.to(device) if isinstance(t, torch.Tensor) else torch.from_numpy(t).to(device) for t in batch_initset]
    return torch.stack(batch_initset, dim=0)

def batch_experiences(experiences, device, phi, gamma, batch_states=batch_states):
    """Takes a batch of k experiences each of which contains j

    consecutive transitions and vectorizes them, where j is between 1 and n.

    Args:
        experiences: list of experiences. Each experience is a list
            containing between 1 and n dicts containing
              - state (object): State
              - action (object): Action
              - reward (float): Reward
              - is_state_terminal (bool): True iff next state is terminal
              - next_state (object): Next state
        device : GPU or CPU the tensor should be placed on
        phi : Preprocessing function
        gamma: discount factor
        batch_states: function that converts a list to a batch
    Returns:
        dict of batched transitions
    """

    batch_exp = {
        "state": batch_states([elem[0]["state"] for elem in experiences], device, phi),
        "action": torch.as_tensor(
            [elem[0]["action"] for elem in experiences], device=device
        ),
        "reward": torch.as_tensor(
            [
                sum((gamma ** i) * exp[i]["reward"] for i in range(len(exp)))
                for exp in experiences
            ],
            dtype=torch.float32,
            device=device,
        ),
        "next_state": batch_states(
            [elem[-1]["next_state"] for elem in experiences], device, phi
        ),
        "is_state_terminal": torch.as_tensor(
            [
                any(transition["is_state_terminal"] for transition in exp)
                for exp in experiences
            ],
            dtype=torch.float32,
            device=device,
        ),
        "discount": torch.as_tensor(
            [(gamma ** elem[0]['tau']) for elem in experiences],
            dtype=torch.float32,
            device=device,
        ),
        "initset_s": batch_initset(
            [elem[0]["initset_s"] for elem in experiences], device=device
        ),
        "initset_next_s": batch_initset(
            [elem[0]["initset_next_s"] for elem in experiences], device=device
        )

    }
    if all(elem[-1]["next_action"] is not None for elem in experiences):
        batch_exp["next_action"] = torch.as_tensor(
            [elem[-1]["next_action"] for elem in experiences], device=device
        )
    return batch_exp



class AbstractDoubleDQN(DoubleDQN):
    saved_attributes = ("model", "target_model", "optimizer")
    def __init__(self, action_mask_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_mask = action_mask_fn

    def _compute_target_values(self, exp_batch):

        batch_next_state = exp_batch["next_state"] 
        action_mask = (1-exp_batch["initset_next_s"]) * -1e12
        # action_mask = self.action_mask(batch_next_state)
        with evaluating(self.model):
            if self.recurrent:
                next_qout, _ = pack_and_forward(
                    self.model,
                    batch_next_state,
                    exp_batch["next_recurrent_state"],
                )
            else:
                next_qout = self.model(batch_next_state)

        if self.recurrent:
            target_next_qout, _ = pack_and_forward(
                self.target_model,
                batch_next_state,
                exp_batch["next_recurrent_state"],
            )
        else:
            target_next_qout = self.target_model(batch_next_state)


        greedy_actions = (next_qout.q_values + action_mask).argmax(axis=1)

        next_q_max = target_next_qout.evaluate_actions(greedy_actions)

        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        discount = exp_batch["discount"]

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_max


    def act(self, obs: Any, initset: Any =None):
        if initset is None:
            return self.batch_act([obs])[0]
        return self.batch_act([obs], [initset])[0]

    def batch_act(self, batch_obs: Sequence[Any], batch_initset_s: Sequence[Any] = None) -> Sequence[Any]:
        batch_s = self.batch_states(batch_obs, self.device, self.phi)
        if batch_initset_s is not None:
            if isinstance(batch_initset_s[0], np.ndarray):
                batch_initset_s = torch.from_numpy(np.stack(batch_initset_s, axis=0)).to(device=self.device)
            else:
                batch_initset_s = torch.stack(batch_initset_s, dim=0).to(device=self.device)
        else:
            print('Warning: Executing Initset function')
            batch_initset_s = self.action_mask(batch_s)

        action_mask = (1-batch_initset_s) * -1e12
    
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)
            batch_argmax = (batch_av.q_values + action_mask).argmax(-1)
            # printarr(batch_argmax, batch_av.q_values, action_mask)
        if self.training:
            batch_action = [
                self.explorer.select_action(
                    self.t,
                    lambda: batch_argmax[i],
                    action_value=batch_av[i : i + 1],
                    batch_obs=batch_initset_s[i : i + 1],
                )
                for i in range(len(batch_obs))
            ]
            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = batch_argmax
        return batch_action
    
    def observe(self, obs, r, done, reset, info=None):
        self.batch_observe([obs], [r], [done], [reset], [info])


    def batch_observe(self, batch_obs: Sequence[Any], batch_reward: Sequence[float], batch_done: Sequence[bool], batch_reset: Sequence[bool], batch_info: Sequence[Dict] = None) -> None:
        if self.training:
            return self._batch_observe_train(
                batch_obs, batch_reward, batch_done, batch_reset, batch_info
            )
        else:
            return self._batch_observe_eval(
                batch_obs, batch_reward, batch_done, batch_reset, batch_info
            )
    
    def _batch_observe_train(self, batch_obs: Sequence[Any], batch_reward: Sequence[float], batch_done: Sequence[bool], batch_reset: Sequence[bool], batch_info: Sequence[Dict] = None) -> None:
        for i in range(len(batch_obs)):
            self.t += 1
            self._cumulative_steps += 1
            # Update the target network
            if self.t % self.target_update_interval == 0:
                self.sync_target_network()
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer

                transition = {
                    "state": self.batch_last_obs[i],
                    "action": self.batch_last_action[i],
                    "reward": batch_reward[i],
                    "next_state": batch_obs[i],
                    "next_action": None,
                    "is_state_terminal": batch_done[i],
                    "tau": batch_info[i]['tau'],
                    "initset_s": batch_info[i]['initset_s'],
                    "initset_next_s": batch_info[i]['initset_next_s']
                }
                if self.recurrent:
                    transition["recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_prev_recurrent_states, i, detach=True
                        )
                    )
                    transition["next_recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_recurrent_states, i, detach=True
                        )
                    )
                self.replay_buffer.append(env_id=i, **transition)
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)
    
    def _batch_observe_eval(self, batch_obs: Sequence[Any], batch_reward: Sequence[float], batch_done: Sequence[bool], batch_reset: Sequence[bool], batch_tau: Sequence[Dict] = None) -> None:
        if self.recurrent:
            # Reset recurrent states when episodes end
            self.test_recurrent_states = (
                _batch_reset_recurrent_states_when_episodes_end(  # NOQA
                    batch_done=batch_done,
                    batch_reset=batch_reset,
                    recurrent_states=self.test_recurrent_states,
                )
            )

    def update(
        self, experiences: List[List[Dict[str, Any]]], errors_out: Optional[list] = None
    ) -> None:
        """Update the model from experiences

        Args:
            experiences (list): List of lists of dicts.
                For DQN, each dict must contains:
                  - state (object): State
                  - action (object): Action
                  - reward (float): Reward
                  - is_state_terminal (bool): True iff next state is terminal
                  - next_state (object): Next state
                  - weight (float, optional): Weight coefficient. It can be
                    used for importance sampling.
            errors_out (list or None): If set to a list, then TD-errors
                computed from the given experiences are appended to the list.

        Returns:
            None
        """
        has_weight = "weight" in experiences[0][0]
        exp_batch = batch_experiences(
            experiences,
            device=self.device,
            phi=self.phi,
            gamma=self.gamma,
            batch_states=self.batch_states,
        )
        if has_weight:
            exp_batch["weights"] = torch.tensor(
                [elem[0]["weight"] for elem in experiences],
                device=self.device,
                dtype=torch.float32,
            )
            if errors_out is None:
                errors_out = []
        loss = self._compute_loss(exp_batch, errors_out=errors_out)
        if has_weight:
            assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
            self.replay_buffer.update_errors(errors_out)

        self.loss_record.append(float(loss.detach().cpu().numpy()))

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            pfrl.utils.clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optim_t += 1

    def save(self, dirname):
        super().save(dirname)
        self.replay_buffer.save(f'{dirname}/replay_buffer.pt')

    def load(self, dirname):
        super().load(dirname)
        try:
            self.replay_buffer.load(f'{dirname}/replay_buffer.pt')
        except: 
            print('replay buffer not loaded.')


def select_action_epsilon_greedily(batch_obs, epsilon, random_action_func, greedy_action_func):
    if np.random.rand() < epsilon:
        return random_action_func(batch_obs), False
    else:
        return greedy_action_func(), True

class AbstractLinearDecayEpsilonGreedy(explorers.LinearDecayEpsilonGreedy):
    def select_action(self, t, greedy_action_func, action_value=None, batch_obs=None):
        self.epsilon = self.compute_epsilon(t)
        if batch_obs is None:
            raise ValueError("batch_obs must be specified")
        a, greedy = select_action_epsilon_greedily(
            batch_obs, self.epsilon, self.random_action_func, greedy_action_func
        )
        greedy_str = "greedy" if greedy else "non-greedy"
        self.logger.debug("t:%s a:%s %s", t, a, greedy_str)
        # printarr(a)
        return a
    

class AbstractDDQNGrounded(pfrl.agent.Agent):
    def __init__(self, encoder, agent, action_mask, device='cpu'):
        self.agent = agent
        self.encoder = encoder
        self.action_mask = action_mask
        self.gamma = agent.gamma
        self.device = device
        self.agent.device = device
        print(self.gamma)
        
    def act(self, obs, initset=None):
        z = self.encoder(torch.from_numpy(obs).to(self.device))
        action = self.agent.act(z, initset)
        return action
    
    def load(self, dirname):
        self.agent.load(dirname)
    
    def observe(self, obs, reward, done, reset, info):
        z = self.encoder(torch.from_numpy(obs).to(self.device))
        self.agent.observe(z, reward, done, reset, info)
    
    def save(self, dirname):
        self.agent.save(dirname)
    
    def get_statistics(self):
        return self.agent.get_statistics()
        
    
# class GroundedHierarchicalAgent(pfrl.Agent):
#     def __init__(self, agent, encoder, initset_fn):
#         self.agent = agent
#         self._encoder = encoder
#         self._initset_fn = initset_fn
    
#     def act(self, obs):
#         z = self.encoder(obs)
#         return self.agent.act(z)
    
#     def observe(self, obs, reward, done, reset, info):
#         z = self.encoder(obs)
#         self.agent.observe(z, reward, done, reset, info)

#     @property
#     def encoder(self):
#         return self._encoder
    
#     @property
#     def initset_fn(self):
#         return self._initset_fn

#     @encoder.setter
#     def encoder(self, encoder):
#         self._encoder = encoder
    
#     @initset_fn.setter
#     def initset_fn(self, initset_fn):
#         self._initset_fn = initset_fn