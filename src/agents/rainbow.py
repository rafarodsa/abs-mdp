import ipdb
import torch
import random
import numpy as np
import pfrl
from pfrl import nn as pnn
from pfrl import replay_buffers
from pfrl import agents, explorers
from pfrl.wrappers import atari_wrappers
from src.agents.dueling_dqn import DistributionalDuelingDQN
from pfrl.utils import batch_states as pfrl_batch_states
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple  

from pfrl.utils.recurrent import (
    get_recurrent_state_at,
    recurrent_state_as_numpy,
)

class AbstractCategoricalDDQN(agents.CategoricalDoubleDQN):
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
    
    def _batch_observe_eval(self, batch_obs: Sequence[Any], batch_reward: Sequence[float], batch_done: Sequence[bool], batch_reset: Sequence[bool], batch_info: Sequence[Dict] = None) -> None:
        self._batch_observe_eval(batch_obs, batch_reward, batch_done, batch_reset)
    
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
        exp_batch = self.batch_experiences(
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
            assert isinstance(self.replay_buffer, replay_buffers.PrioritizedReplayBuffer)
            self.replay_buffer.update_errors(errors_out)

        self.loss_record.append(float(loss.detach().cpu().numpy()))

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            pfrl.utils.clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optim_t += 1
    
    @staticmethod
    def batch_experiences(experiences, device, phi, gamma, batch_states=None):
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
                    sum((gamma ** (exp[i-1]['tau']*(float(i>0)))) * exp[i]["reward"] for i in range(len(exp)))
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
                [gamma ** sum([exp[i]['tau'] for i in range(len(exp))]) for exp in experiences],
                dtype=torch.float32,
                device=device,
            ),
        }
        if all(elem[-1]["next_action"] is not None for elem in experiences):
            batch_exp["next_action"] = torch.as_tensor(
                [elem[-1]["next_action"] for elem in experiences], device=device
            )
        return batch_exp

class Rainbow:
    def __init__(self, q_func_model, n_actions=8, n_atoms=51, v_min=-10, v_max=10, noisy_net_sigma=0.5, lr=1e-5, 
                 n_steps=3, betasteps=1, replay_start_size=1000, replay_buffer_size=int(1e6), gpu=-1,
                 goal_conditioned=False, gamma=0.995, target_update_interval=32_000, update_interval=5, use_custom_batch_states=True, explorer=None):
        self.n_actions = n_actions
        self.goal_conditioned = goal_conditioned
        self.use_custom_batch_states = use_custom_batch_states
        self.gamma=gamma

        self.q_func = DistributionalDuelingDQN(q_func_model, n_actions, n_atoms, v_min, v_max)
        pnn.to_factorized_noisy(self.q_func, sigma_scale=noisy_net_sigma)

        explorer = explorers.Greedy() if explorer is None else explorer
        opt = torch.optim.Adam(self.q_func.parameters(), lr, eps=1.5e-4)

        self.rbuf = replay_buffers.PrioritizedReplayBuffer(
            replay_buffer_size,
            alpha=0.5, 
            beta0=0.4,
            betasteps=betasteps,
            num_steps=n_steps,
            normalize_by_max="memory"
        )

        self.agent = AbstractCategoricalDDQN(
            self.q_func,
            opt,
            self.rbuf,
            gpu=gpu,
            gamma=gamma,
            explorer=explorer,
            minibatch_size=32,
            replay_start_size=replay_start_size,
            target_update_interval=target_update_interval,
            update_interval=update_interval,
            batch_accumulator="mean",
            phi=self.phi,
            batch_states=self.batch_states if use_custom_batch_states else pfrl_batch_states
        )

        self.T = 0
        self.device = torch.device(f"cuda:{gpu}" if gpu > -1 else "cpu")

    @staticmethod
    def batch_states(states, device, phi):
        # assert isinstance(states, list), type(states)
        # assert isinstance(states[0], atari_wrappers.LazyFrames), type(states[0])
        features = np.array([phi(s) for s in states])
        return torch.as_tensor(features).to(device)

    @staticmethod
    def phi(x):
        """ Observation pre-processing for convolutional layers. """
        # return np.asarray(x, dtype=np.float32) / 255.
        return x

    def act(self, state):
        """ Action selection method at the current state. """
        return torch.tensor(self.agent.act(state))

    def step(self, state, action, reward, next_state, done, reset):
        """ Learning update based on a given transition from the environment. """
        self._overwrite_pfrl_state(state, action)
        self.agent.observe(next_state, reward, done, reset)
    
    def observe(self, next_state, reward, done, reset, info):
        return self.agent.observe(next_state, reward, done, reset, info)

    def _overwrite_pfrl_state(self, state, action):
        """ Hack the pfrl state so that we can call act() consecutively during an episode before calling step(). """
        self.agent.batch_last_obs = [state]
        self.agent.batch_last_action = [action]

    @torch.no_grad()
    def value_function(self, states):
        batch_states = self.agent.batch_states(states, self.device, self.phi)
        action_values = self.agent.model(batch_states).q_values
        return action_values.max(dim=1).values

    def experience_replay(self, trajectory):
        """ Add trajectory to the replay buffer and perform agent learning updates. """

        for transition in trajectory:
            self.step(*transition)

    def gc_experience_replay(self, trajectory, goal, goal_position):
        """ Add trajectory to the replay buffer and perform agent learning updates. """

        def is_close(pos1, pos2, tol):
            return abs(pos1[0] - pos2[0]) <= tol and abs(pos1[1] - pos2[1]) <= tol

        def rf(pos, goal_pos):
            pos = np.array([pos['player_x'], pos['player_y']]) if isinstance(pos, dict) else pos
            d = is_close(pos, goal_pos, tol=2)
            return float(d), d
        
        for state, action, _, next_state, done, reset, next_pos in trajectory:
            augmented_state = self.get_augmented_state(state, goal)
            augmented_next_state = self.get_augmented_state(next_state, goal)
            reward, reached = rf(next_pos, goal_position)
            relabeled_transition = augmented_state, action, reward, augmented_next_state, reached or done, reset
            self.step(*relabeled_transition)
            if reached: break  # it helps to truncate the trajectory for HER strategy `future`

    def get_augmented_state(self, state, goal):
        assert isinstance(goal, atari_wrappers.LazyFrames), type(goal)
        assert isinstance(state, atari_wrappers.LazyFrames), type(state)
        features = list(state._frames) + [goal._frames[-1]]
        return atari_wrappers.LazyFrames(features, stack_axis=0)

    def rollout(self, env, state, episode, max_reward_so_far):
        """ Single episodic rollout of the agent's policy. """

        def is_close(pos1, pos2, tol):
            return abs(pos1[0] - pos2[0]) <= tol and abs(pos1[1] - pos2[1]) <= tol

        def rf(info_dict):
            p1 = info_dict["player_x"], info_dict["player_y"]
            p2 = 123, 148
            d = is_close(p1, p2, 2)
            return float(d), d

        done = False
        reset = False
        reached = False

        episode_length = 0
        episode_reward = 0.
        episode_trajectory = []

        while not done and not reset and not reached:
            action = self.act(state)
            next_state, reward, done, info  = env.step(action)
            reset = info.get("needs_reset", False)

            reward, reached = rf(info)

            episode_trajectory.append((state,
                                       action,
                                       np.sign(reward), 
                                       next_state, 
                                       done or reached, 
                                       reset))

            self.T += 1
            episode_length += 1
            episode_reward += reward

            state = next_state

        self.experience_replay(episode_trajectory)
        max_reward_so_far = max(episode_reward, max_reward_so_far)
        print(f"Episode: {episode}, T: {self.T}, Reward: {episode_reward}, Max reward: {max_reward_so_far}")        

        return episode_reward, episode_length, max_reward_so_far

    def gc_rollout(self, env, state, goal, goal_pos, rf, episode, max_reward_so_far):
        """ Single episodic rollout of the agent's policy. """

        info = {}
        done = False
        reset = False
        reached = False

        episode_length = 0
        episode_reward = 0.
        episode_positions = []
        episode_trajectory = []

        while not done and not reset and not reached:
            sg = self.get_augmented_state(state, goal)
            action = self.act(sg)
            next_state, reward, done, info  = env.step(action)
            reset = info.get("needs_reset", False)

            reward, reached = rf(info, goal_pos)

            player_pos = info["player_x"], info["player_y"]
            episode_positions.append(player_pos)
            episode_trajectory.append(
                                      (state,
                                       action,
                                       np.sign(reward), 
                                       next_state, 
                                       done or reached, 
                                       reset,
                                       player_pos
                                    )
                                )

            self.T += 1
            episode_length += 1
            episode_reward += reward

            state = next_state

        self.her(episode_trajectory, episode_positions, goal, pursued_goal_position=goal_pos)

        max_reward_so_far = max(episode_reward, max_reward_so_far)
        print(f"G: {goal_pos}, T: {self.T}, Reward: {episode_reward}, Max reward: {max_reward_so_far}")        

        return episode_reward, episode_length, max_reward_so_far, done or reset
    
    def her(self, trajectory, visited_positions, pursued_goal, pursued_goal_position):
        hindsight_goal, hindsight_goal_idx = self.pick_hindsight_goal(trajectory)
        self.gc_experience_replay(trajectory, pursued_goal, pursued_goal_position)
        self.gc_experience_replay(trajectory, hindsight_goal, visited_positions[hindsight_goal_idx])

    def pick_hindsight_goal(self, trajectory, strategy="future"):
        """ Select a hindsight goal from the input trajectory. """
        assert strategy in ("final", "future"), strategy
        
        goal_idx = -1
        
        if strategy == "future":
            start_idx = len(trajectory) // 2
            goal_idx = random.randint(start_idx, len(trajectory) - 1)

        goal_transition = trajectory[goal_idx]
        goal_state = goal_transition[3]
        assert isinstance(goal_state, atari_wrappers.LazyFrames), type(goal_state)
        return goal_state, goal_idx
    
    def get_statistics(self):
        return self.agent.get_statistics()
    def save(self, dirname):
        self.agent.save(dirname)
    
class AbstractRainbow(pfrl.agent.Agent):
    def __init__(self, encoder, agent, action_mask=None, device='cpu'):
        self.agent = agent
        self.encoder = encoder
        self.action_mask = action_mask
        self.gamma = agent.gamma
        self.device = device
        self.agent.device = device
        print(self.gamma)
        
    def act(self, obs, initset=None):
        with torch.no_grad():
            z = self.encoder(torch.from_numpy(obs).to(self.device))
        action = self.agent.act(z)
        return action
    
    def load(self, dirname):
        self.agent.load(dirname)
    
    def observe(self, obs, reward, done, reset, info):
        with torch.no_grad():
            z = self.encoder(torch.from_numpy(obs).to(self.device))
        self.agent.observe(z, reward, done, reset, info)
    
    def save(self, dirname):
        self.agent.save(dirname)
    
    def get_statistics(self):
        return self.agent.get_statistics()
    
    def getattr(self, name):
        return getattr(self.agent, name)