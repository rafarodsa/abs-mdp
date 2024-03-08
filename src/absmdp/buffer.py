import numpy as np
from collections import deque
import torch
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from src.utils import printarr

from jax.tree_util import tree_map
import copy
import pathlib 
import uuid
import os
import datetime
import io

import functools

@dataclass
class Transition:
    state: any
    action: any
    reward: any
    next_state: any
    duration: float
    success: bool
    done: bool
    info: dict

class TrajectoryReplayBuffer:
    def __init__(self, capacity, device='cpu'):
        """
        Initialize the replay buffer.
        
        Args:
        - capacity (int): Maximum number of trajectories to store.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.task_reward_pos = deque(maxlen=capacity)
        self.task_reward_neg = deque(maxlen=capacity)
        self.current_trajectory = []
        self.current_trajectories = []
        self.device = device
        self.data_device = device  # Keep track of which device the data is on

    def push_batch(self, states, actions, rewards, next_states, durations, successes, dones, infos):
        if len(self.current_trajectories) == 0:
            self._n_trajs = len(states)
            self.current_trajectories = [[] for _ in range(self._n_trajs)]
        assert len(states) == self._n_trajs

        for i, traj  in enumerate(self.current_trajectories):
            _copy = copy.deepcopy
            state = tree_map(self._to_numpy, _copy(states[i]))
            action = self._to_numpy(_copy(actions[i]))
            reward = self._to_numpy(_copy(rewards[i]))
            next_state = tree_map(self._to_numpy, _copy(next_states[i]))
            duration = self._to_numpy(_copy(durations[i]))
            success = self._to_numpy(_copy(successes[i]), dtype=torch.float32)
            done = self._to_numpy(_copy(dones[i]), dtype=torch.float32)
            traj.append([state, action, reward, next_state, duration, success, done, infos[i]])
            if 'goal_reached' in infos[i]:
                self.push_task_reward_sample(next_state, infos[i]['goal_reached'] > 0)


    def push(self, state, action, reward, next_state, duration, success, done, info):
        """
        Add a transition to the current trajectory.
        """
        self.current_trajectory.append([state, action, reward, next_state, duration, success, done, info])
        if 'goal_reached' in info:
            self.push_task_reward_sample(next_state, info['goal_reached'] > 0)

    def push(self, state, action, reward, next_state, duration, success, done, info):
        """
        Add a transition to the current trajectory.
        """
        _copy = copy.deepcopy
        state = tree_map(self._to_torch, _copy(state))
        action = self._to_torch(_copy(action))
        reward = self._to_torch(_copy(reward))
        next_state = tree_map(self._to_torch, _copy(next_state))
        duration = self._to_torch(_copy(duration))
        success = self._to_torch(_copy(success), dtype=torch.float32)
        done = self._to_torch(_copy(done), dtype=torch.float32)

        if 'goal_reached' in info:
            self.push_task_reward_sample(next_state, info['goal_reached'] > 0)
        
        self.current_trajectory.append([state, action, reward, next_state, duration, success, done, info])

    def push_task_reward_sample(self, state, pos=True):
        if not isinstance(state, torch.Tensor):
            state = self._to_torch(state)
        if pos:
            self.task_reward_pos.append(state)
        else:
            self.task_reward_neg.append(state)


    def end_trajectory(self):
        """
        End the current trajectory and add it to the buffer.
        """
        if len(self.current_trajectory) > 0:
            self.buffer.append(self.current_trajectory)
            self.current_trajectory = []


    def end_trajectory_batch(self, dones):
        assert len(dones) == len(self.current_trajectories)
        lens = [len(traj) for traj in self.current_trajectories]
        for i, traj in enumerate(self.current_trajectories):
            if dones[i]:
                self.buffer.append(traj)
                self.current_trajectories[i] = []

        new_lens = [len(traj) for traj in self.current_trajectories]
        for l, n_l, done in zip(lens, new_lens, dones):
            if done:
                assert n_l == 0
            else:
                assert l == n_l

    def _sample_eps(self, num_trajectories):
        indices = np.random.choice(len(self), num_trajectories, replace=False)
        sampled_trajectories = [self.load_episode(idx) for idx in indices]
        return sampled_trajectories
    
    def _last_eps(self, num_trajectories):
        indices = list(range(len(self) - num_trajectories, len(self)))
        sampled_trajectories = [self.load_episode(idx) for idx in indices]
        return sampled_trajectories

    def sample(self, num_trajectories):
        """
        Sample trajectories from the buffer and return PyTorch tensors, padding shorter trajectories.
        
        Args:
        - num_trajectories (int): Number of trajectories to sample.
        
        Returns:
        - List of trajectories. Each trajectory is a tuple containing:
            - List of transitions represented as PyTorch tensors.
            - Mask indicating the non-padded elements.
        """

        if len(self) < num_trajectories:
            return None  # Not enough trajectories in the buffer.
        
        sampled_trajectories = self._sample_eps(num_trajectories)

        # Find the maximum trajectory length for padding
        max_length = max([len(trajectory) for trajectory in sampled_trajectories])

        tensor_trajectories = []
        infos = []
        for trajectory in sampled_trajectories:
            
            # Unpack the trajectory
            states, actions, rewards, next_states, durations, success, done, info = zip(*trajectory)
            
            # Create mask for this trajectory
            mask = torch.Tensor([1] * len(trajectory) + [0] * (max_length - len(trajectory)))
            # Pad each component of the trajectory
            states = self._pad_sequence(states, max_length)
            actions = self._pad_sequence(actions, max_length)
            rewards = self._pad_sequence(rewards, max_length)
            next_states = self._pad_sequence(next_states, max_length)
            durations = self._pad_sequence(durations, max_length)
            success = self._pad_sequence(success, max_length, dtype=torch.float32)
            done = self._pad_sequence(done, max_length, dtype=torch.float32)
            # printarr(states, actions, rewards, next_states, durations, success, done, mask)
            tensor_trajectory = (states, actions, rewards, next_states, durations, success, done, mask)
            tensor_trajectories.append(tensor_trajectory)
            infos.append(info)
        
        tensor_trajectories = [tree_map(lambda *tensors: torch.stack(tensors).to(self.device), *t) for t in zip(*tensor_trajectories)]
        return tensor_trajectories, infos
    

    def get_last_eps(self, num_trajectories):
        """
        Sample trajectories from the buffer and return PyTorch tensors, padding shorter trajectories.
        
        Args:
        - num_trajectories (int): Number of trajectories to sample.
        
        Returns:
        - List of trajectories. Each trajectory is a tuple containing:
            - List of transitions represented as PyTorch tensors.
            - Mask indicating the non-padded elements.
        """

        if len(self) < num_trajectories:
            return None  # Not enough trajectories in the buffer.
        
        sampled_trajectories = self._last_eps(num_trajectories)

        # Find the maximum trajectory length for padding
        max_length = max([len(trajectory) for trajectory in sampled_trajectories])

        tensor_trajectories = []
        infos = []
        for trajectory in sampled_trajectories:
            
            # Unpack the trajectory
            states, actions, rewards, next_states, durations, success, done, info = zip(*trajectory)
            
            # Create mask for this trajectory
            mask = torch.Tensor([1] * len(trajectory) + [0] * (max_length - len(trajectory)))
            # Pad each component of the trajectory
            states = self._pad_sequence(states, max_length)
            actions = self._pad_sequence(actions, max_length)
            rewards = self._pad_sequence(rewards, max_length)
            next_states = self._pad_sequence(next_states, max_length)
            durations = self._pad_sequence(durations, max_length)
            success = self._pad_sequence(success, max_length, dtype=torch.float32)
            done = self._pad_sequence(done, max_length, dtype=torch.float32)
            # printarr(states, actions, rewards, next_states, durations, success, done, mask)
            tensor_trajectory = (states, actions, rewards, next_states, durations, success, done, mask)
            tensor_trajectories.append(tensor_trajectory)
            infos.append(info)
        
        tensor_trajectories = [tree_map(lambda *tensors: torch.stack(tensors).to(self.device), *t) for t in zip(*tensor_trajectories)]
        return tensor_trajectories, infos
    

    def load_episode(self, episode_idx):
        assert episode_idx < len(self.buffer)
        return self.buffer[episode_idx]
    

    def sample_task_reward(self, batch_size):
        pos_samples = []
        neg_samples = []
        if len(self.task_reward_pos) > 0:
            pos_samples_idx = np.random.choice(len(self.task_reward_pos), batch_size // 2)
            pos_samples = [self.task_reward_pos[pos_samples_idx[i]]for i in range(len(pos_samples_idx))]
        if len(self.task_reward_neg) > 0:
            neg_samples_idx = np.random.choice(len(self.task_reward_neg), batch_size // 2)
            neg_samples = [self.task_reward_neg[neg_samples_idx[i]] for i in range(len(neg_samples_idx))]
        samples = pos_samples + neg_samples
        labels = torch.zeros(len(samples))
        labels[:len(pos_samples)] = 1.
        return tree_map(lambda *tensors: torch.stack(tensors).to(self.device), *samples), labels.to(self.device)


    def _pad_sequence(self, sequence, max_length, dtype=torch.float32):
        """Pad the sequence with zeros to match max_length."""
        
        pad_size = max_length - len(sequence)
        _sequence = tree_map(lambda x: self._to_torch(x, dtype=dtype), sequence)
        padded_sequence = _sequence + tree_map(lambda x: torch.zeros_like(x, dtype=dtype), _sequence[0:1]) * pad_size
        return tree_map(lambda *tensors: torch.stack(tensors), *padded_sequence)

    def __len__(self):
        """
        Return the number of trajectories in the buffer.
        """
        return len(self.buffer)

    def _to_torch(self, x, dtype=torch.float32):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).type(dtype)
        elif isinstance(x, torch.Tensor):
            # Only move to device if it's not on the desired device
            # if x.device != self.device:
                # return x.type(dtype)
            return x
        elif isinstance(x, (int, float, np.int64, np.float32, np.float64, bool, np.bool_)):
            return torch.tensor(x, dtype=dtype)
        elif isinstance(x, dict):
            return x
        else:
            raise ValueError(f"Unsupported type {type(x)}")
    
    def _to_numpy(self, x, dtype=torch.float32):
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            # Only move to device if it's not on the desired device
            # if x.device != self.device:
                # return x.type(dtype)
            return x.cpu().numpy()
        elif isinstance(x, (int, float, np.int64, np.float32, np.float64, bool, np.bool_)):
            return x
        elif isinstance(x, dict):
            return x
        else:
            raise ValueError(f"Unsupported type {type(x)}")

    def to(self, device):
        if device != self.data_device:
            # Transfer all trajectories to the new device
            for trajectory in self.buffer:
                for i in range(len(trajectory)):
                    trajectory[i] = tuple(self._to_torch(item) for item in trajectory[i])
            self.data_device = device
        self.device = device


class TrajectoryReplayBufferStored(TrajectoryReplayBuffer, Dataset):
    EPISODE_ELEMS = ['state', 'action', 'reward', 'next_state', 'duration', 'success', 'done', 'info']

    def __init__(self, capacity, device='cpu', save_path='.', length=64):
        super().__init__(capacity=capacity, device=device)
        self.save_path = pathlib.Path(save_path).expanduser()
        self.episodes = []
        self.prepare_storage()
        self.loaded = set()
        self.length = length
        self.buffer = {}

    def prepare_storage(self):
        if os.path.exists(self.save_path / 'replay'):
            self.episodes = sorted((self.save_path / 'replay').glob('*.npz'))
            print(f'Replay buffer exists at {self.save_path}. Loading {len(self.episodes)} episodes')
        else:
            os.makedirs(self.save_path / 'replay')
            print(f'Replay buffer at {self.save_path / "replay"}')

    def _sample_eps(self, num_trajectories, min_length=8):
        sampled = set()
        sample_eps = []
        while len(sampled) < num_trajectories:  
            idx = int(np.random.choice(len(self), 1, replace=False))
            if idx not in sampled:
                ep = self.load_episode(idx)
                if len(ep) >= min_length:
                    sample_eps.append(ep)
                    sampled.add(idx)
        return sample_eps

    def load_goal_examples(self, n_samples=128):
        n_samples = min(n_samples, len(self))
        indices = np.random.choice(n_samples, n_samples, replace=False)

        for i in list(indices):
            ep = self.load_episode(i)
            last_state, last_info = ep[-1][3], ep[-1][-1]
            assert 'goal_reached' in last_info
            self.push_task_reward_sample(last_state, last_info['goal_reached'] > 0)

    def save_episode(self, episode):

        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        identifier = str(uuid.uuid4().hex)
        eplen = len(episode['action'])
        filename = self.save_path / 'replay' / f'{timestamp}-{identifier}-{eplen}.npz'

        with io.BytesIO() as f1:
            np.savez_compressed(f1, **{k: v for k,v in episode.items()})
            f1.seek(0)
            with filename.open('wb') as f2:
                f2.write(f1.read())
        return filename

    def load_episode(self, episode_idx, allow_pickle=True):
        filename = self.episodes[episode_idx]
        # if str(filename) in self.buffer:
            # return self.buffer[str(filename)]
        try:
            with filename.open('rb') as f:
                episode = np.load(f, allow_pickle=allow_pickle)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            raise ValueError(f'Could not load episode {str(filename)}: {e}')

        ep = list(zip(*[episode[k] for k in self.EPISODE_ELEMS]))

        if not str(filename) in self.loaded:
            self.loaded.add(str(filename))
            state, info = ep[-1][3], ep[-1][-1]
            self.push_task_reward_sample(state, info['goal_reached'] > 0)
        # self.buffer[str(filename)] = ep
        return ep
    
    def end_trajectory(self):
        """
        End the current trajectory and add it to the buffer.
        """
        if len(self.current_trajectory) > 0:
            episode = {k:v for k, v in zip(self.EPISODE_ELEMS, self.current_trajectory)}
            episode_filename = self.save_episode(episode)
            self.episodes.append(episode_filename)
            self.buffer.append(self.current_trajectory)
            self.current_trajectory = []

    def end_trajectory_batch(self, dones):
        assert len(dones) == len(self.current_trajectories)
        lens = [len(traj) for traj in self.current_trajectories]
        for i, traj in enumerate(self.current_trajectories):
            if dones[i]:
                episode = {k:v for k, v in zip(self.EPISODE_ELEMS, zip(*traj))}
                episode_filename = self.save_episode(episode)
                self.episodes.append(episode_filename)
                # self.buffer.append(traj)
                self.current_trajectories[i] = []

        new_lens = [len(traj) for traj in self.current_trajectories]
        for l, n_l, done in zip(lens, new_lens, dones):
            if done:
                assert n_l == 0
            else:
                assert l == n_l
    
    def __len__(self):
        return len(self.episodes)

    def to(self, device):
        self.device = device

    def __getitem__(self, idx):
        ep = self.load_episode(idx)

        # Unpack the trajectory
        states, actions, rewards, next_states, durations, success, done, info = zip(*ep)
        
        # Create mask for this trajectory
        max_length = self.length
        mask = torch.Tensor([1] * len(ep) + [0] * (self.length - len(ep)))
        # Pad each component of the trajectory
        states = self._pad_sequence(states, max_length)
        actions = self._pad_sequence(actions, max_length)
        rewards = self._pad_sequence(rewards, max_length)
        next_states = self._pad_sequence(next_states, max_length)
        durations = self._pad_sequence(durations, max_length)
        success = self._pad_sequence(success, max_length, dtype=torch.float32)
        done = self._pad_sequence(done, max_length, dtype=torch.float32)

        tensor_trajectory = (states, actions, rewards, next_states, durations, success, done, mask)
        info = list(info) + [info[0] for i in range((self.length - len(ep)))]
        return tensor_trajectory, info



