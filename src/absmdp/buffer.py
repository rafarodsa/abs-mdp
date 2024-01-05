import numpy as np
from collections import deque
import torch
from dataclasses import dataclass
from src.utils import printarr

from jax.tree_util import tree_map
import copy

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
        self.device = device
        self.data_device = device  # Keep track of which device the data is on

    def push(self, state, action, reward, next_state, duration, success, done, info):
        """
        Add a transition to the current trajectory.
        """
        self.current_trajectory.append([state, action, reward, next_state, duration, success, done, info])
        if 'goal_reached' in info:
            self.push_task_reward_sample(state, info['goal_reached'] > 0)

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
            self.push_task_reward_sample(state, info['goal_reached'] > 0)
        
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
        if len(self.buffer) < num_trajectories:
            return None  # Not enough trajectories in the buffer.
        
        indices = np.random.choice(len(self.buffer), num_trajectories, replace=False)
        sampled_trajectories = [self.buffer[idx] for idx in indices]

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

    def _to_torch(self, x, dtype=torch.float32):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).type(dtype).to(self.device)
        elif isinstance(x, torch.Tensor):
            return x.type(dtype).to(self.device)
        elif isinstance(x, (int, float, np.float32, np.float64, bool, np.bool_)):
            return torch.tensor(x, dtype=dtype).to(self.device)
        else:
            raise ValueError(f"Unsupported type {type(x)}")

    def __len__(self):
        """
        Return the number of trajectories in the buffer.
        """
        return len(self.buffer)

    def _to_torch(self, x, dtype=torch.float32):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).type(dtype).to(self.device)
        elif isinstance(x, torch.Tensor):
            # Only move to device if it's not on the desired device
            if x.device != self.device:
                return x.type(dtype).to(self.device)
            return x
        elif isinstance(x, (int, float, np.float32, np.float64, bool, np.bool_)):
            return torch.tensor(x, dtype=dtype).to(self.device)
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
