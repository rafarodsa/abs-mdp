import numpy as np
from collections import deque
import torch
from dataclasses import dataclass
from src.utils import printarr
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
        self.current_trajectory = []
        self.device = device

    def push(self, state, action, reward, next_state, duration, success, done, info):
        """
        Add a transition to the current trajectory.
        """
        self.current_trajectory.append([state, action, reward, next_state, duration, success, done, info])


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
        for trajectory in sampled_trajectories:
            
            # Unpack the trajectory
            states, actions, rewards, next_states, durations, success, done, _ = zip(*trajectory)
            
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

        tensor_trajectories = [torch.stack(t).to(self.device) for t in zip(*tensor_trajectories)]
        return tensor_trajectories

    def _pad_sequence(self, sequence, max_length, dtype=torch.float32):
        """Pad the sequence with zeros to match max_length."""
        pad_size = max_length - len(sequence)
        _sequence = list(map(lambda x: self._to_torch(x, dtype=dtype), sequence))
        padded_sequence = _sequence + [torch.zeros_like(_sequence[0], dtype=dtype)] * pad_size
        return torch.stack(padded_sequence)

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

    def to(self, device):
        self.device = device
