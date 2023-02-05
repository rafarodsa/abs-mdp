import numpy as np
import torch
import torch.nn.functional as F

def _geometric_return(rewards, gamma):
    _gammas = gamma **  torch.arange(rewards.shape[-1])
    return (torch.Tensor(rewards) * _gammas.unsqueeze(0)).sum(-1)

def compute_return(gamma=0.99):
    def _transform(datum):
        # s, a, s', r, executed, duration, init_mask
        datum = list(datum)
        datum[3] = datum[3][:2] + [_geometric_return(datum[3][-1], gamma),]
        return datum
    return _transform

def one_hot_actions(n_actions=4):
    def _transform(datum):
        datum = list(datum)
        datum[1] = F.one_hot(torch.Tensor([datum[1]]).long(), n_actions).squeeze()
        return datum
    return _transform

class PinballDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_file, n_reward_samples=5, transforms=None, dtype=torch.float32):
        self.data, self.rewards = torch.load(path_to_file)
        self.n_reward_samples = n_reward_samples
        self.transforms = transforms
        self.dtype = dtype

    def __getitem__(self, index):
        datum = self._set_dtype(list(self.data[index]))
        rewards = self._get_rewards(datum, n_samples=self.n_reward_samples-1)
        datum[3] = rewards
        for t in self.transforms:
            datum = t(datum)
        datum[3][-1] = datum[3][-1].to(self.dtype)
        return datum

    def _set_dtype(self, datum):
        s, a, s_, r, executed, duration, init_mask = datum
        s = torch.from_numpy(s).type(self.dtype)
        s_ = torch.from_numpy(s_).type(self.dtype)
        init_mask = torch.from_numpy(init_mask).type(self.dtype)
        return [s, a, s_, r, executed, duration, init_mask]

    def _get_rewards(self, datum, n_samples):
        current_obs, action, current_next_obs, current_rewards, _, _, _ = datum
        obs, next_obs, rews = self.rewards[action] # tuple of arrays (obs, next_obs, rewards).
        N = rews.shape[0]
        sample = np.random.choice(N, n_samples, replace=False)
        # append current datum
        obs = np.concatenate([current_obs[np.newaxis, :], obs[sample]], axis=0)
        next_obs = np.concatenate([current_next_obs[None, :], next_obs[sample]], axis=0)
        current_rewards = np.array(current_rewards)[None, :]
        rewards = np.concatenate([current_rewards, rews[sample]], axis=0)
        return [torch.from_numpy(obs).to(self.dtype), torch.from_numpy(next_obs).to(self.dtype), rewards]
	
    def __len__(self):
        return len(self.data)