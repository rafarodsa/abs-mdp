import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import partial

def _geometric_return(rewards, gamma):
    _gammas = gamma **  torch.arange(rewards.shape[-1])
    return (torch.Tensor(rewards) * _gammas.unsqueeze(0)).sum(-1)

def compute_return(gamma=0.99, datum=None):
    # s, a, s', r, executed, duration, init_mask
    datum = list(datum)
    datum[3] = datum[3][:2] + [_geometric_return(datum[3][-1], gamma),]
    return datum

def one_hot_actions(n_actions=4, datum=None):
    datum = list(datum)
    datum[1] = F.one_hot(torch.Tensor([datum[1]]).long(), n_actions).squeeze()
    return datum

class PinballDataset_(torch.utils.data.Dataset):
    def __init__(self, path_to_file, n_reward_samples=5, transforms=None, obs_type='full', dtype=torch.float32):
        self.data, self.rewards = torch.load(path_to_file)
        self.n_reward_samples = n_reward_samples
        self.transforms = transforms
        self.dtype = dtype
        self.obs_type = obs_type

    def __getitem__(self, index):
        datum = self._set_dtype(list(self.data[index]))
        rewards = self._get_rewards(datum, n_samples=self.n_reward_samples-1)
        datum[3] = rewards
        for t in self.transforms:
            datum = t(datum)
        datum[3][-1] = datum[3][-1].to(self.dtype)
        if self.obs_type == 'pixels':
            datum[0] = self._process_pixel_obs(datum[0])
            datum[2] = self._process_pixel_obs(datum[2])
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

    def _process_pixel_obs(self, obs):
        return obs.type(self.dtype).permute(2, 0, 1)

class PinballDataset(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.path_to_file = cfg.data_path
        self.n_reward_samples = cfg.n_reward_samples
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.obs_type = cfg.obs_type
        self.train_split, self.val_split, self.test_split = cfg.train_split, cfg.val_split, cfg.test_split
        self.shuffe = cfg.shuffle
        self.transforms = [partial(compute_return, cfg.gamma), partial(one_hot_actions, cfg.n_options)]

    def setup(self, stage=None):
        self.dataset = PinballDataset_(self.path_to_file, self.n_reward_samples, self.transforms, obs_type=self.obs_type)
        self.train, self.val, self.test = torch.utils.data.random_split(self.dataset, [self.train_split, self.val_split, self.test_split])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)