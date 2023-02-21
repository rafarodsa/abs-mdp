'''
    Pinball Environment Datasets.
'''
<<<<<<< HEAD
import os
=======

>>>>>>> debugging-loss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
<<<<<<< HEAD
from functools import partial, reduce
from typing import NamedTuple, List, Dict, Optional

=======
from functools import partial
from typing import NamedTuple, List
import os
>>>>>>> debugging-loss

class InitiationSet(NamedTuple):
    obs: torch.Tensor
    action: torch.Tensor
    executed: torch.Tensor
    duration: torch.Tensor
    def modify(self, **kwargs):
        params = self._asdict()
        params.update(kwargs)
        return InitiationSet(**params)

class Transition(NamedTuple):
<<<<<<< HEAD
    obs: np.ndarray
    action: int
    next_obs: np.ndarray
    rewards: List[float]
    done: bool
    executed: bool
    duration: int
    initsets: np.ndarray
    info: Dict
    p0: Optional[np.float32]
=======
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    rewards: List[torch.Tensor]
    executed: torch.Tensor
    duration: torch.Tensor
    init_mask: torch.Tensor
>>>>>>> debugging-loss

    def modify(self, **kwargs):
        params = self._asdict()
        params.update(kwargs)
        return Transition(**params)


def _geometric_return(rewards, gamma):
    _gammas = gamma **  torch.arange(rewards.shape[-1])
    return (torch.Tensor(rewards) * _gammas.unsqueeze(0)).sum(-1)

def compute_return(datum, gamma=0.99):
    rewards = datum.rewards
    obs, next_obs, rews = rewards
    rews = _geometric_return(rews, gamma)
    return datum.modify(rewards=[obs, next_obs, rews])

def one_hot_actions(n_actions=4, datum=None):
    action = datum.action
    action = F.one_hot(torch.Tensor([action]).long(), n_actions).squeeze()
    return datum.modify(action=action)

def linear_projection(datum, linear_projection):
    w = linear_projection
    r_obs = torch.tensordot(datum.rewards[0], w, dims=1) # N x state_dim | state_dim x n_features
    r_next_obs = torch.tensordot(datum.rewards[1], w, dims=1)
    obs, next_obs = datum.obs, datum.next_obs
    obs = torch.tensordot(obs, w, dims=1) 
    next_obs = torch.tensordot(next_obs, w, dims=1)
    return datum.modify(obs=obs, next_obs=next_obs, rewards=[r_obs, r_next_obs, datum.rewards[2]])


class PinballDataset_(torch.utils.data.Dataset):
    def __init__(self, path_to_file, n_reward_samples=5, transforms=None, obs_type='full', dtype=torch.float32):
<<<<<<< HEAD
        self.trajectories, self.rewards = torch.load(path_to_file)
        self.data = self.process_trajectories(self.trajectories)
=======
        self.data, self.rewards = torch.load(path_to_file)
        self.data = [Transition(*d) for d in self.data] # transform all to named tuple
        self.data = self._filter_failed_executions(self.data)
>>>>>>> debugging-loss
        self.n_reward_samples = n_reward_samples
        self.transforms = transforms
        self.dtype = dtype
        self.obs_type = obs_type

    def __getitem__(self, index):
        datum = self.data[index]
<<<<<<< HEAD
        datum = self._set_dtype(datum)
        obs, next_obs = datum.obs, datum.next_obs
=======
        obs, action, next_obs, rew, executed, duration, init_mask = self._set_dtype(datum)
>>>>>>> debugging-loss
        rew = self._get_rewards(datum, n_samples=self.n_reward_samples-1)
        rew[-1] = rew[-1].to(self.dtype) # rewards to dtype
        if self.obs_type == 'pixels':
            obs = self._process_pixel_obs(obs)
            next_obs = self._process_pixel_obs(next_obs)
<<<<<<< HEAD
        datum = datum.modify(obs=obs, next_obs=next_obs, rewards=rew)
=======
        datum = Transition(obs, action, next_obs, rew, executed, duration, init_mask)
>>>>>>> debugging-loss
        for t in self.transforms:
            datum = t(datum)
        return datum

    def __len__(self):
        return len(self.data)

    def _set_dtype(self, datum):
        s, s_, initsets, executed = datum.obs, datum.next_obs, datum.initsets, datum.executed
        s = torch.from_numpy(s).to(self.dtype)
        s_ = torch.from_numpy(s_).to(self.dtype)
        initsets = torch.from_numpy(initsets).to(self.dtype)
        executed = torch.tensor(executed).to(self.dtype)
        return datum.modify(obs=s, next_obs=s_, initsets=initsets, executed=executed, info={})

    def _get_rewards(self, datum, n_samples):
        current_obs, action, current_next_obs, current_rewards = datum.obs, datum.action, datum.next_obs, datum.rewards
        obs, next_obs, rews = self.rewards[action] # tuple of arrays (obs, next_obs, rewards).
        N = rews.shape[0]
        sample = np.random.choice(N, n_samples, replace=False)
        # append current datum
        obs = np.concatenate([current_obs[np.newaxis, :], obs[sample]], axis=0)
        next_obs = np.concatenate([current_next_obs[None, :], next_obs[sample]], axis=0)
        
        # transpose if images
        if self.obs_type == 'pixels':
            obs = obs.transpose(0, 3, 1, 2)
            next_obs = next_obs.transpose(0, 3, 1, 2)

        current_rewards = np.array(current_rewards)[None, :]
        rewards = np.concatenate([current_rewards, rews[sample]], axis=0)
<<<<<<< HEAD
        return [torch.from_numpy(obs).to(self.dtype), torch.from_numpy(next_obs).to(self.dtype), torch.from_numpy(rewards).to(self.dtype)]
=======
        return [torch.from_numpy(obs).to(self.dtype), torch.from_numpy(next_obs).to(self.dtype), torch.from_numpy(rewards)]
>>>>>>> debugging-loss

    def _process_pixel_obs(self, obs):
        return obs.type(self.dtype).permute(2, 0, 1)
    
    def _filter_failed_executions(self, data):
        return [datum for datum in data if datum.executed == 1]
    
<<<<<<< HEAD
    def process_trajectories(self, trajectories):
        data = reduce(lambda x, acc: x + acc, trajectories, [])
        data = self._filter_failed_executions(data)
        return data
=======
>>>>>>> debugging-loss

class PinballDataset(pl.LightningDataModule):
    def __init__(self, cfg, linear_transform=True):
        super().__init__()
        self.path_to_file = cfg.data_path
        self.n_reward_samples = cfg.n_reward_samples
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.obs_type = cfg.obs_type
        self.train_split, self.val_split, self.test_split = cfg.train_split, cfg.val_split, cfg.test_split
        self.shuffle = cfg.shuffle

        self.cfg = cfg
        
        self._load_linear_transform()
        self.transforms = [
                            partial(compute_return, gamma=cfg.gamma), 
                            partial(one_hot_actions, cfg.n_options),
<<<<<<< HEAD
                            partial(linear_projection, linear_projection=self.linear_transform)
                        ]
=======
                        ]
        # if linear_projection:
        #     self.transforms.append(partial(linear_projection, linear_projection=self.linear_transform))
>>>>>>> debugging-loss

    def setup(self, stage=None):
        self.dataset = PinballDataset_(self.path_to_file, self.n_reward_samples, self.transforms, obs_type=self.obs_type)
        self.train, self.val, self.test = torch.utils.data.random_split(self.dataset, [self.train_split, self.val_split, self.test_split])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def _load_linear_transform(self):
        path, file = os.path.split(self.path_to_file)
        if os.path.isfile(f'{path}/lintransform.pt'):
<<<<<<< HEAD
=======
            print(f'Linear transform loaded from {path}/lintransform.pt')
>>>>>>> debugging-loss
            self.linear_transform = torch.load(f'{path}/lintransform.pt')
            assert self.linear_transform.shape == torch.Size([self.cfg.state_dim, self.cfg.n_out_feats])
        else:
            self.linear_transform = torch.rand(self.cfg.state_dim, self.cfg.n_out_feats)
            torch.save(self.linear_transform, f'{path}/lintransform.pt')
            print(f'Linear transform matrix saved at {path}/lintransform.pt')

    def state_dict(self):
        state = {'weights': self.linear_transform}
        return state
    
    def load_state_dict(self, state_dict):
        self.linear_transform = state_dict['weights']
    
class InitiationDS(Dataset):
    def __init__(self, dataset_path, n_actions, dtype=torch.float32):
        self.n_actions = n_actions
        self.data, _ = torch.load(dataset_path)
        self.data = [InitiationSet(data[0], data[1], data[4], data[5]) for data in self.data]
        self.data = self._group_by_state()
        self.dtype = dtype

    def __getitem__(self, index):
        d = self.data[index]
        obs = torch.from_numpy(d.obs).to(self.dtype)
        action = F.one_hot(d.action.long(), self.n_actions).squeeze().to(self.dtype)
        executed = d.executed.to(self.dtype)
        return d.modify(obs=obs, action=action, executed=executed)

    def __len__(self):
        return len(self.data)
    
    def _group_by_state(self):
        groups = len(self.data) // self.n_actions
        data_by_state = []
        for i in range(groups):
            
            action = torch.from_numpy(np.array([d.action for d in self.data[i*self.n_actions:(i+1)*self.n_actions]]))
            executed = torch.from_numpy(np.array([d.executed for d in self.data[i*self.n_actions:(i+1)*self.n_actions]]))
            duration = torch.from_numpy(np.array([d.duration for d in self.data[i*self.n_actions:(i+1)*self.n_actions]]))
            data_by_state.append(InitiationSet(self.data[i*self.n_actions].obs, action, executed, duration))

            ## TEST
            # assert torch.allclose(action.long(), torch.arange(self.n_actions).long()), f"action: {action}"
            # obs = torch.from_numpy(np.array([d.obs for d in self.data[i*self.n_actions:(i+1)*self.n_actions]])).to(self.dtype)
            # _obs = torch.from_numpy(self.data[i*self.n_actions].obs).to(self.dtype).repeat(self.n_actions, 1)
            # assert torch.allclose(obs, _obs), f"obs: {obs}, _obs: {_obs}"

        return data_by_state


class InitiationDataset(pl.LightningDataModule):
    def __init__(self, dataset_path, cfg, dtype=torch.float32):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.train_split, self.val_split, self.test_split = cfg.train_split, cfg.val_split, cfg.test_split
        self.shuffle = cfg.shuffle
        self.n_actions = cfg.n_options
        self.dtype = dtype
    
    def setup(self, stage=None):
        # split dataset
        self.dataset = InitiationDS(self.dataset_path, self.n_actions, self.dtype)
        self.train, self.val, self.test = random_split(self.dataset, [self.train_split, self.val_split, self.test_split])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
<<<<<<< HEAD
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
=======
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
>>>>>>> debugging-loss
