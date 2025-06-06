'''
    Pinball Environment Datasets.
'''
import os, io
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import lightning as pl
from functools import partial, reduce
from typing import NamedTuple, List, Dict, Optional

import zipfile
from PIL import Image
from collections import namedtuple
from typing import NamedTuple

from src.utils.printarr import printarr
from joblib import Parallel, delayed
from itertools import tee, chain
from tqdm import tqdm


class ObservationImgFile(NamedTuple):
    traj: int
    timestep: int

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

# def linear_projection(datum, linear_projection, noise_level=0.01):
#     w = linear_projection
#     w = w/(w ** 2).sum(dim=0, keepdims=True).sqrt()
#     f = lambda o : torch.cat([o[..., :2] * 2 - 1, o[..., 2:]], dim=-1)
#     r_obs = torch.tensordot(datum.rewards[0], w, dims=1) # N x state_dim | state_dim x n_features
#     r_next_obs = torch.tensordot(datum.rewards[1], w, dims=1)
#     r_obs = r_obs + torch.randn_like(r_obs) * noise_level
#     r_next_obs = r_next_obs + torch.randn_like(r_next_obs) * noise_level

#     obs, next_obs = datum.obs, datum.next_obs
#     obs = torch.tensordot(obs, w, dims=1) 
#     next_obs = torch.tensordot(next_obs, w, dims=1)
#     obs = obs + torch.randn_like(obs) * noise_level
#     next_obs = next_obs + torch.randn_like(next_obs) * noise_level
#     # print('linear_transforming', noise_level)
#     return datum.modify(obs=obs, next_obs=next_obs, rewards=[r_obs, r_next_obs, datum.rewards[2]])

def linear_projection(datum, linear_projection, noise_level=0.0):
    w = linear_projection
    w = w/(w ** 2).sum(dim=0, keepdims=True).sqrt()
    f = lambda o : torch.cat([o[..., :2] * 2 - 1, o[..., 2:]], dim=-1)
    r_obs = torch.tensordot(f(datum.rewards[0]), w, dims=1) # N x state_dim | state_dim x n_features
    r_next_obs = torch.tensordot(f(datum.rewards[1]), w, dims=1)
    r_obs = r_obs + torch.randn_like(r_obs) * noise_level
    r_next_obs = r_next_obs + torch.randn_like(r_next_obs) * noise_level

    obs, next_obs = f(datum.obs), f(datum.next_obs)
    obs = torch.tensordot(obs, w, dims=1) 
    next_obs = torch.tensordot(next_obs, w, dims=1)
    obs = obs + torch.randn_like(obs) * noise_level
    next_obs = next_obs + torch.randn_like(next_obs) * noise_level
    # print('linear_transforming', noise_level)
    return datum.modify(obs=obs, next_obs=next_obs, rewards=[r_obs, r_next_obs, datum.rewards[2]])

def cubed(datum):
    
    r_obs = datum.rewards[0] ** 3
    r_next_obs = datum.rewards[1] ** 3

    obs, next_obs = datum.obs ** 3, datum.next_obs ** 3
    # printarr(obs, next_obs)
    return datum.modify(obs=obs, next_obs=next_obs, rewards=[r_obs, r_next_obs, datum.rewards[2]])

def cosine_transform(datum):

    f = lambda o : torch.cat([o[..., :2] * 2 - 1, o[..., 2:]], dim=-1)

    r_obs = torch.cat([torch.cos(f(datum.rewards[0])), torch.sin(f(datum.rewards[0]))], dim=-1)
    r_next_obs = torch.cat([torch.cos(f(datum.rewards[1])), torch.sin(f(datum.rewards[1]))], dim=-1)

    obs = torch.cat([torch.cos(f(datum.obs)), torch.sin(f(datum.obs))], dim=-1)
    next_obs = torch.cat([torch.cos(f(datum.next_obs)), torch.sin(f(datum.next_obs))], dim=-1)
    return datum.modify(obs=obs, next_obs=next_obs, rewards=[r_obs, r_next_obs, datum.rewards[2]])    

def _process_pixel_obs(obs):
    r,g,b = obs[:, :, 0], obs[:, :, 1], obs[:, :, 2]
    noise = np.random.randn(*r.shape) * 1/255
    return np.clip((0.2989 * r + 0.5870 * g + 0.1140 * b)[np.newaxis] / 255 + noise, 0, 1)


def load_image(zfile, name):
    try:
        img = zfile.open(name)
        img_ = Image.open(img)
        img_ = np.array(img_)
        img_ = _process_pixel_obs(img_)
    except:
        raise ValueError(f'Image {name} could not be opened')
    return (name, img_)  

def load_images(zfile_name, nl):
    with zipfile.ZipFile(zfile_name, 'r') as zfile:
        imgs = [load_image(zfile, n) for n in tqdm(nl)]
    return imgs

def split(_list, batch_size):
    n_batches = len(_list) // batch_size
    batches = [_list[i*batch_size:(i+1)*batch_size] for i in range(n_batches-1)]
    batches.append(_list[(n_batches-1)*batch_size:])
    return batches

class PinballDataset_(torch.utils.data.Dataset):
    IMG_FORMAT = 'tj_{}_obs_{}.png'

    def __init__(self, path_to_file, n_reward_samples=5, transforms=None, obs_type='full', dtype=torch.float32, noise_level=0.01, noise_dim=0, ar_coeff=0.9, num_workers=1):
        self.zfile_name = path_to_file
        # self.zfile = zipfile.ZipFile(self.zfile_name) if obs_type == 'pixels' else None
        self.n_reward_samples = n_reward_samples
        self.transforms = transforms
        self.dtype = dtype
        self.obs_type = obs_type
        self.noise_level = noise_level
        self.noise_dim = noise_dim
        self.ar_coeff = ar_coeff
        self.num_workers = num_workers
        self.load()
        self.data = self.process_trajectories(self.trajectories)
        
    def load(self):
        with zipfile.ZipFile(self.zfile_name, 'r') as zfile:
            print(f'Loading trajectories... from {self.zfile_name}')
            self.trajectories = torch.load(zfile.open('transitions.pt'))
            print('Loading rewards...')
            self.rewards = torch.load(zfile.open('rewards.pt'))
            nl = list(filter(lambda n: '.png' in n, zfile.namelist()))
        if self.obs_type == 'pixels':
            print(self.num_workers)
            batch_size = len(nl) // self.num_workers
            images_loaded = Parallel(n_jobs=self.num_workers)(delayed(load_images)(self.zfile_name, imgs) for imgs in split(nl, batch_size))
            images_loaded = reduce(lambda x,acc: x+acc, images_loaded, [])
            print(f'{len(images_loaded)} images loaded! ')
            self.images_loaded = dict(images_loaded)     

    def __getitem__(self, index):
        datum = self.data[index]
        datum = self._set_dtype(datum)
        rew = self._get_rewards(datum, n_samples=self.n_reward_samples-1)
        rew[-1] = rew[-1].to(self.dtype) # rewards to dtype

        if self.obs_type != 'pixels':
            obs, next_obs = datum.obs, datum.next_obs  
        else:
            obs, next_obs = self.get_image_obs(datum.obs), self.get_image_obs(datum.next_obs)
            obs, next_obs = torch.from_numpy(obs).to(self.dtype), torch.from_numpy(next_obs).to(self.dtype)
        datum = datum.modify(obs=obs, next_obs=next_obs, rewards=rew)
        for t in self.transforms:
            datum = t(datum)
        return datum

    def __len__(self):
        return len(self.data)
    
    def get_image_obs(self, obs):
        img_path = self.IMG_FORMAT.format(obs.traj, obs.timestep)
        if img_path in self.images_loaded:
            return self.images_loaded[img_path]
        else: #load
            try:
                img = self.zfile.open(img_path)
                img_ = Image.open(img)
                img_ = np.array(img_)
                img_ = self._process_pixel_obs(img_)
                self.images_loaded[img_path] = img_
            except:
                raise ValueError(f'Image {img_path} could not be opened')
            return img_


    def _set_dtype(self, datum):
        s, s_, initsets, executed, duration = datum.obs, datum.next_obs, datum.initsets, datum.executed, datum.duration
        if self.obs_type != 'pixels':
            s = torch.from_numpy(s).to(self.dtype)
            s_ = torch.from_numpy(s_).to(self.dtype)
        duration = torch.tensor(duration).to(self.dtype)
        initsets = torch.from_numpy(initsets).to(self.dtype)
        executed = torch.tensor(executed).to(self.dtype)
        return datum.modify(obs=s, next_obs=s_, initsets=initsets, executed=executed, duration=duration)

    def _get_rewards(self, datum, n_samples):
        current_obs, action, current_next_obs, current_rewards = datum.obs, datum.action, datum.next_obs, datum.rewards
        obs, next_obs, rews = self.rewards[action] # tuple of arrays (obs, next_obs, rewards).
        if n_samples > 0:
            N = rews.shape[0]
            sample = np.random.choice(N, n_samples, replace=False)
            
            # transpose if images
            if self.obs_type == 'pixels':
                obs_imgs = []
                next_obs_imgs = []
                for i in range(n_samples):
                    obs_imgs.append(self.get_image_obs(ObservationImgFile(*list(obs[sample[i]]))))
                    next_obs_imgs.append(self.get_image_obs(ObservationImgFile(*list(next_obs[sample[i]]))))

                obs_imgs.append(self.get_image_obs(current_obs))
                next_obs_imgs.append(self.get_image_obs(current_next_obs))
                obs, next_obs = np.array(obs_imgs), np.array(next_obs_imgs)
                current_obs, current_next_obs = self.get_image_obs(current_obs), self.get_image_obs(current_next_obs)
                # obs = obs.transpose(0, 3, 1, 2)
                # next_obs = next_obs.transpose(0, 3, 1, 2)
            else:
                # add noise process
                if self.noise_dim > 0:
                    noise, noise_next = self._noise_process(n_samples, self.noise_level, self.noise_dim, self.ar_coeff)
                    obs = np.concatenate([obs[sample], noise], axis=1)
                    next_obs = np.concatenate([next_obs[sample], noise_next], axis=1)
                else:
                    obs = obs[sample]
                    next_obs = next_obs[sample]
                # append current datum
            obs = np.concatenate([current_obs[np.newaxis, :], obs], axis=0)
            next_obs = np.concatenate([current_next_obs[None, :], next_obs], axis=0)
            current_rewards = np.array(current_rewards)[None, :]
            rewards = np.concatenate([current_rewards, rews[sample]], axis=0)
        else:
            obs, next_obs = current_obs.numpy(), current_next_obs.numpy()
            rewards = np.array(current_rewards)
        
        return [torch.from_numpy(obs).to(self.dtype), torch.from_numpy(next_obs).to(self.dtype), torch.from_numpy(rewards).to(self.dtype)]

    def _process_pixel_obs(self, obs):
        r,g,b = obs[:, :, 0], obs[:, :, 1], obs[:, :, 2]
        noise = np.random.randn(*r.shape) * 1/255
        return np.clip((0.2989 * r + 0.5870 * g + 0.1140 * b)[np.newaxis] / 255 + noise, 0, 1)
    
    def _filter_failed_executions(self, data):
        return [datum for datum in data if datum.executed == 1]
    
    def process_trajectories(self, trajectories):
        if self.noise_dim > 0 and self.noise_level > 0:
            trajectories = [self.add_ar_noise(trajectory, noise_level=self.noise_level, noise_dim=self.noise_dim, ar_coeff=self.ar_coeff) for trajectory in trajectories]
        data = reduce(lambda x, acc: x + acc, trajectories, [])
        # data = self._filter_failed_executions(data)
        return data
    
    def add_ar_noise(self, trajectory, noise_level=0.01, noise_dim=2, ar_coeff=0.9):
        '''
            Add extra noise dimensions that evolve according to random AR(1)
        '''
        for i, transition in enumerate(trajectory):
            transition = self._add_ar_noise_transition(transition, noise_level=noise_level, noise_dim=noise_dim, ar_coeff=ar_coeff)
            trajectory[i] = transition
        return trajectory

    def _add_ar_noise_transition(self, transition, noise_level=0.01, noise_dim=2, ar_coeff=0.9):
        noise, next_noise = self._noise_process(1, noise_level, noise_dim, ar_coeff)
        _obs = np.concatenate([transition.obs, noise])
        _next_obs = np.concatenate([transition.next_obs, next_noise])
        return transition.modify(obs=_obs, next_obs=_next_obs)
    
    def _noise_process(self, n_samples, noise_level, noise_dim, ar_coeff):
        noise = np.random.normal(0, noise_level, (n_samples, noise_dim))
        noise_bar = ar_coeff * noise
        next_noise = noise_bar + (1-ar_coeff) * np.random.normal(0, noise_level, (n_samples, noise_dim))
        if n_samples == 1:
            noise, next_noise = noise[0], next_noise[0]
        return noise, next_noise

class PinballDataset(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.path_to_file = cfg.data_path
        self.n_reward_samples = cfg.n_reward_samples
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.obs_type = cfg.obs_type
        self.train_split, self.val_split, self.test_split = cfg.train_split, cfg.val_split, cfg.test_split
        self.shuffle = cfg.shuffle

        self.cfg = cfg
        
        
        self.transforms = [
                            partial(compute_return, gamma=cfg.gamma), 
                            partial(one_hot_actions, cfg.n_options),
                        ]
        if (cfg.linear_transform or cfg.state_dim + cfg.noise_dim != cfg.n_out_feats) and self.obs_type != 'pixels':
            self._load_linear_transform()
            self.transforms.append(partial(linear_projection, linear_projection=self.linear_transform, noise_level=cfg.noise_level))
        
        if cfg.non_linear_transform and self.obs_type != 'pixels':
            # self._load_linear_transform()
            # self.transforms.append(partial(linear_projection, linear_projection=self.linear_transform, noise_level=cfg.noise_level))
            self.transforms.append(cosine_transform)
            


    def setup(self, stage=None):
        self.dataset = PinballDataset_(self.path_to_file, self.n_reward_samples, self.transforms, obs_type=self.obs_type, noise_dim=self.cfg.noise_dim, noise_level=self.cfg.noise_level, ar_coeff=self.cfg.ar_coeff, num_workers=self.num_workers)
        self.train, self.val, self.test = torch.utils.data.random_split(self.dataset, [self.train_split, self.val_split, self.test_split])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def _load_linear_transform(self):
        if os.path.isfile(f'{self.cfg.save_path}/lintransform.pt'):
            print(f'Linear transform loaded from {self.cfg.save_path}/lintransform.pt')
            self.linear_transform = torch.load(f'{self.cfg.save_path}/lintransform.pt')
            assert self.linear_transform.shape == torch.Size([self.cfg.state_dim + self.cfg.noise_dim, self.cfg.n_out_feats])
        else:
            self.linear_transform = torch.rand(self.cfg.state_dim + self.cfg.noise_dim, self.cfg.n_out_feats)
            torch.save(self.linear_transform, f'{self.cfg.save_path}/lintransform.pt')
            print(f'Linear transform matrix saved at {self.cfg.save_path}/lintransform.pt')

    def state_dict(self):
        if self.cfg.linear_transform:
            state = {'weights': self.linear_transform}
            return state
        return None
    
    def load_state_dict(self, state_dict):
        if self.cfg.linear_transform:
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
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
