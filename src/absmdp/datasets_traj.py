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
    
class Trajectory(NamedTuple):
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    rewards: torch.Tensor
    done: torch.Tensor
    executed: torch.Tensor
    duration: torch.Tensor
    initsets: torch.Tensor
    # info: List[Dict]
    p0: torch.Tensor
    length: torch.Tensor


def _geometric_return(rewards, gamma):
    _gammas = gamma **  torch.arange(rewards.shape[-1])
    return (torch.Tensor(rewards) * _gammas.unsqueeze(0)).sum(-1)

def compute_return(datum, gamma=0.99):
    rewards = datum.rewards
    rews = _geometric_return(rewards, gamma)
    return datum.modify(rewards=rews)

def one_hot_actions(n_actions=4, datum=None):
    action = datum.action
    action = F.one_hot(torch.Tensor([action]).long(), n_actions).squeeze()
    return datum.modify(action=action)

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
    return datum.modify(obs=obs, next_obs=next_obs, rewards=[r_obs, r_next_obs, datum.rewards[2]])

def cubed(datum):
    
    r_obs = datum.rewards[0] ** 3
    r_next_obs = datum.rewards[1] ** 3

    obs, next_obs = datum.obs ** 3, datum.next_obs ** 3
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

class PinballDatasetTrajectory_(torch.utils.data.Dataset):
    IMG_FORMAT = 'tj_{}_obs_{}.png'

    def __init__(self, path_to_file, transforms=None, obs_type='full', length=32, dtype=torch.float32, noise_level=0., num_workers=1):
        self.zfile_name = path_to_file
        self.transforms = transforms
        self.dtype = dtype
        self.obs_type = obs_type
        self.noise_level = noise_level
        self.num_workers = num_workers
        self.length = length
        self.load()

    def load(self):
        try:
            with zipfile.ZipFile(self.zfile_name, 'r') as zfile:
                print(f'Loading trajectories... from {self.zfile_name}')
                self.trajectories = torch.load(zfile.open('transitions.pt'))
                self.trajectories = list(filter(lambda t: len(t) > 0, self.trajectories))
                nl = list(filter(lambda n: '.png' in n, zfile.namelist()))
            if self.obs_type == 'pixels':
                print(f'Loading with {self.num_workers} workers')
                batch_size = len(nl) // self.num_workers if self.num_workers > 1 else len(nl)
                images_loaded = Parallel(n_jobs=self.num_workers)(delayed(load_images)(self.zfile_name, imgs) for imgs in split(nl, batch_size))
                images_loaded = reduce(lambda x,acc: x+acc, images_loaded, [])
                print(f'{len(images_loaded)} images loaded! ')
                self.images_loaded = dict(images_loaded)     
        except:
            raise ValueError(f'File {self.zfile_name} could not be opened')

    def __transform_transition(self, datum):
        datum = self._set_dtype(datum)
        rew = torch.Tensor(datum.rewards).to(self.dtype)


        if self.obs_type != 'pixels':
            obs, next_obs = datum.obs, datum.next_obs  
        else:
            obs, next_obs = self.get_image_obs(datum.obs), self.get_image_obs(datum.next_obs)
            obs, next_obs = torch.from_numpy(obs).to(self.dtype), torch.from_numpy(next_obs).to(self.dtype)
        datum = datum.modify(obs=obs, next_obs=next_obs, rewards=rew)
        for t in self.transforms:
            datum = t(datum)
        return datum
    

    def __getitem__(self, index):
        trajectory = self.trajectories[index]

        trajectory = [self.__transform_transition(datum) for datum in trajectory]
        length = len(trajectory)
        padding = self.length - length
        assert padding >= 0 and length > 0, f'Padding {padding}, Traj Length {length}, Buffer Length {self.length}'

        s, a, next_s, rewards, done, executed, duration, initsets, _, p0 = zip(*trajectory)
        s = torch.stack(list(s) + [torch.zeros_like(s[0]) for _ in range(padding)])
        a = torch.stack(list(a) + [torch.zeros_like(a[0]) for _ in range(padding)])
        next_s = torch.stack(list(next_s) + [torch.zeros_like(next_s[0]) for _ in range(padding)])
        rewards = torch.cat([torch.stack(rewards), torch.zeros(padding, 1)], dim=0)
        done = torch.cat([torch.Tensor(done), torch.zeros(padding)], dim=0)
        executed = torch.cat([torch.Tensor(executed), torch.zeros(padding)], dim=0)
        duration = torch.cat([torch.Tensor(duration), torch.zeros(padding)], dim=0)
        initsets = torch.stack(list(initsets) + [torch.zeros_like(initsets[0]) for _ in range(padding)])
        p0 = torch.cat([torch.Tensor(p0), torch.zeros(padding)], dim=0)
        # info = list(info) + [dict() for _ in range(padding)]

        return Trajectory(s, a, next_s, rewards, done, executed, duration, initsets, p0, length)

    def __len__(self):
        return len(self.trajectories)
    
    def get_image_obs(self, obs):
        img_path = self.IMG_FORMAT.format(obs.traj, obs.timestep)
        if img_path in self.images_loaded:
            return self.images_loaded[img_path]
        else: # load
            try:
                print('Warning: reloading image')
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
        executed = torch.tensor(executed).to(self.dtype) if not isinstance(executed, torch.Tensor) else torch.tensor(executed.item()).to(self.dtype)
        return datum.modify(obs=s, next_obs=s_, initsets=initsets, executed=executed, duration=duration)

    def _process_pixel_obs(self, obs):
        r,g,b = obs[:, :, 0], obs[:, :, 1], obs[:, :, 2]
        noise = np.random.randn(*r.shape) * 1/255
        return np.clip((0.2989 * r + 0.5870 * g + 0.1140 * b)[np.newaxis] / 255 + noise, 0, 1)

class PinballDatasetTrajectory(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.path_to_file = cfg.data_path
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.obs_type = cfg.obs_type
        self.train_split, self.val_split, self.test_split = cfg.train_split, cfg.val_split, cfg.test_split
        self.shuffle = cfg.shuffle
        self.length = cfg.length
        self.cfg = cfg
        self.transforms = [
                            partial(compute_return, gamma=cfg.gamma), 
                            partial(one_hot_actions, cfg.n_options),
                        ]
        if cfg.linear_transform  and self.obs_type != 'pixels':
            self._load_linear_transform()
            self.transforms.append(partial(linear_projection, linear_projection=self.linear_transform, noise_level=cfg.noise_level))
        
        if cfg.non_linear_transform and self.obs_type != 'pixels':
            self.transforms.append(cosine_transform)
            


    def setup(self, stage=None):
        print(f'Loading dataset at {self.path_to_file}')
        self.dataset = PinballDatasetTrajectory_(self.path_to_file, self.transforms, obs_type=self.obs_type, noise_level=self.cfg.noise_level, num_workers=self.num_workers, length=self.length)
        self.train, self.val, self.test = torch.utils.data.random_split(self.dataset, [self.train_split, self.val_split, self.test_split])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers-1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers-1)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers-1)
    
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
    
