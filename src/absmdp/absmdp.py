'''
    Implementation of Abstract MDP
    author: Rafael Rodriguez-Sanchez
    date: 23 August 2023

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

from src.utils.symlog import symlog

from src.models.factories import build_distribution, build_model

from buffer import TrajectoryReplayBuffer
from datasets_traj import PinballDatasetTrajectory_, compute_return, one_hot_actions

from functools import partial
import src.absmdp.configs
from tqdm import tqdm

import lightning.fabric as Fabric
import lightning as L

class AbstractMDP(nn.Module, gym.Env):
    SMOOTHING_NOISE_STD = 0.2

    def __init__(self, cfg, fabric=None):
        nn.Module.__init__(self)
        gym.Env.__init__(self)
       
        oc.resolve(cfg)
        self.cfg = cfg
        self.fabric = fabric

        self.data_cfg = cfg.data
        self.obs_dim = cfg.model.obs_dims
        self.latent_dim = cfg.model.latent_dim
        self.n_options = cfg.model.n_options

        self.encoder = build_model(cfg.model.encoder.features)
        self.transition = build_distribution(cfg.model.transition)
        self.initsets = build_model(cfg.model.init_class)
        self.tau = build_model(cfg.model.tau)
        self.reward_fn = build_model(cfg.model.reward)
        self.lr = cfg.optimizer.params.lr
        self.data = TrajectoryReplayBuffer(cfg.data.buffer_size)
        self.hyperparams = cfg.loss
        self.optimizer = None
        self.action_space = gym.spaces.Discrete(self.n_options)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.latent_dim,), dtype=np.float32)
        self.current_z = None

    def step(self, action):
        with torch.no_grad():
            next_z = self.transition(self.current_z, action)
            r = self.reward(self.current_z, action, next_z)
            r_g, done = self.task_reward(next_z)
            tau = self.tau(self.current_z, action)
            self.current_z = next_z

        return self.current_z, r, done, {'r_g': r_g, 'tau': tau}

    def reset(self, ground_state=None):
        if ground_state is None:
            ground_state = self.data.sample(1)[0][0]
        self.current_z = self.encoder(ground_state)
        return self.current_z


    def _transition(self, z, a):
        if self.model_success:
            success_prob = self.success(z)[..., a]
            success = torch.bernoulli(success_prob)
            next_z_dist = self.transition.distribution(torch.cat([z, a, success], dim=-1))
        else:
            next_z_dist = self.transition.distribution(torch.cat([z, a], dim=-1))
        
        if self.sample_transition:
            next_z = next_z_dist.sample() + z
        else:
            next_z = next_z_dist.mode() + z 

        return next_z
    
    def _reward(self, z, a, next_z):
        r = self.reward_fn(torch.cat([z, a, next_z], dim=-1))
        return r
    
    def _tau(self, z, a):
        tau = self.tau(torch.cat([z, a], dim=-1))
        return tau

    def grounding(self, s, z, std=0.1):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s)
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        
        _z = self.encoder(s)
        return torch.exp(-(((z - _z)/std) ** 2).sum(-1))

    def forward(self, s, action):
        '''
            Forward pass
        '''
        _z = self.current_z
        z = self.encoder(s)
        self.current_z = z
        _r = self.step(action)
        self.current_z = z
        return _r
        
    def training_step(self):
        '''
            Training step.
        '''
        batch = self.data.sample(8)
        loss = self.training_loss(batch)
        # train 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def observe(self, s, action, reward, next_s, done, duration, success, info={}, last=False):
        '''
            Add transition to the replay buffer
        '''
        self.data.push(s, action, reward, next_s, done, duration, success)
        if last:
            self.data.end_trajectory()
        

    def configure_optimizers(self):
        self.optimizer =  torch.optim.Adam(self.parameters(), lr=self.lr)


    def training_loss(self, batch):
        s, action, reward, next_s, duration, success, done, masks = batch
        # encode
        z_c = self.encoder(s)
        next_z_c = self.encoder(next_s)
        z = z_c + self.SMOOTHING_NOISE_STD * torch.randn_like(z_c)
        next_z = next_z_c + self.SMOOTHING_NOISE_STD * torch.randn_like(next_z_c)

        batch_size, length = s.shape[0], s.shape[1]
    
        # transition
        next_z_dist = self.transition.distribution(torch.cat([z, action], dim=-1))

        
        lengths = masks.sum(-1)
        _mask = masks.bool()

        grounding_loss = self.grounding_loss(next_z[_mask])
        reward_loss = self.reward_loss(reward[_mask], z_c[_mask], action[_mask], next_z_c[_mask])
        tau_loss = self.duration_loss(duration[_mask], z_c[_mask], action[_mask])

        # transition losses
        next_z_dist = self.transition.distribution(torch.cat([z, action], dim=-1))
        transition_loss = self.consistency_loss(z, next_z, next_z_dist, mask=_mask)


        tpc_loss = self.tpc_loss(z, next_z, next_z_dist, min_length=int(lengths.min().item()))

        initset_loss = self.initset_loss_from_executed(z, action, success, _mask)
        loss =  self.hyperparams.grounding_const * grounding_loss.mean() + \
                self.hyperparams.transition_const * transition_loss.mean() + \
                self.hyperparams.tpc_const * tpc_loss.mean() + \
                self.hyperparams.initset_const * initset_loss.mean() + \
                self.hyperparams.reward_const * reward_loss.mean() + \
                self.hyperparams.tau_const * tau_loss.mean()
        
        return loss


    def initset_loss_from_executed(self, z, action, executed, mask):
        _act_idx = action.argmax(-1, keepdim=True)
        initset_pred = torch.gather(self.initsets(z), -1, _act_idx).squeeze(-1)
        n_pos = executed.sum(-1, keepdim=True)
        n_neg = executed.shape[1] - n_pos
        if torch.all(n_neg==0):
            pos_weight = torch.tensor(1.)
        else:
            pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)

        initset_loss = (F.binary_cross_entropy_with_logits(initset_pred, executed, pos_weight=pos_weight, reduction='none') * mask).sum(-1)
        return initset_loss

    def grounding_loss(self, next_z, std=0.1):
        '''
            critic f(s', z') = exp(||\phi(s')-z'||^2/std^2)
        '''
        b = next_z.shape[:-1]
        b_size = np.prod(b)
        # print(b_size)
        next_z = next_z.reshape(int(b_size), -1)

        _norm = ((next_z[:, None, :] - next_z[None, :, :]) / std) ** 2 
        _norm = -_norm.sum(-1) # b_size x b_size
        _loss =  torch.diag(_norm) - (torch.logsumexp(_norm, dim=-1) - np.log(b_size))
        return -_loss.reshape(*b)
    
    def tpc_loss(self, z, next_z, next_z_dist, min_length):
        '''
            -MI(z'; z, a)
        '''
        b, z_dim = next_z.shape[:-1], next_z.shape[-1]
        b_size = np.prod(b)
        n_traj, length = b[0], b[1]
        _next_z = torch.repeat_interleave(next_z, n_traj, dim=0)
        _z = z.repeat(n_traj, 1, 1)
        _log_t = next_z_dist.log_prob(_next_z-_z, batched=True).reshape(n_traj, n_traj, length)[..., :min_length]
        _diag = torch.diagonal(_log_t).T 
        _loss = _diag - (torch.logsumexp(_log_t, dim=1) - np.log(n_traj)) # n_traj x length

        return -_loss.sum(-1)
    
    def reward_loss(self, r_target, z, a, next_z):
        '''
            MSE(R, R_pred)
        '''
        r = symlog(r_target)
        r_pred = self.reward_fn(torch.cat([z, a, next_z], dim=-1).detach()).squeeze()
        loss = F.mse_loss(r.reshape(-1), r_pred.reshape(-1), reduction='none')
        return loss

    def duration_loss(self, tau_target, z, a):
        tau = symlog(tau_target)
        t = self.tau(torch.cat([z, a], dim=-1).detach()).squeeze()
        loss = F.mse_loss(tau.reshape(-1), t.reshape(-1), reduction='none')

        return loss
    
    def consistency_loss(self, z, next_z, next_z_dist, mask, executed=None):
        '''
            -log T(z'|z, a) 
        '''
        pos_weight = torch.tensor(1.)
        if executed is not None:
            n_pos = executed.sum(-1, keepdim=True)
            n_neg = executed.shape[1] - n_pos
            if torch.any(n_neg==0):
                pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)

        return -(next_z_dist.log_prob(next_z-z) * mask * pos_weight).sum(-1)


    def initialize_replay_buffer_from_dataset(self, dataset_cfg):
        '''
            Legacy function.
            Initialize the replay buffer with trajectories from an offline dataset
        '''

        dataset = PinballDatasetTrajectory_(
                                                dataset_cfg.data_path,
                                                transforms=[partial(compute_return, gamma=dataset_cfg.gamma), partial(one_hot_actions, self.n_options),],
                                                obs_type=dataset_cfg.obs_type,
                                                length=dataset_cfg.length,
                                                num_workers=dataset_cfg.num_workers,
                                                noise_level=0.
                                            )
        
        for i in tqdm(range(len(dataset))):
            trajectory = dataset[i]
            for j in range(trajectory.length):
                self.data.push(
                                state=trajectory.obs[j], 
                                action=trajectory.action[j], 
                                reward=trajectory.rewards[j], 
                                next_state=trajectory.next_obs[j], 
                                done=trajectory.done[j], 
                                duration=trajectory.duration[j], 
                                success=trajectory.executed[j],
                                info={} # TODO add info
                            )
            self.data.end_trajectory()


    def get_model_state(self):
        '''
            Get model state
        '''
        state = {
            "cfg": self.cfg,
            "encoder": self.encoder,
            "transition": self.transition,
            "initsets": self.initsets,
            "tau": self.tau,
            "reward_fn": self.reward_fn,
            "optimizer": self.optimizer,
            "data": self.data,
        }
        return state


    @staticmethod
    def load_from_old_checkpoint(ckpt_path, fabric=None):
        '''
            Load model from checkpoint
        '''
        loaded_state = fabric.load(ckpt_path, {})
        cfg = oc.create(loaded_state['hyper_parameters']['cfg'])
        cfg.data.buffer_size = int(1e6)
        mdp = AbstractMDP(cfg)
        weights = loaded_state['state_dict']
        state_dicts = {}
        for k, v in weights.items():
            module_name = k.split('.')[0]
            if module_name not in state_dicts:
                state_dicts[module_name] = {}
            state_dicts[module_name]['.'.join(k.split('.')[1:])] = v
        
        # load state dicts
        modules_to_load = ['encoder', 'transition', 'initsets', 'tau', 'reward_fn']

        for module_name in modules_to_load:
            if module_name in state_dicts:
                getattr(mdp, module_name).load_state_dict(state_dicts[module_name])
            else:
                raise ValueError(f"Module {module_name} not found in checkpoint")    
        
        # try to initialize replay buffer
        try:
            mdp.initialize_replay_buffer_from_dataset(cfg.data)
        except Exception as e:
            print(f'Could not initialize replay buffer from dataset from {cfg.data.data_path}')
        return mdp


    def save_checkpoint(self, ckpt_path):
        '''
            Save model to checkpoint
        '''
        Fabric.save(ckpt_path, self.get_model_state())


    def freeze(self, modules=[]):
        '''
            Freeze modules
        '''
        if not isinstance(modules, list):
            modules = [modules]
        for module in modules:
            getattr(self, module).eval()
            for param in getattr(self, module).parameters():
                param.requires_grad = False


if __name__ == '__main__':
    ## TEST
    import argparse
    from omegaconf import OmegaConf as oc
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='experiments/pb_obstacles/fullstate/config/tpc_cfg_rssm.yaml')
    parser.add_argument('--test-training', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)

    args = parser.parse_args()

    cfg = oc.load(args.cfg)
    cfg.data.buffer_size = int(1e6)

    if args.ckpt is not None:
        fabric = L.Fabric(accelerator='cpu')
        fabric.launch()
        mdp = AbstractMDP.load_from_old_checkpoint(args.ckpt, fabric=fabric)

    if args.test_training:
        mdp = AbstractMDP(cfg)
        mdp.initialize_replay_buffer_from_dataset(cfg.data)
        mdp.configure_optimizers()
        
        for i in range(10):
            L = mdp.training_step()
            print(L)

