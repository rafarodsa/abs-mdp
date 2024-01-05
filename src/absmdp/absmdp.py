'''
    Implementation of Abstract MDP
    author: Rafael Rodriguez-Sanchez
    date: 23 August 2023

'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from src.utils import printarr
from src.utils.symlog import symlog, symexp

from src.models.factories import build_distribution, build_model

from .buffer import TrajectoryReplayBuffer
from .datasets_traj import PinballDatasetTrajectory_, compute_return, one_hot_actions

from functools import partial
import src.absmdp.configs
from src.absmdp.utils import Every
from tqdm import tqdm

import lightning.fabric as Fabric
from lightning.fabric.loggers import TensorBoardLogger
from lightning.pytorch.loggers import CSVLogger
import lightning as L
from lightning.pytorch.utilities.model_summary import ModelSummary
import matplotlib.pyplot as plt

from omegaconf import OmegaConf as oc


class AbstractMDP(nn.Module, gym.Env):
    SMOOTHING_NOISE_STD = 0.2

    def __init__(self, cfg, fabric=None):
        nn.Module.__init__(self)
        gym.Env.__init__(self)

        model_cfg = oc.masked_copy(cfg, 'model') 
        oc.resolve(model_cfg)
        cfg.model = model_cfg
        oc.update(cfg, 'model', model_cfg.model)

        self.cfg = cfg
        self.fabric = fabric

        self.obs_dim = model_cfg.model.obs_dims
        self.latent_dim = model_cfg.model.latent_dim
        self.n_options = model_cfg.model.n_options

        self.encoder = build_model(model_cfg.model.encoder.features)
        self.transition = build_distribution(model_cfg.model.transition)
        self.initsets = build_model(model_cfg.model.init_class)
        self.tau = build_model(model_cfg.model.tau)
        self.reward_fn = build_model(model_cfg.model.reward)

        self.lr = cfg.optimizer.params.lr
        self.data = TrajectoryReplayBuffer(int(cfg.data.buffer_size))
        self.batch_size = cfg.data.batch_size
        self.hyperparams = cfg.loss
        self.optimizer = None
        self.action_space = gym.spaces.Discrete(self.n_options)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.latent_dim,), dtype=np.float32)
        self.current_z = None
        self.last_initset = None
        self.outdir = None
        self.reward_scale = cfg.reward_scale
        self.model_success = cfg.model_success
        self.sample_transition = cfg.sample_transition
        self.gamma = cfg.gamma
        self._task_reward = None
        if self.cfg.name is not None:
            self.name = self.cfg.name
        else:
            self.name = 'world_model'
        self.init_state_sampler = None
        self.initset_thresh = 0.5
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        super().to(device)
        self.data.to(device)


    def set_no_initset(self):
        self.initset_thresh = -1.

    @torch.no_grad()
    def step(self, action):
        info = {}
        action = F.one_hot(action.long(), self.n_options).to(self.current_z.device).unsqueeze(0)
        z = self.current_z.unsqueeze(0)
        next_z = self._transition(z, action)
        r = self._reward(z, action, next_z)[0]
        if self._task_reward is not None:
            r_g, done = self.task_reward(next_z)
        tau = self._tau(z, action)[0]
        info['tau'] = tau
        info['env_reward'] = r
        info['task_reward'] = r_g
        info['task_done'] = done
        info['initset_s'] = self.last_initset
        info['initset_next_s'] = self.initset(next_z)[0]
        self.last_initset = info['initset_next_s']

        done = done or self.last_initset.sum() == 0

        r = self.reward_scale * r + self.gamma ** (tau-1) * r_g  # add task reward
        self.current_z = next_z.squeeze(0)


        return self.current_z, r, done, info
    
    @torch.no_grad()
    def reset(self, ground_state=None):
        if ground_state is None:
            if self.init_state_sampler is None:
                traj = self.data.sample(1)[0]
                ground_state = traj[0][0][0]
            else:
                # print('sampling')
                ground_state = self.init_state_sampler()
                if isinstance(ground_state, np.ndarray):
                    ground_state = torch.from_numpy(ground_state).to(self.device)
        self.current_z = self.encoder(ground_state)
        self.last_initset = self.initset(self.current_z)

        while self.last_initset.sum() == 0:
            if self.init_state_sampler is None:
                traj = self.data.sample(1)[0]
                ground_state = traj[0][0][0] 
            else:
                # print('sampling')
                ground_state = self.init_state_sampler()
                if isinstance(ground_state, np.ndarray):
                    ground_state = torch.from_numpy(ground_state).to(self.device)
            self.current_z = self.encoder(ground_state)
            self.last_initset = self.initset(self.current_z)

        return self.current_z

    def set_task_reward(self, task_reward):
        self._task_reward = task_reward

    def set_init_state_sampler(self, init_state_sampler):
        self.init_state_sampler = init_state_sampler
    
    def task_reward(self, z):
        return self._task_reward(z)

    def set_outdir(self, outdir):
        self.outdir = outdir

    @torch.no_grad()
    def initset(self, z):
        probs = self.initsets(z).sigmoid()
        return  (probs > self.initset_thresh).float()
   
    
    def _transition(self, z, a):

        if self.model_success:
            success_prob = self.success(z)[..., a]
            success = torch.bernoulli(success_prob)
            next_z_dist = self.transition.distribution(torch.cat([z, a, success], dim=-1))
        else:
            next_z_dist = self.transition.distribution(torch.cat([z, a], dim=-1))
        
        if self.sample_transition:
            next_z = next_z_dist.sample()[0] + z
        else:
            next_z = next_z_dist.mode() + z 

        return next_z
    
    def _reward(self, z, a, next_z):
        r = symexp(self.reward_fn(torch.cat([z, a, next_z], dim=-1)))
        return r
    
    def _tau(self, z, a):
        tau = symexp(self.tau(torch.cat([z, a], dim=-1)))
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
    
    def setup_loggers(self):
        if self.outdir is None:
            print("WARNING: outdir not set. Loggers will not be saved")
            return []

        tensorboard = TensorBoardLogger(root_dir=f'{self.outdir}/tensorboard', name=self.name)
        csvlogger = CSVLogger(save_dir=f'{self.outdir}/csv', name=self.name, flush_logs_every_n_steps=10)
        self.csvlogger = csvlogger
        self.flush = Every(100)
        
        return [csvlogger, tensorboard]

    def setup_fabric(self, cfg_fabric):
        loggers = self.setup_loggers()

        self.fabric = Fabric.Fabric(
                                    accelerator=cfg_fabric.accelerator,
                                    devices=cfg_fabric.devices,
                                    strategy=cfg_fabric.strategy,
                                    loggers = loggers,
                                )

    def setup_trainer(self, cfg=None, log=True):
        if self.fabric is None:
            self.setup_fabric(cfg.fabric)
        self.fabric.launch()
        self.configure_optimizers()
        _ , self.optimizer = self.fabric.setup(self, self.optimizer)

        # print(ModelSummary(self))

    def train_world_model(self, steps=1, timestep=None):
        '''
            Training step.
        '''
        self.train()
        loss = None
        for _ in range(steps):
            batch = self.data.sample(self.batch_size)
            if batch is None:
                print("WARNING: not enough trajectories in the buffer")
                break
            loss, logs = self.training_loss(batch)
            self.optimizer.zero_grad()
            # train 
            logs['step'] = timestep
            if self.fabric is None:
                loss.backward()
                self.optimizer.step()
            else:   
                self.fabric.backward(loss)
                self.fabric.log_dict(logs, step=timestep)

            if self.flush(timestep):
                self.csvlogger.save()


            # pretty print logs
            metrics = list(logs.items())
            metrics = [f'{k}: {v:.3f}' for k, v in metrics if k != 'step']
            metrics = ' | '.join(metrics)
            print(f'[world model training step {timestep}] {metrics}')

        return loss
    
    def observe_batch(self, trajectories_batch):
        self.data_push_batch(trajectories_batch)


    def observe(self, s, action, reward, next_s, done, duration, success, info={}, last=False):
        '''
            Add transition to the replay buffer
        '''
        self.data.push(s, action, reward, next_s, duration, success, done, info)
        if last:
            self.data.end_trajectory()

    def end_episode(self, log_dict=None, step=None):
        if log_dict is not None:
            self.fabric.log_dict(log_dict, step=step)
        self.data.end_trajectory()
        

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)


    def training_loss(self, batch):
        (s, action, reward, next_s, duration, success, done, masks), _ = batch
        action = F.one_hot(action.long(), self.n_options)
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
        
        log_dict = {
            'loss': loss,
            'grounding_loss': grounding_loss.mean(),
            'transition_loss': transition_loss.mean(),
            'tpc_loss': tpc_loss.mean(),
            'initset_loss': initset_loss.mean(),
            'reward_loss': reward_loss.mean(),
            'tau_loss': tau_loss.mean(),
            'rbuffer_len': len(self.data)
        }

        return loss, log_dict


    def initset_loss_from_executed(self, z, action, executed, mask):
        _act_idx = action.argmax(-1, keepdim=True)
        initset_pred = torch.gather(self.initsets(z), -1, _act_idx).squeeze(-1)
        n_pos = executed.sum(-1, keepdim=True)
        n_neg = executed.shape[1] - n_pos
        if torch.all(n_neg==0):
            pos_weight = torch.tensor(1.)
        else:
            pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)

        initset_loss = (F.binary_cross_entropy_with_logits(initset_pred, executed, pos_weight=pos_weight, reduction='none') * mask).sum(-1) / mask.sum(-1)
        return initset_loss

    def success_prediction_loss(self, z, action, executed, mask):
        _act_idx = action.argmax(-1, keepdim=True)
        initset_pred = torch.gather(self.success_model(z), -1, _act_idx).squeeze(-1)
        pos_weight = torch.tensor(1.)
        initset_loss = (F.binary_cross_entropy_with_logits(initset_pred, executed, pos_weight=pos_weight, reduction='none') * mask).sum(-1) / mask.sum(-1)
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

        def action_to_tensor(datum):
            action = torch.tensor(datum.action)
            return datum.modify(action=action)


        print(f'Loading dataset from {dataset_cfg.data_path}')
        dataset = PinballDatasetTrajectory_(
                                                dataset_cfg.data_path,
                                                transforms=[
                                                                partial(compute_return, gamma=dataset_cfg.gamma), 
                                                                action_to_tensor],
                                                obs_type=dataset_cfg.obs_type,
                                                length=dataset_cfg.length,
                                                num_workers=1,
                                                noise_level=0.
                                            )
        
        for i in tqdm(range(len(dataset))):
            trajectory = dataset[i]
            for j in range(trajectory.length):
                self.data.push(
                                state=trajectory.obs[j], 
                                action=trajectory.action[j], 
                                reward=trajectory.rewards[j, 0], 
                                next_state=trajectory.next_obs[j], 
                                done=trajectory.done[j], 
                                duration=trajectory.duration[j], 
                                success=trajectory.executed[j],
                                info=trajectory.info[j] # TODO add info
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
    def load_from_old_checkpoint(world_model_cfg=None, fabric=None):
        '''
            Load model from checkpoint
        '''
        if fabric is None:
            fabric = L.Fabric()

        ckpt_path = world_model_cfg.ckpt
        dataset_path = world_model_cfg.data_path
        loaded_state = fabric.load(ckpt_path, {})
        cfg = oc.create(loaded_state['hyper_parameters']['cfg'])
        cfg.reward_scale = world_model_cfg.reward_scale
        world_model_cfg.model = cfg.model
        mdp = AbstractMDP(world_model_cfg)
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
            if dataset_path is None:
                mdp.initialize_replay_buffer_from_dataset(cfg.data)
            else:
                cfg.data.data_path = dataset_path
                mdp.initialize_replay_buffer_from_dataset(cfg.data)
        except Exception as e:
            print(f'Could not initialize replay buffer from dataset from {cfg.data.data_path} with exception {e.with_traceback()}')
        return mdp


    def save_checkpoint(self, ckpt_path=None):
        '''
            Save model to checkpoint
        '''
        if self.fabric is None:
            raise ValueError("Fabric not initialized")
        if ckpt_path is None:
            ckpt_path = f'{self.outdir}/checkpoints/{self.name}.ckpt'
        self.fabric.save(ckpt_path, self.get_model_state())
    
    # @staticmethod
    # def load_checkpoint(self, ckpt_path, fabric=None):
    #     if fabric is None:
    #         fabric = L.Fabric()

    #     loaded_ckpt = fabric.load(ckpt_path, {})
    #     cfg = loaded_ckpt['cfg']        

    def freeze(self, modules=[]):
        '''
            Freeze modules
        '''
        if isinstance(modules, str):
            modules = [modules]
        for module in modules:
            getattr(self, module).eval()
            for param in getattr(self, module).parameters():
                param.requires_grad = False


class AbstractMDPGoal(AbstractMDP):
    def __init__(self, cfg, goal_cfg, fabric=None):
        super().__init__(cfg, fabric)
        self.goal_class = build_model(goal_cfg)
        self.warmup_steps = goal_cfg.reward_warmup_steps
        self.timestep = 0


    def setup_trainer(self, cfg=None, log=True):
        if self.fabric is None:
            self.setup_fabric(cfg.fabric)
        self.fabric.launch()
        self.configure_optimizers()
        _ , self.optimizer = self.fabric.setup(super(), self.optimizer)
        _, self.goal_optimizer = self.fabric.setup(self.goal_class, self.goal_optimizer)

    def configure_optimizers(self):
        super().configure_optimizers()
        self.goal_optimizer = torch.optim.Adam(self.goal_class.parameters(), lr=3e-4)

    def task_reward(self, z):
        if self.timestep < self.warmup_steps:
            return 0., False
        goal_pred = (self.goal_class(z).squeeze().sigmoid() > 0.5).float()
        return goal_pred, goal_pred > 0
    
    def train_world_model(self, steps=1, timestep=None):
        self.timestep = timestep
        self.goal_optimizer.zero_grad()
        loss = super().train_world_model(steps, timestep)
        # if self.timestep > self.warmup_steps:
        self.goal_optimizer.step()
        return loss

    def goal_training_loss(self, batch):
        (s, action, reward, next_s, duration, success, done, masks), info = batch
        lengths = masks.sum(-1)
        _mask = masks.bool()
        z_c = self.encoder(s)
        z = z_c + self.SMOOTHING_NOISE_STD * torch.randn_like(z_c)
        _z, goal_reward = self._get_goal_reward_from_traj(z, info, lengths, _mask)

        if _z is not None and goal_reward is not None:
            goal_loss = self.goal_loss(_z, goal_reward)
        else:
            goal_loss = torch.tensor(0.)
        return goal_loss

    def training_loss(self, batch):
        (s, action, reward, next_s, duration, success, done, masks), info = batch
        action = F.one_hot(action.long(), self.n_options)
        # encode
        z_c = self.encoder(s)
        next_z_c = self.encoder(next_s)
        z = z_c + self.SMOOTHING_NOISE_STD * torch.randn_like(z_c)
        next_z = next_z_c + self.SMOOTHING_NOISE_STD * torch.randn_like(next_z_c)

        # batch_size, length = action.shape[0], action.shape[1]
    
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

        # _z, goal_reward = self._get_goal_reward_from_traj(z, info, lengths, _mask)
        s, goal_reward = self._get_task_reward_examples()
        with torch.no_grad():
            _z = self.encoder(s)
        goal_loss = self.goal_loss(_z, goal_reward) if _z is not None and goal_reward is not None else torch.tensor(0.)

        if self.warmup_steps <= self.timestep <= self.warmup_steps+100:
            samples, target = self.data.sample_task_reward(2000)
            plt.figure()
            plt.subplot(1,2,1)
            plt.scatter(samples[:, 0], samples[:, 1], c=target, s=1)
            plt.subplot(1,2,2)
            with torch.no_grad():
                _prediction, _ = self.task_reward(self.encoder(samples))
            printarr(samples, _prediction)
            plt.scatter(samples[:, 0], samples[:, 1], c=_prediction, s=1)
            plt.savefig('predition_task_reward.png')
            plt.close()

        loss =  self.hyperparams.grounding_const * grounding_loss.mean() + \
                self.hyperparams.transition_const * transition_loss.mean() + \
                self.hyperparams.tpc_const * tpc_loss.mean() + \
                self.hyperparams.initset_const * initset_loss.mean() + \
                self.hyperparams.reward_const * reward_loss.mean() + \
                self.hyperparams.tau_const * tau_loss.mean() + \
                goal_loss
        
        log_dict = {
            'loss': loss,
            'grounding_loss': grounding_loss.mean(),
            'transition_loss': transition_loss.mean(),
            'tpc_loss': tpc_loss.mean(),
            'initset_loss': initset_loss.mean(),
            'reward_loss': reward_loss.mean(),
            'tau_loss': tau_loss.mean(),
            'goal_class_loss': goal_loss,
        }

        return loss, log_dict

    def goal_loss(self, z, target):

        goal_pred = self.goal_class(z).squeeze(1)
        # print(goal_pred, target)
        # n_pos = target.sum(-1, keepdim=True)
        # n_neg = target.shape[0] - n_pos
        # if torch.any(n_neg==0):
        #     pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)
        # else:
        #     pos_weight = torch.tensor(1.)
        # pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)
        try:
            # loss = F.binary_cross_entropy_with_logits(goal_pred, target, pos_weight=pos_weight)
            loss = F.binary_cross_entropy_with_logits(goal_pred, target)
        except Exception as e:
            printarr(z, target, goal_pred)
        # accuracy
        goal_pred = (goal_pred.sigmoid() > 0.5).float() 
        tpr = (goal_pred * target).sum() / target.sum()
        tnr = ((1-goal_pred) * (1-target)).sum() / (1-target).sum()
        print(f'Goal TPR: {tpr}, TNR: {tnr}')
        return loss

    def get_model_state(self):
        state =  super().get_model_state()
        state['goal_class'] = self.goal_class
        return state
    
    def preload_task_reward_samples(self, positive_samples, negative_samples):
        for i in range(len(positive_samples)):
            self.data.push_task_reward_sample(positive_samples[i], pos=True)
        for i in range(len(negative_samples)):
            self.data.push_task_reward_sample(negative_samples[i], pos=False)

    def _get_goal_reward_from_traj(self, z, info, lengths, mask):
        goal_reached = []
        idx = []

        for i, (traj_info, _len) in enumerate(zip(info, list(lengths))):
            if 'goal_reached' not in traj_info[0]:
                continue
            idx.append(i)
            _goal_reached = []
            for j in range(int(_len.item())):
                _info = traj_info[j]
                _goal_reached.append(_info['goal_reached'])
            goal_reached.extend(_goal_reached)
        
        mask = mask[idx]
        z = z[idx]
        if len(idx) == 0 or len(goal_reached) == 0:
            return None, None
        device = f'cuda:{z.get_device()}' if z.get_device() >= 0 else 'cpu'
        goal_reward = torch.Tensor(goal_reached).to(device)
        return z[mask], goal_reward

    def _get_task_reward_examples(self):
        return  self.data.sample_task_reward(1024)

    @staticmethod
    def load_from_old_checkpoint(world_model_cfg=None, fabric=None):
        '''
            Load model from checkpoint
        '''
        if fabric is None:
            fabric = L.Fabric()

        ckpt_path = world_model_cfg.ckpt
        dataset_path = world_model_cfg.data_path
        loaded_state = fabric.load(ckpt_path, {})
        cfg = oc.create(loaded_state['hyper_parameters']['cfg'])
        cfg.reward_scale = world_model_cfg.reward_scale
        goal_class_cfg = world_model_cfg.model.goal_class
        world_model_cfg.model = oc.create(loaded_state['hyper_parameters']['cfg']['model'])
        mdp = AbstractMDPGoal(world_model_cfg, fabric=None, goal_cfg=goal_class_cfg)
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
            if dataset_path is None:
                mdp.initialize_replay_buffer_from_dataset(cfg.data)
            else:
                cfg.data.data_path = dataset_path
                mdp.initialize_replay_buffer_from_dataset(cfg.data)
        except Exception as e:
            print(f'Could not initialize replay buffer from dataset from {cfg.data.data_path} with exception {e.with_traceback()}')
        return mdp


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

