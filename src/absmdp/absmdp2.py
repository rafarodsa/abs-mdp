import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from src.utils import printarr
from src.utils.symlog import symlog, symexp

from src.models.factories import build_distribution, build_model

from .buffer import TrajectoryReplayBufferStored

from functools import partial
from src.absmdp.utils import Every

import lightning.fabric as Fabric
from lightning.fabric.loggers import TensorBoardLogger
from lightning.pytorch.loggers import CSVLogger
import lightning as L
from lightning.pytorch.utilities.model_summary import ModelSummary

from omegaconf import OmegaConf as oc
from jax import tree_map


def get_config(cfg):
    model_cfg = oc.masked_copy(cfg, 'model') 
    oc.resolve(model_cfg)
    cfg.model = model_cfg
    oc.update(cfg, 'model', model_cfg.model)
    return cfg

class AMDP(gym.Env, nn.Module):
    SMOOTHING_NOISE_STD = 0.05
    def __init__(self, cfg, name='world_model'):
        nn.Module.__init__(self)
        gym.Env.__init__(self)
        self.cfg = get_config(cfg)
        
        model_cfg = self.cfg.model

        ### World Model building

        self.encoder = build_model(model_cfg.model.encoder.features)
        self.transition = build_distribution(model_cfg.model.transition)
        self.initsets = build_model(model_cfg.model.init_class)
        self.tau = build_model(model_cfg.model.tau)
        self.gamma_model = build_model(model_cfg.model.tau)
        self.reward_fn = build_model(model_cfg.model.reward)
        self.termination = build_model(model_cfg.termination)
        self.goal_class = build_model(model_cfg.goal_class)
        self.hyperparams = cfg.loss

        self._models = {
            'encoder': self.encoder,
            'transition': self.transition,
            'initsets': self.initsets,
            'tau': self.tau,
            'gamma': self.gamma_model,
            'reward': self.reward_fn,
            'termination': self.termination,
            'goal_class': self.goal_class
        }

        ######## 
        self.n_options = model_cfg.n_options
        self.action_space = gym.spaces.Discrete(model_cfg.n_options)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.latent_dim,), dtype=np.float32)

        self.n_gradient_steps = 0
        self.lr = cfg.optimizer.params.lr
        self.batch_size = cfg.data.batch_size
        self.sample_transition = cfg.sample_transition
        self.residual_transition = True
        self.data = None
        self.outdir = None
        self.reward_scale = cfg.reward_scale
        self.gamma = cfg.gamma
        self._n_feats = model_cfg.latent_dim
        self.recurrent=False
        self.warmup_steps = model_cfg.goal_class.reward_warmup_steps
        self.fabric = None
        self.device = 'cpu'
        self.optimizer = None
        
        
        ### Simulation state
        self.imagination_state = None
        self.name = name
        self.timestep = 0
        


    ############### Gym Interface / World Model rollout ################

    def preprocess(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(np.array(x)).to(self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, dict):
            # complex state
            return tree_map(self.preprocess, x)
        else:
            raise ValueError(f'Unknown type {type(x)}')
        
    def postprocess(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        elif isinstance(x, np.ndarray):
            return np.array(x)
        elif isinstance(x, dict):
            return tree_map(self.postprocess, x)
        else:
            raise ValueError(f'Unknown type {type(x)}')
        
    def preprocess_action(self, action):
        if isinstance(action, (float, int, np.int64, np.intc)):
            return F.one_hot(torch.tensor(action).long().to(self.device), self.n_options)
        elif isinstance(action, np.ndarray):
            return F.one_hot(torch.from_numpy(action).long().to(self.device), self.n_options)
        else:
            raise ValueError(f'Action must be an int or numpy array')

    def get_initial_state(self):
        ep, _ = self.data.sample(1)
        state = ep[0]
        return tree_map(lambda x: x[0][0], state, is_leaf=lambda s: isinstance(s, (np.ndarray, torch.Tensor)))


    @property
    def n_feats(self):
        return self.cfg.model.latent_dim
    
    def train(self, mode=True):
        for m in self._models.values():
            m.train(mode=mode)

    def eval(self):
        self.train(mode=False)

    @torch.no_grad()
    def step(self, action):
        self.eval()
        
        z = self.simulation_state['z']
        action = self.preprocess_action(action)
        next_z = self.imagine(z, action)
        rew = self._reward(z, action, next_z) # low level reward
        r_g = self.task_reward(next_z)
        duration = self._tau(z, action)
        termination = self._termination(next_z)
        initset = self.initset(z)
        reward = self.reward_scale * rew + r_g
        done = termination or r_g > 0
        info = {
            'tau': duration.cpu().item(),
            'env_reward': rew.cpu().item(),
            'task_reward': r_g.cpu().item(),
            'task_done': (r_g > 0).cpu().item(),
            'initset_s': self.postprocess(self.simulation_state['initset']),
            'initset_next_s': self.preprocess(initset),
        } 
        # TODO done = done or self.last_initset.sum() == 0 
        self.simulation_state = dict(z=next_z, initset=initset)
        return self.postprocess(next_z), reward.cpu().item(), done, info

    def imagine(self, z, action):
        next_z_dist = self.transition.distribution(torch.cat([z, action], dim=-1))
        if self.sample_transition:
            next_z = next_z_dist.sample()[0]
        else:
            next_z = next_z_dist.mode()
        if self.residual_transition:
            next_z = next_z + z
        return next_z
    
    def _reward(self, z, a, next_z):
        _inp = torch.cat([z, a, next_z], dim=-1)
        r = symexp(self.reward_fn(_inp))
        return r
    
    def _tau(self, z, a):
        _inp = torch.cat([z, a], dim=-1)
        log_tau = torch.sigmoid(self.tau(_inp))*100
        return log_tau
    
    def _termination(self, z):
        termination_prob = torch.softmax(self.termination(z), dim=-1)
        # done = torch.bernoulli(termination_prob)
        done = termination_prob.argmax(-1).float()
        return done
    
    def initset(self, z):
        return (torch.sigmoid(self.initsets(z)) > 0.5).float()
    
    def task_reward(self, z):
        goal_prob = F.softmax(self.goal_class(z), dim=-1)
        return goal_prob.argmax(-1).float()
    
    @torch.no_grad()
    def reset(self, ground_state=None):
        done = False
        while not done:
            if ground_state is None:
                ground_state = self.get_initial_state()
            else:
                done = True
            ground_state = self.preprocess(ground_state)
            z = self.encoder(ground_state) 
            initset = self.initset(z)
            done = initset.sum().item() > 0
        self.simulation_state = dict(z=z, initset=initset)
        return self.postprocess(z)


    ####################### TRAINING METHODS ###########################

    # def observe_batch(self, trajectories_batch):
    #     self.data_push_batch(trajectories_batch)

    def observe(self, s, action, reward, next_s, done, duration, success, info={}, last=False):
        '''
            Add transition to the replay buffer
        '''
        self.data.push_batch(s, action, reward, next_s, duration, success, done, info)
        self.data.end_trajectory_batch(last)

    def end_episode(self, log_dict=None, step=None):
        if log_dict is not None:
            self.fabric.log_dict(log_dict, step=step)
        self.data.end_trajectory()

    def train_world_model(self, steps=1, timestep=None):
        '''
            Training step.
        '''
        assert not self.fabric is None, f'Initialize Fabric to train World Model'
        assert not self.data is None, f'No data buffer initialized'
        self.train()
        for _ in range(steps):
            batch = self.data.sample(self.batch_size)
            if batch is None:
                print("WARNING: not enough trajectories in the buffer")
                break
            self.optimizer.zero_grad()
            loss, logs = self.training_loss(batch)
            
            # train 
            logs['step'] = timestep
            if self.fabric is None:
                loss.backward()
            else:   
                self.fabric.backward(loss)
                self.fabric.log_dict(logs, step=timestep)

            if self.flush(timestep):
                self.csvlogger.save()
            
            self.optimizer.step()
            # pretty print logs
            metrics = list(logs.items())
            metrics = [f'{k}: {v:.3f}' for k, v in metrics if k != 'step']
            metrics = ' | '.join(metrics)
        print(f'[world model training step {timestep} / gradient steps {self.n_gradient_steps}] {metrics}')
        self.n_gradient_steps += steps
        self.timestep = timestep
        return loss
    
    def _get_task_reward_examples(self):
        return  self.data.sample_task_reward(1024)

    def training_loss(self, batch):
        (state, action, reward, next_state, duration, success, done, masks), info = batch
        action = F.one_hot(action.long(), self.n_options)
        # encode
        z_c = self.encoder(state)
        next_z_c = self.encoder(next_state)
        z = z_c + self.SMOOTHING_NOISE_STD * torch.randn_like(z_c)
        next_z = next_z_c + self.SMOOTHING_NOISE_STD * torch.randn_like(next_z_c)
        # transition
        next_z_dist = self.transition.distribution(torch.cat([z, action], dim=-1))
        lengths = masks.sum(-1)
        _mask = masks.bool()

        grounding_loss = self.grounding_loss(next_z_c[_mask])
        reward_loss = self.reward_loss(reward[_mask], z_c[_mask], action[_mask], next_z_c[_mask])
        tau_loss = self.duration_loss(duration[_mask], z_c[_mask], action[_mask])
        termination_loss = self.termination_loss(done[_mask], next_z[_mask])
        transition_loss = self.transition_loss(z, next_z, next_z_dist, mask=_mask)
        tpc_loss = self.predictibility_loss(z, next_z, next_z_dist, min_length=int(lengths.min().item()), mask=_mask)
        initset_loss = self.initset_loss_from_executed(z, action, success, _mask)
        gamma_loss = self.gamma_loss(z_c[_mask], action[_mask], duration[_mask])

        # feats, goal_reward = self._get_goal_reward_from_traj(next_z_c, info, lengths, _mask)
        states, goal_reward = self._get_task_reward_examples()
        feats = self.encoder(states).detach()
        feats = feats + torch.randn_like(feats) * self.SMOOTHING_NOISE_STD 
        goal_loss, goal_log_dict = self.goal_loss(feats, goal_reward)

        loss =  self.hyperparams.grounding_const * grounding_loss.mean() + \
                self.hyperparams.transition_const * transition_loss.mean() + \
                self.hyperparams.tpc_const * tpc_loss.mean() + \
                self.hyperparams.initset_const * initset_loss.mean() + \
                self.hyperparams.reward_const * reward_loss.mean() + \
                self.hyperparams.tau_const * tau_loss.mean() + \
                self.hyperparams.termination_const * termination_loss.mean() + \
                self.hyperparams.goal_class_const * goal_loss.mean() + \
                self.hyperparams.tau_const * gamma_loss.mean()
        
        log_dict = {
            'loss': loss,
            'grounding_loss': grounding_loss.mean(),
            'transition_loss': transition_loss.mean(),
            'predictibility_loss': tpc_loss.mean(),
            'initset_loss': initset_loss.mean(),
            'reward_loss': reward_loss.mean(),
            'tau_loss': tau_loss.mean(),
            'termination_loss': termination_loss.mean(),
            'gamma_loss': gamma_loss.mean(),
            'rbuffer_len': len(self.data),
            'norm2': (z_c ** 2).sum(-1).mean(),
            'goal_loss': goal_loss.mean(),
            **goal_log_dict
        }

        return loss, log_dict
    
    def termination_loss(self, done, next_z):
        n_pos = done.sum()
        n_neg = (1-done).sum()
        if torch.all(n_neg==0):
            pos_weight = torch.tensor(1.)
        else:
            pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)
        done_pred = torch.softmax(self.termination(next_z).squeeze(-1), dim=-1)
        # loss = nn.functional.binary_cross_entropy_with_logits(done_pred, done, pos_weight=pos_weight)
        # loss = nn.functional.binary_cross_entropy_with_logits(done_pred, done)
        loss = nn.functional.cross_entropy(done_pred, done.long())
        return loss
    
    def preload_task_reward_samples(self, positive_samples, negative_samples):
        for i in range(len(positive_samples)):
            self.data.push_task_reward_sample(positive_samples[i], pos=True)
        for i in range(len(negative_samples)):
            self.data.push_task_reward_sample(negative_samples[i], pos=False)
    
    def gamma_loss(self, z, a, duration):
        gamma = self.cfg.gamma ** duration
        gamma_pred = self.gamma_model(torch.cat([z, a], dim=-1).detach()).squeeze(-1)
        return F.mse_loss(gamma_pred, gamma)
    
    def grounding_loss(self, next_z, std=0.1, batch_size=256):
        '''
            critic f(s', z') = exp(||\phi(s')-z'||^2/std^2)
        '''
        b = next_z.shape[:-1]
        b_size = np.prod(b)
        next_z = next_z.reshape(int(b_size), -1)[:batch_size]

        next_z_1 = next_z + torch.randn_like(next_z) * self.SMOOTHING_NOISE_STD
        next_z_2 = next_z + torch.randn_like(next_z) * self.SMOOTHING_NOISE_STD

        _norm = ((next_z_1[:, None, :] - next_z_2[None, :, :]) / std) ** 2 
        _norm = -_norm.sum(-1) # b_size x b_size
        _loss =  torch.diag(_norm) - (torch.logsumexp(_norm, dim=-1) - np.log(b_size))
        return -_loss.mean()


    def predictibility_loss(self, z, next_z, next_z_dist, min_length, mask):
        '''
            -MI(z'; z, a)
        '''
        b = next_z.shape[:-1]
        n_traj, length = b[0], b[1]
        _next_z = torch.repeat_interleave(next_z, n_traj, dim=0)
        _z = z.repeat(n_traj, 1, 1)
        if not self.residual_transition:
            _log_t = next_z_dist.log_prob(_next_z, batched=True).reshape(n_traj, n_traj, length)[..., :min_length]
        else:
            _log_t = next_z_dist.log_prob(_next_z-_z, batched=True).reshape(n_traj, n_traj, length)[..., :min_length]
        _diag = torch.diagonal(_log_t).T 
        _loss = _diag - (torch.logsumexp(_log_t, dim=1) - np.log(n_traj)) # n_traj x length

        return -_loss[..., 1:].mean(-1)

    
    def reward_loss(self, r_target, z, a, next_z, feats=None):
        r = symlog(r_target)
        r_pred = self.reward_fn(torch.cat([z, a, next_z], dim=-1).detach()).squeeze() if not self.recurrent else self.reward_fn(feats.detach()).squeeze() 
        loss = F.mse_loss(r_pred.reshape(-1), r.reshape(-1), reduction='none')
        return loss

    def duration_loss(self, tau_target, z, a):
        tau = tau_target
        _tau = self.tau(torch.cat([z, a], dim=-1).detach()).squeeze()
        t = torch.sigmoid(_tau) * 100
        loss = F.mse_loss(t.reshape(-1), tau.reshape(-1), reduction='none')
        return loss
    
    def transition_loss(self, z, next_z, next_z_dist, mask):
        '''
            -log T(z'|z, a) 
        '''
        loss = -(next_z_dist.log_prob(next_z) * mask).sum(-1) / mask.sum(-1) if not self.residual_transition \
                else -(next_z_dist.log_prob(next_z-z) * mask).sum(-1) / mask.sum(-1)
        
        return loss

    def initset_loss_from_executed(self, z, action, executed, mask, feats=None):
        _act_idx = action.argmax(-1, keepdim=True)
        _inp = feats if self.recurrent else z
        initset_pred = torch.gather(self.initsets(_inp), -1, _act_idx).squeeze(-1)
        n_pos = executed.sum(-1, keepdim=True)
        n_neg = executed.shape[1] - n_pos
        if torch.all(n_neg==0):
            pos_weight = torch.tensor(1.)
        else:
            pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)
        initset_loss = (F.binary_cross_entropy_with_logits(initset_pred, executed, pos_weight=pos_weight, reduction='none') * mask).sum(-1) / mask.sum(-1)
        return initset_loss
    
    def goal_loss(self, z, target):
        _inp = z
        goal_pred = torch.softmax(self.goal_class(_inp).squeeze(1), -1)
        n_pos = target.sum(-1, keepdim=True)
        # n_neg = target.shape[0] - n_pos
        n_neg = (1-target).sum(-1, keepdim=True)
        if torch.any(n_neg==0):
            pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)
        else:
            pos_weight = torch.tensor(1.)
        pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)
        try:
            # loss = F.binary_cross_entropy_with_logits(goal_pred, target, pos_weight=pos_weight)
            loss = F.cross_entropy(goal_pred, target.long())
        except Exception as e:
            printarr(z, target, goal_pred)
        # accuracy
        # goal_pred = (goal_pred.sigmoid() > 0.5).float() 
        goal_pred = goal_pred.argmax(-1)
        n_pos = target.sum()
        n_neg = (1-target).sum()
        tp = (goal_pred * target).sum()
        tn = ((1-goal_pred) * (1-target)).sum()
        fp = (goal_pred * (1-target)).sum()
        fn = ((1-goal_pred) * target).sum()
        tpr = tp / n_pos
        tnr =  tn / n_neg
        f1 =  2*tp / (2*tp + fn + fp)

        log_dict = {
            'goal_pred/tpr': tpr,
            'goal_pred/tnr': tnr,
            'goal_pred/f1': f1,
        }
        return loss, log_dict
    
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
        self.device = cfg_fabric.accelerator
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
    def setup_trainer(self, cfg=None):
        if self.fabric is None:
            self.setup_fabric(cfg.fabric)
        self.fabric.launch()
        self.configure_optimizers()
        self._fabric_model, self.optimizer = self.fabric.setup(self, self.optimizer)

    def set_outdir(self, outdir):
        self.outdir = outdir

    def setup_replay(self, data_path=None):
        data_path = f'{self.outdir}/data' if not data_path else data_path
        self.data = TrajectoryReplayBufferStored(int(self.cfg.data.buffer_size), save_path=data_path, device=self.device)

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

    
    ####################### SAVING ############################

    def log(self, log_dict, step):
       if self.fabric is None:
            print(f'Fabric is not initialized')
            return 
       self.fabric.log_dict(log_dict, step=step)

    def get_model_state(self):
        state = {
            "cfg": self.cfg,
            "timestep": self.timestep,
            "n_gradient_steps": self.n_gradient_steps,
            "optimizer": self.optimizer,
            **self._models
        }
        return state

    def save_checkpoint(self, ckpt_path=None):
        '''
            Save model to checkpoint
        '''
        if self.fabric is None:
            raise ValueError("Fabric not initialized")
        if ckpt_path is None:
            ckpt_path = f'{self.outdir}/checkpoints/{self.name}.ckpt'
        self.fabric.save(ckpt_path, self.get_model_state())
    
    
    @classmethod
    def load_checkpoint(cls, ckpt_path, fabric=None):
        if fabric is None:
            fabric = L.Fabric()
        loaded_ckpt = fabric.load(ckpt_path, {})
        cfg = loaded_ckpt['cfg']
        mdp = cls(cfg)
        mdp.timestep = loaded_ckpt['timestep']
        mdp.n_gradient_steps = loaded_ckpt['n_gradient_steps']
        state = mdp.get_model_state()
        for mdl in mdp._models.keys():
            state[mdl].load_state_dict(loaded_ckpt[mdl])
        return mdp
    
    def to(self, device):
        self.device = device
        if self.data:
            self.data.to(device)
        for _, m in self._models.items():
            if isinstance(m, nn.Module):
                m.to(device)
    
    def __repr__(self):
        return nn.Module.__repr__(self)
    def __str__(self):
        return nn.Module.__str__(self)
