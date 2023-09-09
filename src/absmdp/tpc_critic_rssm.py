"""
    Option Model using CPC/InfoNCE loss
    author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
    date: June 13th, 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as pl


from src.models.factories import build_distribution, build_model
from src.models.optimizer_factories import build_optimizer, build_scheduler
from src.utils.symlog import symlog
from src.absmdp.configs import TrainerConfig

from omegaconf import OmegaConf as oc

import logging

from src.utils.printarr import printarr

class RSSMAbstraction(pl.LightningModule):
    INITSET_CLASS_THRESH = 0.5

    def __init__(self, cfg: TrainerConfig):
        super().__init__()
        oc.resolve(cfg)
        self.save_hyperparameters()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.obs_dim = cfg.model.obs_dims
        self.latent_dim = cfg.model.latent_dim
        self.n_options = cfg.model.n_options
        self.encoder = build_model(cfg.model.encoder.features)
        self.transition = build_distribution(cfg.model.transition)
        self.grounding = build_model(cfg.model.decoder)
        self.initsets = build_model(cfg.model.init_class)
        self.tau = build_model(cfg.model.tau)
        self.reward_fn = build_model(cfg.model.reward)
        self.lr = cfg.optimizer.params.lr
        self.hyperparams = cfg.loss
        self.kl_const =  self.hyperparams.kl_const

    def forward(self, state, action):
        z = self.encoder(state)
        t_in = torch.cat([z, action], dim=-1)
        next_z = self.transition.distribution(t_in).mean + z
        return next_z
    
    def _get_device(self, x):
        d = x.get_device()
        return 'cpu' if d < 0 else f'cuda:{d}'
    def _run_step(self, s, a, next_s, initset_s, reward, duration, lengths, executed=None):
        
        # sample encoding of (s, s') and add noise
        z = self.encoder(s)
        z_c = z
        next_z_c = self.encoder(next_s)
        noise_std = 0.2
        z = z + torch.randn_like(z) * noise_std
        next_z  =  next_z_c + torch.randn_like(z) * noise_std 
        
        batch_size, length = s.shape[0], s.shape[1]


        _mask = torch.arange(length).to(self._get_device(s)).repeat(batch_size, 1) < lengths.unsqueeze(1)
        # grounding_loss = self.grounding_loss(next_z[_mask], next_s[_mask])
        grounding_loss = self.grounding_loss_normal(next_z[_mask])
        reward_loss = self.reward_loss(reward[_mask], z_c[_mask], a[_mask], next_z_c[_mask])
        tau_loss = self.duration_loss(duration[_mask], z_c[_mask], a[_mask])

        # transition losses
        next_z_dist = self.transition.distribution(torch.cat([z, a], dim=-1))
        transition_loss = self.consistency_loss(z, next_z, next_z_dist, mask=_mask)
        tpc_loss = self.tpc_loss(z, next_z, next_z_dist, min_length=lengths.min())
    
        # initsets 
        if self.cfg.initsets_from_success:
            initset_loss = self.initset_loss_from_executed(z, a, executed, _mask)
        else:
            initset_loss = self.initset_loss(z[_mask], initset_s[_mask])


        return grounding_loss.mean(), transition_loss.mean(), tpc_loss.mean(), initset_loss.mean(), reward_loss.mean(), tau_loss.mean()


    def initset_loss(self, z, initset_target):
        pos_samples = (initset_target==1).float()
        n_pos = pos_samples.sum(0)
        n_neg = initset_target.shape[0] - n_pos
        
        pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)
        # printarr(pos_samples, n_pos, n_neg, pos_weight)
        initset_pred = self.initsets(z)
        initset_loss = F.binary_cross_entropy_with_logits(initset_pred, initset_target, reduction='none', pos_weight=pos_weight).mean(-1)
        return initset_loss


    def initset_loss_from_executed(self, z, action, executed, mask):
        _act_idx = action.argmax(-1, keepdim=True)
        initset_pred = torch.gather(self.initsets(z), -1, _act_idx).squeeze(-1)
        n_pos = executed.sum(-1, keepdim=True)
        n_neg = executed.shape[1] - n_pos
        if torch.all(n_neg==0):
            pos_weight = torch.tensor(1.)
        else:
            pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)

        # printarr(initset_pred, executed, pos_weight, z, action)
        initset_loss = (F.binary_cross_entropy_with_logits(initset_pred, executed, pos_weight=pos_weight, reduction='none') * mask).sum(-1)
        return initset_loss


    def step(self, batch, batch_idx):
        s, a, next_s, initset_s, reward, duration, lengths, executed = batch.obs, batch.action, batch.next_obs, batch.initsets, batch.rewards, batch.duration.float(), batch.length, batch.executed
        grounding_loss, transition_loss, tpc_loss, initset_loss, reward_loss, tau_loss = self._run_step(s, a, next_s, initset_s, reward, duration, lengths, executed)

        loss = self.hyperparams.grounding_const * grounding_loss + \
                self.hyperparams.transition_const * transition_loss + \
                self.hyperparams.tpc_const * tpc_loss + \
                self.hyperparams.initset_const * initset_loss + \
                self.hyperparams.reward_const * reward_loss + \
                self.hyperparams.tau_const * tau_loss 

        

        # log std deviations for encoder.
        logs = {
            'grounding_loss': grounding_loss,
            'transition_loss': transition_loss,
            'tpc_loss': tpc_loss,
            'initset_loss': initset_loss,
            'loss': loss,

        }
        # logger.debug(f'Losses: {logs}')
        return loss, logs
	
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
	
    def grounding_loss(self, next_z, next_s):
        '''
            -MI(s'; \phi(s')) 
        '''
        b = next_z.shape[:-1]
        b_size = np.prod(b)
        s_shape = next_s.shape[1:]
        next_z, next_s = next_z.reshape(b_size, -1), next_s.reshape(b_size, *s_shape)
        _next_s = next_s.repeat(b_size, *[1 for _ in range(len(next_s.shape)-1)])
        _next_z = torch.repeat_interleave(next_z, b_size, dim=0)
        _log_t = torch.tanh(self.grounding(_next_s, _next_z).reshape(b_size, b_size)) * 100 #* np.log(b) * 0.5
        _loss = torch.diag(_log_t) - (torch.logsumexp(_log_t, dim=-1) - np.log(b_size))

        return -_loss.reshape(*b)
    

    def grounding_loss_normal(self, next_z, std=0.1):
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
        tau = torch.log(tau_target)
        t = self.tau(torch.cat([z, a], dim=-1).detach()).squeeze()
        loss = F.mse_loss(tau.reshape(-1), t.reshape(-1), reduction='none')

        return loss

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, logger=True)
        return loss

    def get_device(self, s):
        return s.get_device() if s.get_device() >= 0 else 'cpu'

    def validation_step(self, batch, batch_idx):
        s, a, next_s, executed, initset_s, reward, duration, lengths = batch.obs, batch.action, batch.next_obs, batch.executed, batch.initsets, batch.rewards, batch.duration.float(), batch.length
        # assert torch.all(executed) # check all samples are successful executions.
        grounding_loss, transition_loss, tpc_loss, initset_loss, reward_loss, tau_loss = self._run_step(s, a, next_s, initset_s, reward, duration, lengths, executed=executed)    

        batch_size, length = s.shape[0], s.shape[1]
        _mask = torch.arange(length).to(self.get_device(s)).repeat(batch_size, 1) < lengths.unsqueeze(1)

        z = self.encoder(s)
        t_in = torch.cat([z, a], dim=-1)
        next_z = self.transition.distribution(t_in).mean + z
        initset_pred = (torch.sigmoid(self.initsets(z[_mask])) > self.INITSET_CLASS_THRESH).float()
        initset_acc = (initset_pred == initset_s[_mask]).float().mean()
        initset_pred = torch.sigmoid(self.initsets(z))
        executed_pred = torch.gather((initset_pred > self.INITSET_CLASS_THRESH).float(), -1, a.argmax(-1, keepdim=True)).squeeze()
        executed_acc = (executed_pred[_mask] == executed[_mask]).float().mean()

        # nll_loss = self.grounding_loss(next_z[_mask], next_s[_mask]).mean()
        nll_loss = self.grounding_loss_normal(next_z[_mask])

        logs = {
                'val_infomax': grounding_loss.mean(),
                'val_transition': transition_loss.mean(),
                'val_tpc_loss': tpc_loss.mean(),
                'val_executed_acc': executed_acc,
                'val_initset_loss': initset_loss.mean(),
                'val_initset_acc': initset_acc,
                'val_reward_loss': reward_loss.mean(), 
                'val_tau_loss': tau_loss.mean(),
                'val_nll_loss': nll_loss.mean(),
            }
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return logs['val_infomax']
	
    def test_step(self, batch, batch_idx):
        s, a, next_s, executed, initset_s, reward, duration, lengths = batch.obs, batch.action, batch.next_obs, batch.executed, batch.initsets, batch.rewards, batch.duration.float(), batch.length
        grounding_loss, transition_loss, tpc_loss, initset_loss, reward_loss, tau_loss = self._run_step(s, a, next_s, initset_s, reward, duration, lengths, executed=executed)    

        batch_size, length = s.shape[0], s.shape[1]
        _mask = torch.arange(length).to(self._get_device(s)).repeat(batch_size, 1) < lengths.unsqueeze(1)

        z = self.encoder(s)
        t_in = torch.cat([z, a], dim=-1)
        next_z = self.transition.distribution(t_in).mean + z
        initset_pred = (torch.sigmoid(self.initsets(z[_mask])) > self.INITSET_CLASS_THRESH).float()
        initset_acc = (initset_pred == initset_s[_mask]).float().mean()

        # nll_loss = self.grounding_loss(next_z[_mask], next_s[_mask]).mean()
        nll_loss = self.grounding_loss_normal(next_z[_mask]).mean()
        self.log_dict({
                       'nll_loss': nll_loss,
                       'initset_acc': initset_acc,
                       'initset_loss': initset_loss.mean(),
                       'reward_loss': reward_loss.mean(),
                       'transition_loss': transition_loss.mean(),
                       'tau_loss': tau_loss.mean()
                       }
                       ,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return nll_loss
		
    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg.optimizer, self.parameters())
        scheduler = build_scheduler(self.cfg.scheduler, optimizer)
        return {"optimizer": optimizer, "scheduler": scheduler}
	
    @staticmethod
    def load_config(path):
        try:
            with open(path, "r") as f:
                #TODO add structured configs when they have settled.
                # cfg = oc.merge(oc.structured(TrainerConfig), oc.load(f))
                cfg = oc.load(f)
                return cfg
        except FileNotFoundError:
            raise ValueError(f"Could not find config file at {path}")
        

class RSSMAbstractModel(RSSMAbstraction):
    def __init__(self, cfg):
        oc.resolve(cfg)
        cfg.model.transition.features.latent_dim = cfg.model.latent_dim + 1 # add executed flag
        super().__init__(cfg)
        self.grounding = None
        self.success = build_model(cfg.model.init_class)


    def _run_step(self, s, a, next_s, initset_s, reward, duration, lengths, executed=None):
                
        # sample encoding of (s, s') and add noise
        z = self.encoder(s)
        z_c = z
        next_z_c = self.encoder(next_s)
        noise_std = 0.2
        z = z + torch.randn_like(z) * noise_std
        next_z  =  next_z_c + torch.randn_like(z) * noise_std 
        
        batch_size, length = s.shape[0], s.shape[1]


        _mask = torch.arange(length).to(self._get_device(s)).repeat(batch_size, 1) < lengths.unsqueeze(1)
        grounding_loss = self.grounding_loss_normal(next_z[_mask])
        reward_loss = self.reward_loss(reward[_mask], z_c[_mask], a[_mask], next_z_c[_mask])
        tau_loss = self.duration_loss(duration[_mask], z_c[_mask], a[_mask])

        # transition losses
        next_z_dist = self.transition.distribution(torch.cat([z, a, executed.unsqueeze(-1)], dim=-1))
        transition_loss = self.consistency_loss(z, next_z, next_z_dist, mask=_mask)
        tpc_loss = self.tpc_loss(z, next_z, next_z_dist, min_length=lengths.min())


        # initsets 
        initset_loss = self.initset_loss_from_executed(z, a, executed, _mask)
        success_loss = self.prob_success_loss(z[_mask], a[_mask], executed[_mask])


        return grounding_loss.mean(), transition_loss.mean(), tpc_loss.mean(), initset_loss.mean(), reward_loss.mean(), tau_loss.mean(), success_loss.mean()
    
    def prob_success_loss(self, z, a, executed):
        success_logits = torch.gather(self.success(z), -1, a.argmax(-1, keepdim=True)).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(success_logits, executed, reduction='none')
        return loss
    
    def step(self, batch, batch_idx):
        s, a, next_s, initset_s, reward, duration, lengths, executed = batch.obs, batch.action, batch.next_obs, batch.initsets, batch.rewards, batch.duration.float(), batch.length, batch.executed
        grounding_loss, transition_loss, tpc_loss, initset_loss, reward_loss, tau_loss, prob_success = self._run_step(s, a, next_s, initset_s, reward, duration, lengths, executed)

        loss = self.hyperparams.grounding_const * grounding_loss + \
                self.hyperparams.transition_const * transition_loss + \
                self.hyperparams.tpc_const * tpc_loss + \
                self.hyperparams.initset_const * initset_loss + \
                self.hyperparams.reward_const * reward_loss + \
                self.hyperparams.tau_const * tau_loss + \
                prob_success 
        

        # log std deviations for encoder.
        logs = {
            'grounding_loss': grounding_loss,
            'transition_loss': transition_loss,
            'tpc_loss': tpc_loss,
            'initset_loss': initset_loss,
            'loss': loss

        }
        # logger.debug(f'Losses: {logs}')
        return loss, logs
    

    def validation_step(self, batch, batch_idx):
        s, a, next_s, executed, initset_s, reward, duration, lengths = batch.obs, batch.action, batch.next_obs, batch.executed, batch.initsets, batch.rewards, batch.duration.float(), batch.length
        # assert torch.all(executed) # check all samples are successful executions.
        grounding_loss, transition_loss, tpc_loss, initset_loss, reward_loss, tau_loss, prob_success = self._run_step(s, a, next_s, initset_s, reward, duration, lengths, executed=executed)    

        batch_size, length = s.shape[0], s.shape[1]
        _mask = torch.arange(length).to(self.get_device(s)).repeat(batch_size, 1) < lengths.unsqueeze(1)

        z = self.encoder(s)
        t_in = torch.cat([z, a, executed.unsqueeze(-1)], dim=-1)
        next_z = self.transition.distribution(t_in).mean + z
        initset_pred = (torch.sigmoid(self.initsets(z[_mask])) > self.INITSET_CLASS_THRESH).float()
        initset_acc = (initset_pred == initset_s[_mask]).float().mean()
        initset_pred = torch.sigmoid(self.initsets(z))
        executed_pred = torch.gather((initset_pred > self.INITSET_CLASS_THRESH).float(), -1, a.argmax(-1, keepdim=True)).squeeze()
        executed_acc = (executed_pred[_mask] == executed[_mask]).float().mean()

        nll_loss = self.grounding_loss_normal(next_z[_mask])

        logs = {
                'val_infomax': grounding_loss.mean(),
                'val_transition': transition_loss.mean(),
                'val_tpc_loss': tpc_loss.mean(),
                'val_executed_acc': executed_acc,
                'val_prob_success': prob_success.mean(),
                'val_initset_loss': initset_loss.mean(),
                'val_initset_acc': initset_acc,
                'val_reward_loss': reward_loss.mean(), 
                'val_tau_loss': tau_loss.mean(),
                'val_nll_loss': nll_loss.mean(),
            }
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return logs['val_infomax']
	
    def test_step(self, batch, batch_idx):
        s, a, next_s, executed, initset_s, reward, duration, lengths = batch.obs, batch.action, batch.next_obs, batch.executed, batch.initsets, batch.rewards, batch.duration.float(), batch.length
        grounding_loss, transition_loss, tpc_loss, initset_loss, reward_loss, tau_loss, prob_success = self._run_step(s, a, next_s, initset_s, reward, duration, lengths, executed=executed)    

        batch_size, length = s.shape[0], s.shape[1]
        _mask = torch.arange(length).to(self._get_device(s)).repeat(batch_size, 1) < lengths.unsqueeze(1)

        z = self.encoder(s)
        t_in = torch.cat([z, a, executed.unsqueeze(-1)], dim=-1)
        next_z = self.transition.distribution(t_in).mean + z
        initset_pred = (torch.sigmoid(self.initsets(z[_mask])) > self.INITSET_CLASS_THRESH).float()
        initset_acc = (initset_pred == initset_s[_mask]).float().mean()

        nll_loss = self.grounding_loss_normal(next_z[_mask]).mean()
        self.log_dict({
                        'nll_loss': nll_loss,
                        'initset_acc': initset_acc,
                        'initset_loss': initset_loss.mean(),
                        'reward_loss': reward_loss.mean(),
                        'transition_loss': transition_loss.mean(),
                        'tau_loss': tau_loss.mean(),
                        'prob_success': prob_success.mean()
                       }
                       ,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return nll_loss
