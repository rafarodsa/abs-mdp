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

class InfoNCEAbstraction(pl.LightningModule):
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
        # printarr(z)
        next_z = self.transition.distribution(t_in).mean
        return next_z

    def _run_step(self, s, a, next_s, initset_s, reward, duration):
        
        # sample encoding of (s, s') and add noise
        z = self.encoder(s)
        z_c = z
        next_z_c = self.encoder(next_s)
        noise_std = 0.2
        z = z + torch.randn_like(z) * noise_std
        next_z  =  next_z_c + torch.randn_like(z) * noise_std 

        # printarr(t_in, actions, z )
        grounding_loss = self.grounding_loss(next_z, next_s)
        transition_loss = self.consistency_loss(z, next_z, a)
        tpc_loss = self.tpc_loss(z, next_z, a)
        _, _, rews = reward
        reward_loss = self.reward_loss(rews[:, 0], z_c, a, next_z_c)
        tau_loss = self.duration_loss(duration, z_c, a)

        # initsets
        initset_pred = self.initsets(z_c)
        initset_loss = F.binary_cross_entropy_with_logits(initset_pred, initset_s, reduction='none').mean(-1)
        # printarr(info_loss_z, infomax_loss, transition_loss, z_norm, initset_loss)
        return grounding_loss.mean(), transition_loss.mean(), tpc_loss.mean(), initset_loss.mean(), reward_loss.mean(), tau_loss.mean()


    def step(self, batch, batch_idx):
        s, a, next_s, executed, initset_s, reward, duration = batch.obs, batch.action, batch.next_obs, batch.executed, batch.initsets, batch.rewards, batch.duration.float()
        assert torch.all(executed) # check all samples are successful executions.

        grounding_loss, transition_loss, tpc_loss, initset_loss, reward_loss, tau_loss = self._run_step(s, a, next_s, initset_s, reward, duration)

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
            'loss': loss

        }
        # logger.debug(f'Losses: {logs}')
        return loss, logs
	
    def consistency_loss(self, z, next_z, action):
        '''
            -log T(z'|z, a) 
        '''
        return -self.transition.distribution(torch.cat([z, action], dim=-1)).log_prob(next_z)
	
    def grounding_loss(self, next_z, next_s):
        '''
            -MI(s'; \phi(s')) 
        '''
        b = next_z.shape[0]
        _next_s = next_s.repeat(b, *[1 for _ in range(len(next_s.shape)-1)])
        _next_z = torch.repeat_interleave(next_z, b, dim=0)
        _log_t = torch.tanh(self.grounding(_next_s, _next_z).reshape(b, b)) * np.log(b) * 0.5
        _loss = torch.diag(_log_t) - (torch.logsumexp(_log_t, dim=-1) - np.log(b))

        return -_loss
    

    def tpc_loss(self, z, next_z, a):
        '''
            -MI(z'; z, a)
        '''
        b = next_z.shape[0]
        _z_a = torch.cat([z, a], dim=-1).repeat(b, 1)
        _next_z = torch.repeat_interleave(next_z, b, dim=0)
        # printarr(_z_a, _next_z)
        _log_t = self.transition.distribution(_z_a).log_prob(_next_z).reshape(b, b)
        _loss = torch.diag(_log_t) - (torch.logsumexp(_log_t, dim=-1) - np.log(b))

        return -_loss
    
    def reward_loss(self, r_target, z, a, next_z):
        '''
            MSE(R, R_pred)
        '''
        r = symlog(r_target)
        r_pred = self.reward_fn(torch.cat([z, a, next_z], dim=-1).detach()).squeeze()
        loss = F.mse_loss(r, r_pred, reduction='none')
        return loss


    def duration_loss(self, tau_target, z, a):
        tau = symlog(tau_target)
        t = self.tau(torch.cat([z, a], dim=-1).detach()).squeeze()
        loss = F.mse_loss(tau, t, reduction='none')

        return loss

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s, a, next_s, executed, initset_s, reward, duration = batch.obs, batch.action, batch.next_obs, batch.executed, batch.initsets, batch.rewards, batch.duration.float()
        assert torch.all(executed) # check all samples are successful executions.

        grounding_loss, transition_loss, tpc_loss, initset_loss, reward_loss, tau_loss = self._run_step(s, a, next_s, initset_s, reward, duration)    

        z = self.encoder(s)
        t_in = torch.cat([z, a], dim=-1)
        next_z = self.transition.distribution(t_in).mean
        nll_loss = self.grounding_loss(next_z, next_s).mean()

        logs = {
                'val_infomax': grounding_loss.mean(),
                'val_transition': transition_loss.mean(),
                'val_tpc_loss': tpc_loss.mean(),
                'val_initset_loss': initset_loss.mean(),
                'val_reward_loss': reward_loss.mean(), 
                'val_tau_loss': tau_loss.mean(),
                'val_nll_loss': nll_loss.mean()
            }
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return logs['val_infomax']
	
    def test_step(self, batch, batch_idx):
        state, action, next_s = batch.obs, batch.action, batch.next_obs
        z = self.encoder(state)
        t_in = torch.cat([z, action], dim=-1)
        next_z = self.transition.distribution(t_in).mean
        nll_loss = self.grounding_loss(next_z, next_s).mean()
        self.log_dict({'nll_loss': nll_loss},on_step=False, on_epoch=True, prog_bar=True, logger=True)
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