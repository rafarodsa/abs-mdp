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
        self.grounding = build_distribution(cfg.model.decoder)
        self.initsets = build_model(cfg.model.init_class)
        self.lr = cfg.optimizer.params.lr
        self.hyperparams = cfg.loss
        self.kl_const =  self.hyperparams.kl_const


    def forward(self, state, action):
        z = self.encoder(state)
        t_in = torch.cat([z, action], dim=-1)
        # printarr(z)
        next_z = self.transition.distribution(t_in).mean + z
        q_s_prime = self.grounding.distribution(next_z)
        q_s = self.grounding.distribution(z)
        return q_s, q_s_prime

    def _run_step(self, s, a, next_s, initset_s):
        
        # sample encoding of (s, s') and add noise
        z = self.encoder(s)
        z_norm = z.pow(2).sum(-1)
        noise_std = 0.2
        z = z + torch.rand_like(z) * noise_std
        next_z  = self.encoder(next_s) + torch.rand_like(z) * noise_std 

        # printarr(t_in, actions, z )
        grounding_loss = self.grounding_loss(next_z, next_s)
        transition_loss = self.consistency_loss(z, next_z, a)
        tpc_loss = self.tpc_loss(z, next_z, a)

        # initsets
        initset_pred = self.initsets(z)
        initset_loss = F.binary_cross_entropy_with_logits(initset_pred, initset_s, reduction='none').mean(-1)
        # printarr(info_loss_z, infomax_loss, transition_loss, z_norm, initset_loss)
        return grounding_loss.mean(), transition_loss.mean(), tpc_loss.mean(), initset_loss.mean()


    def step(self, batch, batch_idx):
        s, a, next_s, executed, initset_s = batch.obs, batch.action, batch.next_obs, batch.executed, batch.initsets
        assert torch.all(executed) # check all samples are successful executions.

        grounding_loss, transition_loss, tpc_loss, initset_loss = self._run_step(s, a, next_s, initset_s)

        loss = self.hyperparams.grounding_const * grounding_loss + \
                self.hyperparams.transition_const * transition_loss + \
                self.hyperparams.tpc_const * tpc_loss + \
                self.hyperparams.initset_const * initset_loss 
        

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
        return -self.transition.distribution(torch.cat([z, action], dim=-1)).log_prob(next_z-z)
	
    def grounding_loss(self, next_z, next_s):
        '''
            -MI(s'; \phi(s')) 
        '''
        return -self.grounding.distribution(next_z).log_prob(next_s)

    def tpc_loss(self, next_z, z, a):
        '''
            -MI(z'; z, a)
        '''
        b = next_z.shape[0]
        _z_a = torch.cat([z, a], dim=-1).repeat(b, 1)
        _next_z = torch.repeat_interleave(next_z, b, dim=0)
        # printarr(_z_a, _next_z)
        _log_t = self.transition.distribution(_z_a).log_prob(_next_z-_z_a[..., :self.latent_dim]).reshape(b, b)
        _loss = torch.diag(_log_t) - (torch.logsumexp(_log_t, dim=-1) + np.log(b))

        return -_loss
    
    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s, a, next_s, executed, initset_s = batch.obs, batch.action, batch.next_obs, batch.executed, batch.initsets
        assert torch.all(executed) # check all samples are successful executions.
        
        grounding_loss, transition_loss, tpc_loss, initset_loss = self._run_step(s, a, next_s, initset_s)

        # log std deviations for encoder.
        _, q_next_s = self.forward(s, a)
        nll_loss = -q_next_s.log_prob(next_s).mean()
        logs = {
                'val_infomax': grounding_loss.mean(),
                'val_transition': transition_loss.mean(),
                'val_tpc_loss': tpc_loss.mean(),
                'val_initset_loss': initset_loss.mean(),
                'val_nll': nll_loss,
            }
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return nll_loss
	
    def test_step(self, batch, batch_idx):
        state, action, next_s = batch.obs, batch.action, batch.next_obs
        z = self.encoder(state)
        t_in = torch.cat([z, action], dim=-1)
        next_z = self.transition.distribution(t_in).mean + z
        q_s_prime = self.grounding.distribution(torch.cat([next_z, torch.zeros_like(action)], dim=-1))

        nll_loss = -q_s_prime.log_prob(next_s).mean()
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