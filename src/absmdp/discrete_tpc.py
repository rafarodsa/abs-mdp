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
from src.models.grid_quantizer import GridQuantizerST
from src.utils.symlog import symlog
from src.absmdp.configs import TrainerConfig

from omegaconf import OmegaConf as oc

import logging

from src.utils.printarr import printarr

class DiscreteInfoNCEAbstraction(pl.LightningModule):
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
        self.quantizer = build_model(cfg.model.encoder.codebook)
        self.transition = build_distribution(cfg.model.transition)
        self.transition.set_codebook(self.quantizer)
        self.grounding = build_distribution(cfg.model.decoder)
        self.initsets = build_model(cfg.model.init_class)
        self.lr = cfg.optimizer.params.lr
        self.hyperparams = cfg.loss
        self.kl_const =  self.hyperparams.kl_const


    def forward(self, state, action):
        z = self.encoder(state)
        t_in = torch.cat([z, action], dim=-1)
        # printarr(z)
        
        next_z = self.transition.distribution(t_in).mode.reshape(state.shape[0], -1)
        q_s_prime = self.grounding.distribution(next_z)
        q_s = self.grounding.distribution(z)
        return q_s, q_s_prime

    def _run_step(self, s, a, next_s, initset_s):
        b = s.shape[0]
        # sample encoding of (s, s') and add noise
        z = self.encoder(s)
        noise_std = 0.
        z = z + torch.randn_like(z) * noise_std
        next_z  = self.encoder(next_s) + torch.randn_like(z) * noise_std 

        # quantize
        # z_q, z_q_idx = self.quantizer(z.reshape(b, self.latent_dim, -1))
        # next_z_q, next_z_q_idx = self.quantizer(next_z.reshape(b, self.latent_dim, -1))

        z_q, z_q_idx = self.quantizer(z.reshape(b, 1, -1))
        next_z_q, next_z_q_idx = self.quantizer(next_z.reshape(b, 1, -1))


        grounding_loss = self.grounding_loss(next_z_q.reshape(b,-1), next_s)
        transition_loss = self.consistency_loss(z_q.reshape(b, -1), next_z_q_idx, a)
        # printarr(z_q, next_z_q, z_q_idx,  next_z_q_idx, a)
        tpc_loss = self.tpc_loss(z_q.reshape(b, -1), next_z_q_idx, a)

        # discrete representation loss
        representation_loss = self.discrete_representation_loss(z, z_q.reshape(b, -1), next_z, next_z_q.reshape(b, -1), _lamb=self.hyperparams.commitment_const)


        # initsets
        initset_pred = self.initsets(z_q.reshape(b, -1))
        initset_loss = F.binary_cross_entropy_with_logits(initset_pred, initset_s, reduction='none').mean(-1)
        # printarr(info_loss_z, infomax_loss, transition_loss, z_norm, initset_loss)
        return grounding_loss.mean(), transition_loss.mean(), tpc_loss.mean(), initset_loss.mean(), representation_loss.mean()


    def step(self, batch, batch_idx):
        s, a, next_s, executed, initset_s = batch.obs, batch.action, batch.next_obs, batch.executed, batch.initsets
        assert torch.all(executed) # check all samples are successful executions.

        grounding_loss, transition_loss, tpc_loss, initset_loss, representation_loss = self._run_step(s, a, next_s, initset_s)

        loss = self.hyperparams.grounding_const * grounding_loss + \
                self.hyperparams.transition_const * transition_loss + \
                self.hyperparams.tpc_const * tpc_loss + \
                self.hyperparams.initset_const * initset_loss + \
                self.hyperparams.representation_const * representation_loss

        # log std deviations for encoder.
        logs = {
            'grounding_loss': grounding_loss,
            'transition_loss': transition_loss,
            'tpc_loss': tpc_loss,
            'initset_loss': initset_loss,
            'representation_loss': representation_loss,
            'loss': loss
        }
        # logger.debug(f'Losses: {logs}')
        return loss, logs
	
    def consistency_loss(self, z, next_z, action):
        '''
            -log T(z'|z, a) 
        '''
        printarr(z, next_z, action)
        return -self.transition.distribution(torch.cat([z, action], dim=-1)).log_prob(next_z)
	
    def grounding_loss(self, next_z, next_s):
        '''
            -MI(s'; \phi(s')) 
        '''
        return -self.grounding.distribution(next_z).log_prob(next_s)

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
    
    def discrete_representation_loss(self, z, z_q, next_z, next_z_q, _lamb=0.25):
        '''
            Commitment loss: ||z - SG(z_q)||^2 + ||next_z - SG(next_z_q)||^2
            Quantization loss: ||z_q - SG(z)||^2 + ||next_z_q - SG(next_z)||^2
        '''
        printarr(z, z_q, next_z, next_z_q, self.quantizer.codebook)
        l_q = F.mse_loss(z_q, z.detach(), reduction='none').sum(-1)
        l_q_next = F.mse_loss(next_z_q, next_z.detach(), reduction='none').sum(-1)
        l_c = F.mse_loss(z, z_q.detach(), reduction='none').sum(-1)
        l_c_next = F.mse_loss(next_z, next_z_q.detach(), reduction='none').sum(-1)

        # printarr(l_q, l_q_next, l_c, l_c_next)
        return 0.5 * ((l_q + l_q_next) + _lamb * (l_c + l_c_next))

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s, a, next_s, executed, initset_s = batch.obs, batch.action, batch.next_obs, batch.executed, batch.initsets
        assert torch.all(executed) # check all samples are successful executions.
        
        grounding_loss, transition_loss, tpc_loss, initset_loss, representation_loss = self._run_step(s, a, next_s, initset_s)

        # log std deviations for encoder.
        _, q_next_s = self.forward(s, a)
        nll_loss = -q_next_s.log_prob(next_s).mean()
        logs = {
                'val_infomax': grounding_loss.mean(),
                'val_transition': transition_loss.mean(),
                'val_tpc_loss': tpc_loss.mean(),
                'val_initset_loss': initset_loss.mean(),
                'val_nll': nll_loss,
                'val_representation_loss': representation_loss.mean()
            }
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return nll_loss
	
    def test_step(self, batch, batch_idx):
        state, action, next_s = batch.obs, batch.action, batch.next_obs
        z = self.encoder(state)
        t_in = torch.cat([z, action], dim=-1)
        next_z = self.transition.distribution(t_in).mode
        q_s_prime = self.grounding.distribution(next_z)

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