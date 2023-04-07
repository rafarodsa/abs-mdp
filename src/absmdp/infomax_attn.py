"""
	Option Model (Abstract MDPs)
	based on the information bottleneck principle
	with deterministic encoder.

	author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
	date: 6 April 2023
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


from src.models.factories import build_distribution, build_model

from src.utils.symlog import symlog
from src.absmdp.configs import TrainerConfig

from omegaconf import OmegaConf as oc

import logging

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('medium')

class InfomaxAbstraction(pl.LightningModule):
	def __init__(self, cfg: TrainerConfig):
		super().__init__()
		oc.resolve(cfg)
		self.save_hyperparameters()
		self.data_cfg = cfg.data
		self.obs_dim = cfg.model.obs_dims
		self.latent_dim = cfg.model.latent_dim
		self.n_options = cfg.model.n_options


		self.n_heads = 8
		self.key_query_dim = 16
		self.value_dim = 8
		self.option_embedding = nn.Embedding(self.n_options, self.key_query_dim)
		self.query_value = nn.Linear(self.latent_dim, (self.key_query_dim + self.value_dim) * self.n_heads)
		self.projection = nn.Linear(self.value_dim, self.latent_dim)

		self.encoder = build_model(cfg.model.encoder.features)
		self.transition = build_distribution(cfg.model.transition)
		self.grounding = build_distribution(cfg.model.decoder)
		self.decoder = build_distribution(cfg.model.decoder)
		self.initsets = build_model(cfg.model.init_class)
		self.lr = cfg.lr
		self.hyperparams = cfg.loss
		self.kl_const =  self.hyperparams.kl_const

	def forward(self, state, action):
		z = self.encoder(state)
		t_in = torch.cat([z, action], dim=-1)
		next_z = self.transition.distribution(t_in).mean + z
		# q_s_prime = self.grounding.distribution(torch.cat([next_z, torch.zeros_like(action)], dim=-1))
		q_s_prime = self.grounding.distribution(t_in)
		q_s = self.decoder.distribution(torch.cat([next_z, torch.zeros_like(action)], dim=-1))
		return q_s, q_s_prime

	def _run_step(self, s, a, next_s, initset_s):
		# sample encoding of (s, s')
		z = self.encoder(s)
		z_norm = z.pow(2).sum(-1)
		z = z + torch.rand_like(z) * 1e-2
		next_z  = self.encoder(next_s) + torch.rand_like(z) * 1e-2

		actions = a # dims: (batch, n_actions)
		inferred_z = z #torch.randn(z.shape).to(z.get_device()) * q_z.std + z 
		
		# t_in = self.projection(F.relu(self.attn(inferred_z, actions)))
		t_in = z.squeeze()
		# next_z_pred, _, _ = self.transition.sample_n_dist(torch.cat([t_in, actions], dim=-1))
		# next_z_pred = next_z_pred.squeeze() + z
		t_in = torch.cat([t_in, actions], dim=-1)
		next_s_q_pred = self.grounding.distribution(t_in.squeeze())
		infomax_loss = self.infomax_loss(next_s, next_s_q_pred, n_samples=1)
		
		z_in = torch.cat([z, torch.zeros_like(actions)], dim=-1)
		s_q_pred = self.grounding.distribution(z_in.detach())
		info_loss_z = self.infomax_loss(s, s_q_pred, n_samples=1)
		
		transition_loss = self.transition_loss(z.detach(), next_z.detach(), a)

		# decode next latent state
		# q_next_s = self.decoder.distribution(next_z)
		# q_s = self.decoder.distribution(z)
		self.decoder.load_state_dict(self.grounding.state_dict())
		
		q_s = self.decoder.freeze().distribution(z_in)
		info_loss_z = -self.info_bottleneck(s, q_s) + info_loss_z
		
		# initsets
		initset_pred = torch.sigmoid(self.initsets(z))
		initset_loss = F.binary_cross_entropy(initset_pred, initset_s)

		

		return infomax_loss, transition_loss, info_loss_z, initset_loss, z_norm


	def step(self, batch, batch_idx):
		s, a, next_s, executed, initset_s = batch.obs, batch.action, batch.next_obs, batch.executed, batch.initsets
		assert torch.all(executed) # check all samples are successful executions.
		
		infomax, transition_loss, info_loss_z, initset_loss, z_norm = self._run_step(s, a, next_s, initset_s)

		# compute total loss
		free_bits = torch.minimum(z_norm-1e-2, torch.zeros_like(z_norm))
		loss = infomax + self.hyperparams.kl_const * info_loss_z + initset_loss * self.hyperparams.initset_const + transition_loss * self.hyperparams.transition_const + 0.01 * z_norm - free_bits
		loss = loss.mean()

		# log std deviations for encoder.
		logs = {
			'infomax': infomax.mean(),
			'info_loss_z': info_loss_z.mean(),
			'transition_loss': transition_loss.mean(),
			'initset_loss': initset_loss.mean(),
			'train_loss': loss, 
			'z_norm': z_norm.mean()
		}
		logger.debug(f'Losses: {logs}')
		return loss, logs
	

	def attn(self, z, o):
		n_option = o.argmax(dim=-1)
		option_embedding = self.option_embedding(n_option)
		query, value = self.query_value(z).split([self.key_query_dim * self.n_heads, self.value_dim * self.n_heads], dim=-1)
		value = value.reshape(value.shape[0], self.n_heads, self.value_dim)
		query = query.reshape(query.shape[0], self.n_heads, self.key_query_dim)
		attn_weights = torch.einsum('bk,bhq->bh', option_embedding, query) / self.key_query_dim ** 0.5
		attn_weights = torch.softmax(attn_weights, dim=-1)
		attn_value = torch.einsum('bh,bhv->bv', attn_weights, value)
		return attn_value


	def infomax_loss(self, next_s, q_next_s, n_samples=1):
		if n_samples == 1:
			return -q_next_s.log_prob(next_s)
		next_s_ = next_s.unsqueeze(0).repeat_interleave(n_samples, dim=0)
		return -torch.log(torch.exp(q_next_s.log_prob(next_s_)).mean(0))
	
	def info_bottleneck(self, s, decoded_q_s):
		return decoded_q_s.log_prob(s)
	
	def transition_loss(self, z, next_z, action):
		return -self.transition.distribution(torch.cat([z, action], dim=-1)).log_prob(next_z-z)
	
	def inference_loss(self, z, inferred_q):
		return -inferred_q.detach().log_prob(z)

	def training_step(self, batch, batch_idx):
		loss, logs = self.step(batch, batch_idx)
		self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
		self.log_dict({'nll': logs['infomax']}, on_step=True, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		s, a, next_s = batch.obs, batch.action, batch.next_obs
		qs, q_s_prime = self.forward(s, a)
		
		nll_loss = -q_s_prime.log_prob(next_s).mean()
		self.log_dict({'nll_loss': nll_loss},on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return nll_loss
		
	def configure_optimizers(self):

		optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
		# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
		# return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "nll_loss"}
		scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=750)
		return {"optimizer": optimizer, "scheduler": scheduler}
	@staticmethod
	def load_config(path):
		try:
			with open(path, "r") as f:
				cfg = oc.merge(oc.structured(TrainerConfig), oc.load(f))
				return cfg
		except FileNotFoundError:
			raise ValueError(f"Could not find config file at {path}")
		

class AbstractMDPTrainer(pl.LightningModule):
    def __init__(self, cfg, phi: InfomaxAbstraction):
        super().__init__()
        self.phi = phi
        self.cfg = cfg
        self.n_options = cfg.model.n_options
        self.latent_dim = cfg.model.latent_dim
        self.obs_dim = cfg.model.obs_dims
        self.lr = cfg.lr
        self.hyperparams = cfg.loss
    
        # build models
        self.reward = build_model(cfg.model.reward)
        self.tau = build_model(cfg.model.tau)
        self.initial_state = []

    def _step(self, s, a, next_s):
        # encode states
        with torch.no_grad():
            z, next_z = self.phi.encoder(s), self.phi.encoder(next_s)
            q_s, q_next_s = self.phi.grounding.distribution(z), self.phi.grounding.distribution(next_z)
        reward_prediction = self.reward(torch.cat([z, a, next_z], dim=-1))
        tau_prediction = self.tau(torch.cat([z, a], dim=-1))
        return z, next_z, reward_prediction, tau_prediction, q_s, q_next_s
    
    def _reward_loss(self, reward_pred, q_s, q_s_prime, reward_target):
        logger.debug('Reward loss')
        obs, next_obs, rewards = reward_target  # batch x n_samples x obs_dim
        
        obs, next_obs = obs.transpose(0, 1).unsqueeze(1), next_obs.transpose(0, 1).unsqueeze(1)  # n_samples x batch x obs_dim
        logger.debug(f'obs: {obs.shape}, next_obs: {next_obs.shape}')

        _reward_target = symlog(rewards).transpose(0, 1).unsqueeze(1) # n_samples x 1 x batch
        logger.debug(f'reward_target: {_reward_target.shape}')

        weights = torch.exp((q_s.log_prob(obs) + q_s_prime.log_prob(next_obs)))
        Z = weights.sum(0, keepdims=True) # sum over samples
        logger.debug(f'Reward weights: {Z}')
        logger.debug(f'weights: {weights.shape}, Z: {Z.shape}')
        reward_target_ = (weights*_reward_target/Z).sum(0).squeeze()

        logger.debug(f'reward_target_: {reward_target_.shape}, reward_pred: {reward_pred.shape}')
        logger.debug('Reward loss done')
        return torch.nn.functional.mse_loss(reward_pred.squeeze(), reward_target_.squeeze(), reduction="mean")
    
    def _tau_loss(self, tau_pred, tau_target):
        return torch.nn.functional.mse_loss(tau_pred.squeeze(), tau_target.squeeze(), reduction="mean")

    def training_step(self, batch, batch_idx):
        s, a, next_s, rews, duration, p_0 = batch.obs, batch.action, batch.next_obs, batch.rewards, batch.duration.float(), batch.p0

        z, next_z, reward_prediction, tau_prediction, q_s, q_next_s = self._step(s, a, next_s)

        # compute loss
        reward_loss = self._reward_loss(reward_prediction, q_s, q_next_s, rews)
        tau_loss = self._tau_loss(tau_prediction, duration)
        loss = self.hyperparams['reward_const'] * reward_loss + self.hyperparams['tau_const'] * tau_loss

        # log
        self.log('train_loss', loss)
        self.log('train_reward_loss', reward_loss)
        self.log('train_tau_loss', tau_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        s, a, next_s, rews, duration = batch.obs, batch.action, batch.next_obs, batch.rewards, batch.duration
        z, next_z, reward_prediction, tau_prediction, q_s, q_next_s = self._step(s, a, next_s)
        # compute loss
        reward_loss = self._reward_loss(reward_prediction, q_s, q_next_s, rews)
        tau_loss = self._tau_loss(tau_prediction, duration)
        loss = reward_loss + tau_loss

        # log
        self.log('val_loss', loss)
        self.log('val_reward_loss', reward_loss)
        self.log('val_tau_loss', tau_loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=1e-5)
