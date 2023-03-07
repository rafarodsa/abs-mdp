"""
	Option Model (Abstract MDPs)
	based on the information bottleneck principle
	with gaussian encoder.

	author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
	date: 27 February 2023
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from src.utils.printarr import printarr


from src.models.factories import build_distribution, build_model

from src.utils.symlog import symlog
from src.absmdp.configs import TrainerConfig

from omegaconf import OmegaConf as oc

import logging

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('medium')

class AbstractMDPTrainer(pl.LightningModule):
	
	def __init__(self, cfg: TrainerConfig):
		super().__init__()
		oc.resolve(cfg)
		self.save_hyperparameters()
		self.data_cfg = cfg.data
		self.obs_dim = cfg.model.obs_dims
		self.latent_dim = cfg.model.latent_dim
		self.n_options = cfg.model.n_options
		self.encoder = build_distribution(cfg.model.encoder)
		self.transition = build_distribution(cfg.model.transition)
		self.decoder = build_distribution(cfg.model.decoder)
		self.lr = cfg.lr
		self.hyperparams = cfg.loss
		self.kl_const =  self.hyperparams.kl_const
	
	def forward(self, state, action):
		z, _ = self.encoder(state)
		next_z, next_z_q, _ = self.transition.sample_n_dist(torch.cat([z, action], dim=-1))	
		q_s_prime = self.decoder.distribution(next_z) 
		q_s = self.decoder.distribution(z) 
		return q_s, q_s_prime

	def _run_step(self, s, a, next_s):
		# sample encoding of (s, s')
		n_samples = 2
		z, q_z, z_std = self.encoder.sample_n_dist(s, n_samples=n_samples)
		next_z, q_next_z, _  = self.encoder.sample_n_dist(next_s)
		next_z = next_z.squeeze()

		kl_loss = self.kl_loss(q_z, z_std)

		actions = a #.unsqueeze(0).repeat_interleave(2, dim=0) # dims: (batch, n_actions)

		next_z_pred, next_z_q_pred, _ = self.transition.sample_n_dist(torch.cat([z[0], actions], dim=-1)) # 1 x n_samples=2 x batch x latent_dim

		transition_loss = self.transition_kl(q_next_z, next_z_q_pred)

		# decode next latent state
		next_z_pred, next_z_q_pred, _ = self.transition.sample_n_dist(torch.cat([z[1], actions], dim=-1))
		q_next_s = self.decoder.distribution(next_z)
		q_s = self.decoder.distribution(z)

		prediction_loss = self.prediction_loss(next_s, q_next_s, s, q_s)

		# logger.debug('Training running step finished')
		return prediction_loss, kl_loss, transition_loss
		
	
	def step(self, batch, batch_idx):
		s, a, next_s, executed = batch.obs, batch.action, batch.next_obs, batch.executed
		assert torch.all(executed) # check all samples are successful executions.
		
		prediction_loss, kl_loss, transition_loss = self._run_step(s, a, next_s)

		# compute total loss
		
		loss = prediction_loss * self.hyperparams.grounding_const\
			+ kl_loss * self.kl_const\
			+ transition_loss * self.hyperparams.transition_const 
		
		loss = loss.mean()

		# log std deviations for encoder.

		logs = {
			"prediction_loss (grounding function)": prediction_loss.mean(),
			"kl_loss": kl_loss.mean(),
			"transition_loss": transition_loss.mean(),
			"loss": loss
		}

		logger.debug(f'Losses: {logs}')
		return loss, logs

	def prediction_loss(self, next_s, q_next_s, s, q_s):
		log_probs = q_next_s.log_prob(next_s)
		# log_probs_s = q_s.log_prob(s) # reconstruction of s might not be needed.

		return -log_probs 

	def transition_kl(self, q_next_z, transition_dist):
		return 0*torch.distributions.kl_divergence(q_next_z, transition_dist.detach()) + torch.distributions.kl_divergence(transition_dist, q_next_z.detach())
		
	def kl_loss(self, q_z, p_z) -> torch.Tensor:
		'''
			KL regularization. We use free bits to ensure that the KL is at least 1.
			TODO: modify this to learn the initial state distribution
		'''
		return torch.distributions.kl_divergence(q_z, p_z)

	def training_step(self, batch, batch_idx):
		loss, logs = self.step(batch, batch_idx)
		self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
		return loss

	def validation_step(self, batch, batch_idx):
		s, a, next_s = batch.obs, batch.action, batch.next_obs
		qs, q_s_prime = self.forward(s, a)
		
		nll_loss = -q_s_prime.log_prob(next_s).mean()
		mse_error = F.mse_loss(next_s, q_s_prime.mean.squeeze(), reduction='sum') / next_s.shape[0]

		loss = nll_loss
		self.log_dict({'nll_loss': nll_loss},on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log_dict({'mse_error': mse_error}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return nll_loss
		
	def configure_optimizers(self):
		return torch.optim.NAdam(self.parameters(), lr=self.lr)
	
	@staticmethod
	def load_config(path):
		try:
			with open(path, "r") as f:
				cfg = oc.merge(oc.structured(TrainerConfig), oc.load(f))
				return cfg
		except FileNotFoundError:
			raise ValueError(f"Could not find config file at {path}")
		
