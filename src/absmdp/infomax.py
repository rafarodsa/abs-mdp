"""
	Option Model (Abstract MDPs)
	based on the information bottleneck principle
	with deterministic encoder.

	author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
	date: 1 March 2023
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


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
		self.encoder = build_model(cfg.model.encoder.features)
		self.q = build_distribution(cfg.model.inference)
		self.transition = build_distribution(cfg.model.transition)
		self.grounding = build_distribution(cfg.model.decoder)
		self.decoder = build_distribution(cfg.model.decoder)
		self.transition_s = build_distribution(cfg.model.transition_s)
		self.lr = cfg.lr
		self.hyperparams = cfg.loss
		self.kl_const =  self.hyperparams.kl_const

	def forward(self, state, action):
		z = self.encoder(state)
		t_in = torch.cat([z, action], dim=-1)
		next_z, next_z_q, _ = self.transition.sample_n_dist(t_in)	
		# q_s_prime = self.transition_s.distribution(t_in) 
		q_s_prime = self.grounding.distribution(self.transition.distribution(t_in).mean + z)
		q_s = self.decoder.distribution(z) 
		return q_s, q_s_prime

	def _run_step(self, s, a, next_s):
		# sample encoding of (s, s')
		z = self.encoder(s)
		next_z  = self.encoder(next_s)

		actions = a # dims: (batch, n_actions)
	
		inferred_z, q_z, _ = self.q.sample_n_dist(torch.cat([next_s, actions], dim=-1))

		inferred_z = z #torch.randn(z.shape).to(z.get_device()) * q_z.std + z 

		
		# next_s_q_pred  = self.transition_s.distribution(torch.cat([inferred_z.squeeze(), actions], dim=-1))
		next_z_pred, _, _ = self.transition.sample_n_dist(torch.cat([inferred_z.squeeze(), actions], dim=-1))
		next_z_pred = next_z_pred.squeeze() + z
		next_s_q_pred = self.grounding.distribution(next_z_pred.squeeze())
		infomax_loss = self.infomax_loss(next_s, next_s_q_pred, n_samples=1)
		
		# q_loss = 0 * self.inference_loss(z, q_z)
		transition_loss = self.transition_loss(z.detach(), next_z.detach(), a)

		# decode next latent state
		# q_next_s = self.decoder.distribution(next_z)
		# q_s = self.decoder.distribution(z)
		self.decoder.load_state_dict(self.grounding.state_dict())
		q_s = self.decoder.freeze().distribution(z)
		info_loss_z = -self.info_bottleneck(s, q_s) 

		return infomax_loss + transition_loss, info_loss_z


	def step(self, batch, batch_idx):
		s, a, next_s, executed = batch.obs, batch.action, batch.next_obs, batch.executed
		assert torch.all(executed) # check all samples are successful executions.
		
		infomax, info_loss_z = self._run_step(s, a, next_s)


		# compute total loss
		loss = infomax + self.kl_const * info_loss_z
		loss = loss.mean()

		# log std deviations for encoder.

		logs = {
			'infomax': infomax.mean(),
			'info_loss_z': info_loss_z.mean(),
		}

		logger.debug(f'Losses: {logs}')
		return loss, logs

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
		mse_error = F.mse_loss(next_s, q_s_prime.mean.squeeze(), reduction='sum') / next_s.shape[0]

		loss = nll_loss
		self.log_dict({'nll_loss': nll_loss},on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log_dict({'mse_error': mse_error}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return nll_loss
		
	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.lr)
	
	@staticmethod
	def load_config(path):
		try:
			with open(path, "r") as f:
				cfg = oc.merge(oc.structured(TrainerConfig), oc.load(f))
				return cfg
		except FileNotFoundError:
			raise ValueError(f"Could not find config file at {path}")
		
