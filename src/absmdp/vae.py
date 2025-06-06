'''
	VAE with calibrated decoders
'''

import torch
import torch.nn.functional as F
import pytorch_lightning as pl


from src.models.factories import build_distribution, build_model

from src.utils.symlog import symlog
from src.absmdp.configs import TrainerConfig
from src.utils.softplus import Softplus

from omegaconf import OmegaConf as oc

import logging

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('medium')

class AbstractMDPTrainer(pl.LightningModule):
	
	def __init__(self, cfg: TrainerConfig):
		super().__init__()
		oc.resolve(cfg)
		self.save_hyperparameters()
		self.obs_dim = cfg.model.obs_dims
		self.latent_dim = cfg.model.latent_dim
		self.encoder = build_model(cfg.model.encoder)
		self.decoder = build_model(cfg.model.decoder)
		self.lr = cfg.lr
		self.hyperparams = cfg.loss
		self.kl_const =  self.hyperparams.kl_const
	
	
	def forward(self, s):
		x = self.encoder(s)
		mean, log_var = x[:, :self.latent_dim], x[:, self.latent_dim:]
		z = mean
		reconstruction = self.decoder(z)
		return z, reconstruction, (mean, log_var)
	
	def _forward(self, s):
		x = self.encoder(s)
		mean, log_var = x[:, :self.latent_dim], x[:, self.latent_dim:]
		z = mean + torch.randn_like(mean) * torch.exp(log_var/2)

		reconstruction = self.decoder(z)
		return z, reconstruction, (mean, log_var)
		
	
	def step(self, batch, batch_idx):
		s, s_prime, executed = batch.obs, batch.next_obs, batch.executed
		assert torch.all(executed) # check all samples are successful executions.

		
		N = s.shape[0] + s_prime.shape[0]
		
		model = 'cvae'
		# loss = (reconstruction_loss + 1. * kl_loss) / N
		# log std deviations for encoder.
		if model == 'ae':
			z, s_bar, q_z = self.forward(s)
			z_prime, s_prime_bar, q_zprime = self.forward(s_prime)
			kl_loss = 0
			reconstruction_loss = (F.mse_loss(s, s_bar, reduction='none') + F.mse_loss(s_prime, s_prime_bar, reduction='none')).sum()
			loss = reconstruction_loss / N
		elif model == 'vae':
			z, s_bar, q_z = self._forward(s) # samples z from q(z|x)
			z_prime, s_prime_bar, q_zprime = self._forward(s_prime)
			kl_loss = self.kl_loss(q_z[0], q_z[1]/2).sum() + self.kl_loss(q_zprime[0], q_zprime[1]/2).sum()
			reconstruction_loss = (F.mse_loss(s, s_bar, reduction='none') + F.mse_loss(s_prime, s_prime_bar, reduction='none')).sum()
			loss = (self.kl_const * kl_loss + reconstruction_loss)/N
		elif model == 'cvae':
			z, s_bar, q_z = self._forward(s)
			z_prime, s_prime_bar, q_zprime = self._forward(s_prime)
			kl_loss = self.kl_loss(q_z[0], q_z[1]/2).sum() + self.kl_loss(q_zprime[0], q_zprime[1]/2).sum()
			reconstruction_loss = self.calibrated_prediction_loss(s, s_bar).sum() + self.calibrated_prediction_loss(s_prime, s_prime_bar).sum()
			loss = (self.kl_const * kl_loss + reconstruction_loss)/N

		
		logs = {
			'loss': loss,
			'reconstruction_loss': reconstruction_loss / N,
			'kl_loss': kl_loss / N,
			'kl_const': self.kl_const
		}

		logger.debug(f'Losses: {logs}')
		return loss, logs

	def training_step(self, batch, batch_idx):
		loss, logs = self.step(batch, batch_idx)
		self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return loss
	
	def validation_step(self, batch, batch_idx):
		s, next_s = batch.obs, batch.next_obs
		x = torch.cat((s, next_s), dim=0)
		z, x_bar, _ = self(x)
		reconstruction_loss =  F.mse_loss(x, x_bar, reduction='none')
		reconstruction_loss =reconstruction_loss.sum(-1).sqrt().mean()
		logs = {
			'val_loss': reconstruction_loss
		}

		self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return reconstruction_loss

	def calibrated_prediction_loss(self, x, mu):
		logvar = self._log_var(mu, x)
		logsigma = self.softclip(logvar, -6)/2
		nll = self.gaussian_nll(mu, logsigma, x).sum(-1)
		return nll

	def _log_var(self, mean, x):
		return ((mean - x) ** 2).mean((0,1), keepdim=True).log()
	
	def softclip(self, x, min):
		return min + Softplus(x - min)

	def gaussian_nll(self, mu, log_sigma, x):
		return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * torch.log(torch.tensor(2 * torch.pi))

	def kl_loss(self, mu, log_sigma):
		kl = .5 * (-2 * log_sigma + (2 * log_sigma).exp() + mu ** 2 - 1).sum(-1)
		return torch.max(torch.ones_like(kl), kl) # free bits.

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.lr)
	
	@staticmethod
	def load_config(path):
		try:
			with open(path, "r") as f:
				cfg = oc.load(f)
				return cfg
		except FileNotFoundError:
			raise ValueError(f"Could not find config file at {path}")
		
