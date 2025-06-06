"""
	AbstractMDPVAE
	Implementation based on VAE loss

	author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
	date: January 2023
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
		self.encoder = build_distribution(cfg.model.encoder)
		self.transition = build_distribution(cfg.model.transition)
		self.decoder = build_distribution(cfg.model.decoder)
		self.reward_fn = build_model(cfg.model.reward)
		self.init_classifier = build_model(cfg.model.init_class)
		self.lr = cfg.lr
		self.hyperparams = cfg.loss
		self.kl_const =  self.hyperparams.kl_const


	
	def forward(self, state, action, executed):
		logger.debug('Forward pass')
		z, q_z, _ = self.encoder.sample_n_dist(state)
		logger.debug(f'Encoder: z {z.shape}')
		z = q_z.mean.squeeze() # squeezing sample dim
		z_prime = self.transition(torch.cat([z, action], dim=-1))
		logger.debug(f'Transition: z {z.shape}, z_prime {z_prime.shape}, action {action.shape}, executed {executed.shape}')
	
		q_s_prime = self.decoder.distribution(z_prime) 
		q_s = self.decoder.distribution(z) 

		reward = self.reward(z, action, z_prime)
		init_class_z = self.init_classifier(z)
		# init_class_z_prime = self.init_classifier(z_prime)
		# init_class = torch.stack([init_class_z, init_class_z_prime])
		init_class = init_class_z
		logger.debug('Forward pass complete')
		return q_s, q_s_prime, reward, init_class
	
	def _run_step(self, s, a, s_prime, executed):
		logger.debug('Training running step')
		# sample encoding of (s, s')
		z, q_z, q_z_std = self.encoder.sample_n_dist(s)
		z_prime, q_z_prime_encoder, _  = self.encoder.sample_n_dist(s_prime)

		logger.debug(f"Encoder: z {z.shape}, z' {z_prime.shape}")

		# predict next latent state
		executed_ = executed.unsqueeze(-1) # dims: (batch, 1)
		actions = a # dims: (batch, n_actions)
		logger.debug(f"Actions: {actions.shape}, executed: {executed_.shape}")
		
		_z = q_z.mean.squeeze() #z.squeeze(0) # z.squeeze sample dimension because we are using one sample.
		z_prime_pred, q_z_prime_pred, _ = self.transition.sample_n_dist(torch.cat([_z, actions], dim=-1))

		logger.debug(f"Transition: z {z.shape}, z' {z_prime_pred.shape}")
		
		# decode next latent state
		# q_s_prime = self.decoder.distribution(z_prime_pred)
		q_s_prime = self.decoder.distribution(z_prime)
		q_s = self.decoder.distribution(z)

		# logger.debug('Training running step finished')
		return q_z, q_z_std, q_z_prime_pred, q_z_prime_encoder, q_s_prime, q_s
		
	
	def step(self, batch, batch_idx):
		s, a, s_prime, _, _, executed, _, _, p0 = batch.obs, batch.action, batch.next_obs, batch.rewards, batch.done, batch.executed, batch.duration, batch.initsets, batch.p0


		assert torch.all(executed) # check all samples are successful executions.
		
		q_z, q_z_std, q_z_prime_pred, q_z_prime_encoded, s_prime_dist, s_dist = self._run_step(s, a, s_prime, executed)

		# compute losses
		# prediction_loss = self._prediction_loss(s_prime, s_prime_dist, s, s_dist, p0)
		prediction_loss = self.calibrated_prediction_loss(s_prime, s_prime_dist, s, s_dist, p0)
		kl_loss = self.kl_loss(q_z, q_z_std, p0) #+ self._init_state_dist_loss(q_z_prime_encoded, q_z_std, p0)
		# transition_loss, nlog_p = self._transition_loss(q_z_prime_encoded, q_z_prime_pred, alpha=self.hyperparams.kl_balance) 
		transition_loss, nlog_p = self.transition_kl(q_z_prime_encoded, q_z_prime_pred, alpha=self.hyperparams.kl_balance), 0

		# compute total loss
		
		loss = prediction_loss * self.hyperparams.grounding_const\
			+ kl_loss * self.kl_const\
			+ transition_loss * self.kl_const \
			+ nlog_p * self.hyperparams.transition_const 
		
		loss = loss.mean()
		elbo = -(prediction_loss + kl_loss + transition_loss).mean()

		# log std deviations for encoder.

		logs = {
			"grounding_loss": prediction_loss.mean(),
			"kl_loss": kl_loss.mean(),
			"prediction_loss": transition_loss.mean(),
			"kl_const": self.kl_const,
			"elbo": elbo,
			"transition_loss": nlog_p,
			"loss": loss,
			"std_s": s_dist.mean.squeeze().std(0).mean().item(),
			"std_next_s": s_prime_dist.mean.squeeze().std(0).mean().item(),
		}

		logger.debug(f'Losses: {logs}')
		return loss, logs

	def _prediction_loss(self, s_prime, q_s_prime_pred, s, q_s_pred, p0):
		n_samples = self.hyperparams.n_samples
		s_prime_ = s_prime.unsqueeze(0).repeat_interleave(n_samples, dim=0)
		log_probs = q_s_prime_pred.log_prob(s_prime_)

		s_ = s.unsqueeze(0).repeat_interleave(n_samples, dim=0)
		log_probs_s = q_s_pred.log_prob(s_) #* p0

		return -(log_probs + log_probs_s)


	def calibrated_prediction_loss(self, s_prime, q_s_prime_pred, s, q_s_pred, p0):
		log_var_s_prime = self._log_var(q_s_prime_pred.mean.squeeze(), s_prime)
		nlog_prob_s_prime = self.gaussian_nll(q_s_prime_pred.mean.squeeze(), log_var_s_prime / 2, s_prime).sum(-1)

		log_var_s = self._log_var(q_s_pred.mean.squeeze(), s)
		n_log_probs = self.gaussian_nll(q_s_pred.mean.squeeze(), log_var_s / 2, s).sum(-1)
		return n_log_probs + nlog_prob_s_prime

	def _log_var(self, mean, x):
		return ((mean - x) ** 2).mean((0,1), keepdim=True).log()
	
	def softclip(self, x, min):
		return min + F.softplus(x - min)

	def gaussian_nll(self, mu, log_sigma, x):
		return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * torch.log(torch.tensor(2 * torch.pi))

	def transition_kl(self, encoder_dist, transition_dist, alpha=0.01):
	
		kl_1 = torch.distributions.kl_divergence(encoder_dist, transition_dist.detach())
		kl_2 = torch.distributions.kl_divergence(encoder_dist.detach(), transition_dist)
		kl = alpha * kl_1 + (1-alpha) * kl_2 # KL balancing
		return torch.max(torch.ones_like(kl), kl) # free bits


		# h = encoder_dist.entropy().mean()
		
		# mse = F.mse_loss(transition_dist.mean.detach(), encoder_dist.mean).mean()

		# nlog_p = -encoder_dist.detach().log_prob(transition_dist.mean)
		# return -h+mse, nlog_p.mean()
		
	def kl_loss(self, q_z, std_normal_z, p0) -> torch.Tensor:
		'''
			KL regularization. We use free bits to ensure that the KL is at least 1.
			TODO: modify this to learn the initial state distribution
		'''
		kl = torch.distributions.kl_divergence(q_z, std_normal_z)
		return torch.max(torch.ones_like(kl), kl) # free bits

	def training_step(self, batch, batch_idx):
		loss, logs = self.step(batch, batch_idx)
		self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
		return loss

	def validation_step(self, batch, batch_idx):
		logger.debug('Validation step')
		s, a, s_prime, reward, executed, duration, initiation_target = batch.obs, batch.action, batch.next_obs, batch.rewards, batch.executed, batch.duration, batch.initsets
		qs, q_s_prime, reward_pred, initiation = self.forward(s, a, executed)
		logger.debug(f'Batch: s {s.shape}, a {a.shape}, s_prime {s_prime.shape}, reward {len(reward)}, executed {executed.shape}, duration {duration.shape}, initiation_target {initiation_target.shape}')
		
		nll_loss = -q_s_prime.log_prob(s_prime).mean()
		mse_error = F.mse_loss(s_prime, q_s_prime.mean.squeeze(), reduction='sum') / s_prime.shape[0]

		init_loss = self._init_classifier_loss(initiation, initiation_target)
		loss = nll_loss + init_loss
		self.log_dict({'nll_loss': nll_loss,
					   'init_loss': init_loss})
		self.log_dict({'mse_error': mse_error}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return nll_loss
		
	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

	def reward(self, z, a, z_prime):
		n_samples = self.hyperparams.n_samples
		# create batch
		if n_samples > 1:
			#TODO: check this is correct.
			z_ = z.unsqueeze(0).repeat_interleave(n_samples, dim=0)
			a_ = a.unsqueeze(0).repeat_interleave(n_samples, dim=0)
		else:
			z_, a_ = z, a
		input = torch.cat([z_, a_, z_prime], dim=-1)
		return self.reward_fn(input).squeeze()

	def _reward_loss(self, reward_pred, q_s, q_s_prime, reward_target):
		logger.debug('Reward loss')
		obs, next_obs, rewards = reward_target  # batch x n_samples x obs_dim
		# TODO: why is this unsqueezed dimension necessary?
		# I this is for the number of sampled predictions for z and z_prime

		# q(s|z) should have B x n_z_samples distributions
		# q(s'|z') should have B x n_z_samples x n_z_prime_samples distributions
		obs, next_obs = obs.transpose(0, 1).unsqueeze(1), next_obs.transpose(0, 1).unsqueeze(1)  # n_samples x batch x obs_dim
		logger.debug(f'obs: {obs.shape}, next_obs: {next_obs.shape}')
		
		_reward_target = symlog(rewards).transpose(0, 1).unsqueeze(1) # n_samples x 1 x batch
		logger.debug(f'reward_target: {_reward_target.shape}')
		
		with torch.no_grad():
			weights = torch.exp((q_s.log_prob(obs) + q_s_prime.log_prob(next_obs)))
			Z = weights.sum(0, keepdims=True)
			logger.debug(f'Reward weights: {Z}')
			logger.debug(f'weights: {weights.shape}, Z: {Z.shape}')
			reward_target_ = (weights*_reward_target/Z).sum(0).squeeze()

		logger.debug(f'reward_target_: {reward_target_.shape}, reward_pred: {reward_pred.shape}')
		logger.debug('Reward loss done')
		# Note that I'm detaching the target. we don't want to backprop to the grounding function
		return torch.nn.functional.mse_loss(reward_pred, reward_target_, reduction="mean")

	def _init_classifier_loss(self, prediction, target):
		# target_ = target.transpose(0, 1)
		target_ = target
		return torch.nn.functional.binary_cross_entropy_with_logits(prediction, target_, reduction="mean")

	
	@staticmethod
	def load_config(path):
		try:
			with open(path, "r") as f:
				cfg = oc.merge(oc.structured(TrainerConfig), oc.load(f))
				return cfg
		except FileNotFoundError:
			raise ValueError(f"Could not find config file at {path}")
		
