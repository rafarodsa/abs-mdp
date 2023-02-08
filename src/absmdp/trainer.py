"""
	AbstractMDPVAE
	Implementation based in VAE loss

	author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
	date: January 2023
"""
import torch
import pytorch_lightning as pl

from src.models.factories import build_distribution, build_model

from src.utils.symlog import symlog
from src.absmdp.configs import TrainerConfig

from omegaconf import OmegaConf as oc

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
		self.training_hyperparams = cfg.loss

	
	def forward(self, state, action, executed):
		
		z, q_z, _, _ = self.encoder.sample(state)
		z = z.squeeze()
		z_prime = self.transition(torch.cat([q_z.mean, action, executed.unsqueeze(-1)], dim=-1))
		
		# TODO: substitute with distribution
		_, q_s_prime, _, _ = self.decoder.sample(z_prime) 
		_, q_s, _, _ = self.decoder.sample(z) 

		reward = self.reward(z, action, z_prime)
		init_class_z = self.init_classifier(z)
		init_class_z_prime = self.init_classifier(z_prime)
		return q_s, q_s_prime, reward, torch.stack([init_class_z, init_class_z_prime])
	
	def _run_step(self, s, a, s_prime, executed):
		# number of samples to approximate expectations
		n_samples = self.training_hyperparams.n_samples

		# sample encoding of (s, s')
		z, q_z, q_z_std, _ = self.encoder.sample(s)
		z_prime, q_z_prime_encoder, _, _  = self.encoder.sample(s_prime)

		# predict next latent state
		executed_ = executed.unsqueeze(-1)
		actions = a
		
		# z.squeeze sample dimension because we are using one sample.
		z = z.squeeze(0)
		z_prime_pred = self.transition(torch.cat([z, actions, executed_], dim=-1)).squeeze()
		
		# gaussian distribution with constant std dev for deterministic dynamics
		std_dev = 1
		q_z_prime_pred = torch.distributions.Normal(z_prime_pred, std_dev*torch.ones_like(z_prime_pred))
		
		# TODO: subs with .distribution rather than .sample
		# decode next latent state
		_, q_s_prime, _, _ = self.decoder.sample(z_prime_pred, n_samples)

		# reward
		reward_pred = self.reward(z, a, z_prime_pred)
		_, q_s, _, _ = self.decoder.sample(z)

		# initiation classifier
		init_mask_z = self.init_classifier(z)
		init_mask_z_prime = self.init_classifier(z_prime_pred)
		init_masks = torch.stack([init_mask_z, init_mask_z_prime], dim=0)
		
		return q_z, q_z_std, q_z_prime_pred, q_z_prime_encoder, q_s_prime, reward_pred, init_masks, q_s
		
	def reward(self, z, a, z_prime):
		n_samples = self.training_hyperparams.n_samples
		# create batch
		if n_samples > 1:
			z_ = z.unsqueeze(0).repeat_interleave(n_samples, dim=0)
			a_ = a.unsqueeze(0).repeat_interleave(n_samples, dim=0)
		else:
			z_, a_ = z, a
		input = torch.cat([z_, a_, z_prime], dim=-1)
		return self.reward_fn(input).squeeze()
	
	def step(self, batch, batch_idx):
		s, a, s_prime, reward, executed, duration, initiation_target = batch
		
		q_z, q_z_std, q_z_prime_pred, q_z_prime_encoded, s_prime_dist, reward_pred, initiation_pred, s_dist = self._run_step(s, a, s_prime, executed)

		# compute losses
		prediction_loss = self._prediction_loss(s_prime, s_prime_dist, s, s_dist) 
		kl_loss = self._init_state_dist_loss(q_z, q_z_std) 
		reward_loss = self._reward_loss(reward_pred, s_dist, s_prime_dist, reward) 
		init_classifier_loss = self._init_classifier_loss(initiation_pred, initiation_target) 
		transition_loss = self._transition_loss(q_z_prime_encoded, q_z_prime_pred) 


		loss = prediction_loss * self.training_hyperparams.grounding_const\
			+ kl_loss * self.training_hyperparams.kl_const\
			+ reward_loss * self.training_hyperparams.reward_const \
			+ init_classifier_loss * self.training_hyperparams.init_class_const\
			+ transition_loss * self.training_hyperparams.transition_const

		logs = {
			"grounding_loss": prediction_loss,
			"kl_loss": kl_loss,
			"reward_loss": reward_loss,
			"initiation_loss": init_classifier_loss,
			"prediction_loss": transition_loss,
			"loss": loss
		}
		return loss, logs

	def _prediction_loss(self, s_prime, q_s_prime_pred, s, q_s_pred):
		n_samples = self.training_hyperparams.n_samples
		s_prime_ = s_prime.unsqueeze(0).repeat_interleave(n_samples, dim=0)
		log_probs = q_s_prime_pred.log_prob(s_prime_).sum(-1)

		s_ = s.unsqueeze(0).repeat_interleave(n_samples, dim=0)
		log_probs_s = q_s_pred.log_prob(s_).sum(-1)

		return -(log_probs + log_probs_s).mean()
		
	def _init_state_dist_loss(self, q_z, std_normal_z) -> torch.Tensor:
		'''
			KL regularization. We use free bits to ensure that the KL is at least 1.
			TODO: modify this to learn the initial state distribution
		'''
		kl = torch.distributions.kl_divergence(q_z, std_normal_z).sum(-1).mean()
		return torch.max(torch.tensor(1.), kl)

	def _reward_loss(self, reward_pred, q_s, q_s_prime, reward_target):
		obs, next_obs, rewards = reward_target  # batch x n_samples x obs_dim
		obs, next_obs = obs.transpose(0, 1).unsqueeze(1), next_obs.transpose(0, 1).unsqueeze(1)  # n_samples x batch x obs_dim
		
		_reward_target = symlog(rewards).transpose(0, 1).unsqueeze(1) # n_samples x 1 x batch
		weights = torch.exp((q_s.log_prob(obs) + q_s_prime.log_prob(next_obs)).sum(-1))
		Z = weights.sum(0, keepdims=True)
		reward_target_ = (weights*_reward_target/Z).sum(0).squeeze().detach()

		# Note that I'm detaching the target. we don't want to backprop to the grounding function
		return torch.nn.functional.mse_loss(reward_pred, reward_target_, reduction="mean")

	def _init_classifier_loss(self, prediction, target):
		target_ = target.transpose(0, 1)
		return torch.nn.functional.binary_cross_entropy_with_logits(prediction, target_, reduction="mean")

	def _transition_loss(self, encoder_dist, transition_dist, alpha=0.01):

		# if we're using normal distributions
		transition_mean, transition_std = transition_dist.mean.detach(), transition_dist.stddev.detach()
		encoder_mean, encoder_std = encoder_dist.mean.detach(), encoder_dist.stddev.detach()
		
		no_grad_transition_dist = torch.distributions.Normal(transition_mean, transition_std)
		no_grad_encoder_dist = torch.distributions.Normal(encoder_mean, encoder_std)
		kl_1 = torch.distributions.kl_divergence(encoder_dist, no_grad_transition_dist).sum(-1)
		kl_2 = torch.distributions.kl_divergence(no_grad_encoder_dist, transition_dist).sum(-1)
		
		# KL balancing
		return (alpha*kl_1 + (1-alpha)*kl_2).mean()
	
	def training_step(self, batch, batch_idx):
		loss, logs = self.step(batch, batch_idx)
		self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
		return loss

	def validation_step(self, batch, batch_idx):
		s, a, s_prime, reward, executed, duration, initiation_target = batch
		qs, q_s_prime, reward_pred, initiation = self.forward(s, a, executed)
		nll_loss = -q_s_prime.log_prob(s_prime).sum(-1).mean()
		init_loss = self._init_classifier_loss(initiation, initiation_target)
		rew_loss = self._reward_loss(reward_pred, qs, q_s_prime, reward)
		loss = nll_loss + init_loss + rew_loss
		self.log_dict({'val_loss': loss,
					   'nll_loss': nll_loss,
					   'init_loss': init_loss,
					   'rew_loss': rew_loss})
		return loss
		
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
		
