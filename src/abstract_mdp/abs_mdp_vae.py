"""
	AbstractMDPVAE
	Implementation based in VAE loss

	author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
	date: January 2023
"""


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl

from src.abstract_mdp.models import encoder_fc, decoder_fc, reward_fc, initiation_classifier, transition_fc_deterministic
from src.abstract_mdp.abs_mdp import AbstractMDP
from src.utils.symlog import symlog, symexp


import numpy as np
import argparse

from collections import defaultdict

### Hyperparameters
training_hyperparams = {
	"prediction_loss_const": 10.,
	"kl_loss_const": 0.1,
	"reward_loss_const": 1.,
	"transition_loss_const": 0.1,
	"init_loss_const": 0.,
	"lr": 0.5e-3,
	"n_samples": 1
}

class AbstractMDPTrainer(pl.LightningModule):
	
	def __init__(self, obs_dim, latent_dim, n_options, encoder_size, decoder_size, transition_size, init_classifier_size, reward_size, **training_hyperparams):
		super().__init__()
		self.save_hyperparameters()

		self.obs_dim = obs_dim
		self.latent_dim = latent_dim
		self.n_options = n_options
		self.encoder = encoder_fc(obs_dim, encoder_size, latent_dim)
		self.transition = transition_fc_deterministic(latent_dim, n_options, transition_size)
		self.decoder = decoder_fc(obs_dim, decoder_size, latent_dim)
		self.reward_fn = reward_fc(latent_dim, reward_size, n_options)
		self.init_classifier = initiation_classifier(latent_dim, init_classifier_size, n_options)
		self.training_hyperparams = training_hyperparams


	def forward(self, state, action):
		z, _, _ = self.encoder.sample(state)
		z_prime, _, _ = self.transition.sample(z, action)
		s_prime, _, _ = self.decoder.sample(z_prime)
		return s_prime
	
	def _run_step(self, s, a, s_prime, executed):
		# number of samples to approximate expectations
		n_samples = self.training_hyperparams['n_samples']

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
		n_samples = self.training_hyperparams['n_samples']
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


		loss = prediction_loss * self.training_hyperparams['prediction_loss_const']\
			+ kl_loss * self.training_hyperparams['kl_loss_const']\
			+ reward_loss * self.training_hyperparams['reward_loss_const'] \
			+ init_classifier_loss * self.training_hyperparams['init_loss_const']\
			+ transition_loss * self.training_hyperparams['transition_loss_const']

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
		s_prime_ = s_prime.unsqueeze(0).repeat_interleave(self.training_hyperparams['n_samples'], dim=0)
		log_probs = q_s_prime_pred.log_prob(s_prime_).sum(-1)

		s_ = s.unsqueeze(0).repeat_interleave(self.training_hyperparams['n_samples'], dim=0)
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
		
		_reward_target = symlog(rewards).transpose(0, 1).unsqueeze(1) # batch x n_samples x 1
		weights = torch.exp((q_s.log_prob(obs) + q_s_prime.log_prob(next_obs)).sum(-1))
		Z = weights.sum(0, keepdims=True)
		reward_target_ = (weights*_reward_target/Z).sum(0).detach()

		# Note that I'm detaching the target. we don't want to backprop to the grounding function
		return torch.nn.functional.mse_loss(reward_pred, reward_target_, reduction="mean")

	def _init_classifier_loss(self, prediction, target):
		target_ = target.transpose(0, 1)
		return torch.nn.functional.binary_cross_entropy(prediction, target_, reduction="mean")

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
		loss, logs = self.step(batch, batch_idx)
		self.log_dict({f"val_{k}": v for k, v in logs.items()})
		return loss

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.training_hyperparams['lr'])

	@staticmethod
	def add_model_specific_args(parent_parser):
		parser = parent_parser.add_argument_group("AbsMDPTrainer")
		parser.add_argument("--obs_dim", type=int, default=4)
		parser.add_argument("--latent_dim", type=int, default=2)
		parser.add_argument("--encoder_size", type=int, default=32)
		parser.add_argument("--decoder_size", type=int, default=32)
		parser.add_argument("--transition_size", type=int, default=32)
		parser.add_argument("--init_classifier_size", type=int, default=32)
		parser.add_argument("--reward_size", type=int, default=32)
		parser.add_argument("--n_options", type=int, default=4)

		for k, v in training_hyperparams.items():
			parser.add_argument(f"--{k}", type=type(v), default=v)

		return parent_parser

def _geometric_return(rewards, gamma):
	_gammas = gamma **  np.arange(rewards.shape[-1])
	return (np.array(rewards) * _gammas[None, :]).sum(-1)

def preprocess_dataset(gamma=0.99, n_actions=4):
	def _transform(datum):
		# s, a, s', r, executed, duration, init_mask
		datum = list(datum)
		datum[-1] = datum[-1].astype(float)
		datum[3] = datum[3][:2] + (_geometric_return(datum[3][-1], gamma),)
		datum[1] = F.one_hot(torch.Tensor([datum[1]]).long(), n_actions).squeeze()
		return datum
	return _transform

		
class PinballDataset(torch.utils.data.Dataset):
	def __init__(self, path_to_file, n_reward_samples=5, transform=None):
		self.data, self.rewards = torch.load(path_to_file)
		self.n_reward_samples = n_reward_samples
		self.transform = transform

	def __getitem__(self, index):
		datum = list(self.data[index])
		rewards = self._get_rewards(datum, n_samples=self.n_reward_samples-1)
		datum[3] = rewards
		datum = self.transform(datum) if self.transform else datum
		return datum

	def _get_rewards(self, datum, n_samples):
		current_obs, action, current_next_obs, current_rewards, _, _, _ = datum
		obs, next_obs, rews = self.rewards[action] # tuple of arrays (obs, next_obs, rewards).
		N = rews.shape[0]
		sample = np.random.choice(N, n_samples, replace=False)
		# append current datum
		obs = np.concatenate([current_obs[np.newaxis, :], obs[sample]], axis=0)
		next_obs = np.concatenate([current_next_obs[None, :], next_obs[sample]], axis=0)
		current_rewards = np.array(current_rewards)[None, :]
		rewards = np.concatenate([current_rewards, rews[sample]], axis=0)
		return (obs, next_obs, rewards)
	
	def __len__(self):
		return len(self.data)

if __name__=="__main__":
	# default params
	latent_dim = 2
	obs_dim = 4
	n_actions = 4
	# data
	dataset_file_path = '/Users/rrs/Desktop/abs-mdp/data/'
	dataset_name = 'pinball_no_obstacle_rewards.pt'

	## Parser
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_path', type=str, default=dataset_file_path+dataset_name)
	parser.add_argument('--num-epochs', type=int, default=10)
	parser.add_argument('--accelerator', type=str, default='cpu')
	parser.add_argument('--batch-size', type=int, default=32)
	parser.add_argument('--save-path', type=str, default='mdps/abs_mdp.pt')

	parser = AbstractMDPTrainer.add_model_specific_args(parser)
	args = parser.parse_args()
	
	# TODO use Lightning data module.
	dataset = PinballDataset(args.dataset_path, transform=preprocess_dataset())
	pinball_test, pinball_val = random_split(dataset, [0.9, 0.1])

	train_loader = DataLoader(pinball_test, batch_size=args.batch_size)
	val_loader = DataLoader(pinball_val, batch_size=args.batch_size)

	# model
	model = AbstractMDPTrainer(**vars(args)).double()
	
	# training
	trainer = pl.Trainer(accelerator=args.accelerator, max_epochs=args.num_epochs)
	trainer.fit(model, train_loader, val_loader)
    
	# Create abstract MDP & save
	abs_mdp = AbstractMDP(model, dataset)
	abs_mdp.save(args.save_path)


	
	
	
