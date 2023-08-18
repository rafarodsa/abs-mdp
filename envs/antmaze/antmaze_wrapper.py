from copy import deepcopy

import numpy as np
import torch
import ipdb

from hrl.wrappers.gc_mdp_wrapper import GoalConditionedMDPWrapper

XY_LIMITS = {
	'antmaze-umaze-v2': {
							'xlims': (-1.1705554723739624, 9.750748634338379), 
							'ylims': (-1.5460281372070312, 9.752226829528809)
		      			},
	'antmaze-medium-play-v2': {
								'xlims': (-1.0799036026000977, 21.75527572631836), 
								'ylims': (-0.621870219707489, 21.74634552001953)
		      				},
}

class D4RLAntMazeWrapper(GoalConditionedMDPWrapper):
	def __init__(self, env, start_state, goal_state, init_truncate=True, use_dense_reward=False):
		self.env = env
		self.init_truncate = init_truncate
		self.norm_func = lambda x: np.linalg.norm(x, axis=-1) if isinstance(x, np.ndarray) else torch.norm(x, dim=-1)
		self.reward_func = self.dense_gc_reward_func if use_dense_reward else self.sparse_gc_reward_func
		# print('preloading')
		# self.observations = self.env.get_dataset()["observations"]
		# print('loading')
		# self.observations = torch.load('data/d4rl/antmaze-umaze-v2/data.pt')['observations']
		# print(self.observations)
		# self._determine_x_y_lims()
		
		self._get_x_y_lims()

		super().__init__(env, start_state, goal_state)

	def state_space_size(self):
		return self.env.observation_space.shape[0]
	
	def action_space_size(self):
		return self.env.action_space.shape[0]
	
	def sparse_gc_reward_func(self, states, goals, batched=False):
		"""
		overwritting sparse gc reward function for antmaze
		"""
		# assert input is np array or torch tensor
		assert isinstance(states, (np.ndarray, torch.Tensor))
		assert isinstance(goals, (np.ndarray, torch.Tensor))

		if batched:
			current_positions = states[:,:2]
			goal_positions = goals[:,:2]
		else:
			current_positions = states[:2]
			goal_positions = goals[:2]
		distances = self.norm_func(current_positions-goal_positions)
		dones = distances <= self.goal_tolerance

		rewards = np.zeros_like(distances)
		rewards[dones==1] = +0.
		rewards[dones==0] = -1.

		return rewards, dones
	
	def dense_gc_reward_func(self, states, goals, batched=False):
		"""
		overwritting dense gc reward function for antmaze
		"""
		assert isinstance(states, (np.ndarray, torch.Tensor))
		assert isinstance(goals, (np.ndarray, torch.Tensor))

		if batched:
			current_positions = states[:,:2]
			goal_positions = goals[:,:2]
		else:
			current_positions = states[:2]
			goal_positions = goals[:2]
		distances = self.norm_func(current_positions - goal_positions)
		dones = distances <= self.goal_tolerance

		# assert distances.shape == dones.shape == (current_positions.shape[0], ) == (goals.shape[0], ), f'{states.shape}, {goals.shape}, {dones.shape}, {distances.shape}'

		rewards = -distances
		if batched:
			rewards[dones==True] = 0
		elif dones:
			rewards = 0

		return rewards, dones
	
	def set_goal(self, goal):
		self.goal_state = goal

	def step(self, action):
		next_state, reward, done, info = self.env.step(action)
		reward, done = self.reward_func(next_state, self.get_current_goal())
		self.cur_state = next_state
		self.cur_done = done
		return next_state, reward, done, info

	def get_current_goal(self):
		return self.get_position(self.goal_state)

	def is_start_region(self, states):
		dist_to_start = self.norm_func(states - self.start_state)
		return dist_to_start <= self.goal_tolerance
	
	def is_goal_region(self, states):
		dist_to_goal = self.norm_func(states - self.goal_state)
		return dist_to_goal <= self.goal_tolerance
	
	def extract_features_for_initiation_classifier(self, states):
		"""
		for antmaze, the features are the x, y coordinates (first 2 dimensions)
		"""
		def _numpy_extract(states):
			if len(states.shape) == 1:
				return states[:2]
			assert len(states.shape) == 2, states.shape
			return states[:, :2]
		
		def _list_extract(states):
			return [state[:2] for state in states]
		
		if self.init_truncate:
			if isinstance(states, np.ndarray):
				return _numpy_extract(states)
			if isinstance(states, list):
				return _list_extract(states)
			raise ValueError(f"{states} of type {type(states)}")
		
		return states
	
	def set_xy(self, position):
		""" Used at test-time only. """
		position = tuple(position)  # `maze_model.py` expects a tuple
		self.env.env.set_xy(position)
		obs = np.concatenate((np.array(position), self.init_state[2:]), axis=0)
		self.cur_state = obs
		self.cur_done = False
		self.init_state = deepcopy(self.cur_state)

	def _get_x_y_lims(self):
		# ipdb.set_trace()
		env_name = f'{self.env.spec.name}-v{self.env.spec.version}'
		assert env_name in XY_LIMITS, env_name
		self.xlims = XY_LIMITS[env_name]['xlims']
		self.ylims = XY_LIMITS[env_name]['ylims']


	def reset(self):
		super().reset()
		s0 = self.sample_random_state()
		self.set_xy(s0)
		return self.cur_state

    # --------------------------------
    # Used for visualizations only
    # --------------------------------

	def _determine_x_y_lims(self):
		observations = self.observations
		x = [obs[0] for obs in observations]
		y = [obs[1] for obs in observations]
		xlow, xhigh = min(x), max(x)
		ylow, yhigh = min(y), max(y)
		self.xlims = (xlow, xhigh)
		self.ylims = (ylow, yhigh)

		print(f"{xlow}, {xhigh}")
		print(f"{ylow}, {yhigh}")

	def get_x_y_low_lims(self):
		return self.xlims[0], self.ylims[0]

	def get_x_y_high_lims(self):
		return self.xlims[1], self.ylims[1]
	
    # ---------------------------------
    # Used during testing only
    # ---------------------------------

	def sample_random_state(self, cond=lambda x: True):
		num_tries = 0
		rejected = True
		while rejected and num_tries < 200:
			low = np.array((self.xlims[0], self.ylims[0]))
			high = np.array((self.xlims[1], self.ylims[1]))
			sampled_point = np.random.uniform(low=low, high=high)
			rejected = self.env.env.wrapped_env._is_in_collision(sampled_point) or not cond(sampled_point)
			num_tries += 1

			if not rejected:
				return sampled_point
	
	@staticmethod
	def get_position(state):
		"""
		position in the antmaze is the x, y coordinates
		"""
		return state[:2]
