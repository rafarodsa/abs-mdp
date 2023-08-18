
import torch
import lightning as pl
from experiments.antmaze.utils import Initset, TD3, load_td3_agent
import argparse

from src.options.option import Option
from envs.env_options import EnvOptionWrapper
from functools import partial
import envs.antmaze as antmaze
import numpy as np
import gym

GOAL_TOAL = 0.5
DISTANCE = 1.
N_OPTIONS = 8
DIRECTIONS = [[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.], [-1., -1.], [1., -1.], [-1., 1.]]
OPTION_NAMES = ['right', 'up', 'left', 'down', 'up-right', 'down-left', 'up-left', 'down-right']

INITSET_THRESH = 0.5

ANTMAZE_OPTION_MODELS = {
    'antmaze-umaze-v2': {
        'initset_path': 'experiments/antmaze/antmaze-umaze-v2/initset/ckpt/last.ckpt',
        'policy_path': 'experiments/antmaze/antmaze-umaze-v2/policy/policy.pkl',
        'goal_tol': GOAL_TOAL,
        'distance': DISTANCE,
        'directions': DIRECTIONS,
        'names': OPTION_NAMES,
    },
    'antmaze-medium-play-v2': {
        'initset_path': 'experiments/antmaze/antmaze-medium-play-v2/initset/ckpt/last.ckpt',
        'policy_path': 'experiments/antmaze/antmaze-medium-play-v2/policy/policy.pkl',
        'goal_tol': GOAL_TOAL,
        'distance': DISTANCE,
        'directions': DIRECTIONS,
        'names': OPTION_NAMES,
    }
}

def load_policy(policy_path, device='cpu'):
    GOAL_SIZE = 2
    OBS_SIZE = 29
    ACTION_SIZE = 8
    agent = TD3(OBS_SIZE + GOAL_SIZE,
        ACTION_SIZE,
        max_action=1.,
        use_output_normalization=False,
        device=device,
        store_extra_info=False
    )
    load_td3_agent(agent, policy_path, map_location=device)
    return agent

def make_initiation_set(initset, option_name, device='cpu'):
    def initiation_set(s):
        '''
            s : torch.Tensor of shape (batch_size, state_size)
        '''
        return torch.sigmoid(initset(option_name, s)) > INITSET_THRESH
    return initiation_set

def make_termination_probs(dir, distance, goal_tol, device='cpu'):
    displacement = dir * distance
    def termination_probs(s, s0):
        goal = s0 + displacement
        '''
            s : torch.Tensor of shape (batch_size, state_size)
        '''
        if len(s.shape) == 1:
            return torch.norm(s[:2] - goal) < goal_tol
        return torch.norm(s[:, :2] - goal[None], dim=1) < goal_tol
    return lambda s0: partial(termination_probs, s0=s0)

def make_option_policy(policy, direction, distance):
    displacement = direction * distance
    def option_policy(s):
        '''
            s : torch.Tensor of shape (batch_size, state_size)
        '''
        if len(s.shape) == 1:
            goal = s[:2] + displacement
            return policy.act(torch.cat([s, goal]))
        goal = s[:, :2] + displacement[None]
        return policy.act(torch.cat([s[:, :2], goal], dim=1))
    return option_policy


def create_antmaze_options(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='antmaze-umaze-v2')
        parser.add_argument('--save-path', type=str, default=None)
        parser.add_argument('--device', type=int, default=-1)
        args = parser.parse_args()

    configs = ANTMAZE_OPTION_MODELS[args.env]
    initset = Initset.load_from_checkpoint(configs['initset_path'])
    policy = load_policy(configs['policy_path'])

    options = []
    for dir, option_name in zip(configs['directions'], configs['names']):
        o = Option(
            initiation_classifier=make_initiation_set(initset, option_name),
            termination_classifier=make_termination_probs(dir, configs['distance'], configs['goal_tol']),
            policy=make_option_policy(policy, dir, configs['distance']),
        )
        options.append(o)
    
    if args.save_path is not None:
        torch.save(options, args.save_path)
    return options


class CastObservationToTensor(gym.ObservationWrapper):
    """Cast observations to a given type.

    Args:
        env: Env to wrap.
        dtype: Data type object.

    Attributes:
        original_observation: Observation before casting.
    """

    def observation(self, observation):
        self.original_observation = observation
        return torch.from_numpy(observation)

def test_options():

    '''
        Load environment
        Load options
        Wrap environment
        Test options
    '''
    env = antmaze.make_env('antmaze-umaze-v2',
            start=np.array([8., 0.]),
            goal=np.array([0., 0.]),
            seed=0,
            dense_reward=False)
    env = CastObservationToTensor(env)
    options = create_antmaze_options()
    env = EnvOptionWrapper(env, options)

    # run for 1000 steps
    N = 1000
    s = env.reset()
    for i in range(N):
        a = env.action_space.sample()
        s, r, d, _ = env.step(a)
        if d:
            s = env.reset()


if __name__=='__main__':
    create_antmaze_options()