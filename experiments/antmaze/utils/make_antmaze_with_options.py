
import torch
import lightning as pl
from experiments.antmaze.utils import Initset, TD3, load_td3_agent
import argparse

from src.options.option import Option
from envs.env_options import EnvOptionWrapper, EnvInitsetWrapper
from functools import partial
import envs.antmaze as antmaze
import numpy as np
import gym
from tqdm import tqdm

import matplotlib.pyplot as plt
from src.utils import printarr

GOAL_TOAL = 0.5
DISTANCE = 1.
N_OPTIONS = 8
DIRECTIONS = [[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.], [-1., -1.], [1., -1.], [-1., 1.]]
OPTION_NAMES = ['right', 'up', 'left', 'down', 'up-right', 'down-left', 'up-left', 'down-right']
MAX_OPTION_EXECUTING_TIME=100
INITSET_THRESH = 0.5

ANTMAZE_OPTION_MODELS = {
    'antmaze-umaze-v2': {
        'initset_path': 'exp_results/antmaze/antmaze-umaze-v2/initset/ckpt/best-v1.ckpt',
        'policy_path': 'experiments/antmaze/antmaze-umaze-v2/policy/td3',
        'goal_tol': GOAL_TOAL,
        'distance': DISTANCE,
        'directions': DIRECTIONS,
        'names': OPTION_NAMES,
    },
    'antmaze-medium-play-v2': {
        'initset_path': 'exp_results/antmaze/antmaze-medium-play-v2/initset/ckpt/best-v1.ckpt',
        'policy_path': 'experiments/antmaze/antmaze-medium-play-v2/policy/td3',
        'goal_tol': GOAL_TOAL,
        'distance': DISTANCE,
        'directions': DIRECTIONS,
        'names': OPTION_NAMES,
    }
}


class CastObservationToFloat32(gym.ObservationWrapper):
    """Cast observations to a given type.

    Args:
        env: Env to wrap.
        dtype: Data type object.

    Attributes:
        original_observation: Observation before casting.
    """

    def observation(self, observation):
        self.original_observation = observation
        return observation.astype(np.float32)
    
    def reset(self, state=None):
        observation = self.env.reset(state)
        return self.observation(observation)
    
class TruncatedWrapper(gym.Wrapper):
    def step(self, action):
        next_s, r, done, info = self.env.step(action)
        return next_s, r, done, False, info
    def reset(self, state=None):
        return self.env.reset(state)


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
            s : np.ndarray of shape (batch_size, state_size)
        '''
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).to(device)
        return (torch.sigmoid(initset(s, option_name)) > INITSET_THRESH).float()
    return initiation_set

def make_termination_probs(direction, distance, goal_tol, device='cpu'):
    displacement = np.array(direction).astype(np.float32) * distance
    def termination_probs(s, s0):
        '''
            s : np.ndarray of shape (batch_size, state_size)
        '''
        goal = s0[..., :2] + displacement
        if len(s.shape) == 1:
            return (np.linalg.norm(s[:2] - goal) < goal_tol).astype(np.float32)
        return (np.linalg.norm(s[:, :2] - goal[None], dim=1) < goal_tol).astype(np.float32)
    return lambda s0: partial(termination_probs, s0=s0)

def make_option_policy(policy, direction, distance, device='cpu'):
    displacement = torch.Tensor(direction).float().to(device) * distance
    def option_policy(s, s0):
        '''
            s : np.ndarray of shape (batch_size, state_size)
        '''
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).to(device)
        if isinstance(s0, np.ndarray):
            s0 = torch.from_numpy(s0).to(device)
        if len(s.shape) == 1:
            
            goal = s0[:2] + displacement
            return policy.act(torch.cat([s, goal]))
        goal = s0[:, :2] + displacement[None]
        return policy.act(torch.cat([s[:, :2], goal], dim=1))
    return lambda s0 : partial(option_policy, s0=s0)


def create_antmaze_options(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='antmaze-umaze-v2')
        parser.add_argument('--save-path', type=str, default=None)
        parser.add_argument('--device', type=int, default=-1)
        parser.add_argument('--max-exec-time', type=int, default=None)
        args = parser.parse_args()

    configs = ANTMAZE_OPTION_MODELS[args.env]
    initset = Initset.load_from_checkpoint(configs['initset_path'])
    policy = load_policy(configs['policy_path'])
    device = f'cuda:{args.device}' if args.device >= 0 else 'cpu'
    options = []
    for dir, option_name in zip(configs['directions'], configs['names']):
        o = Option(
            initiation_classifier=make_initiation_set(initset, option_name, device=device),
            termination_prob=make_termination_probs(dir, configs['distance'], configs['goal_tol'], device=device),
            policy_func_factory=make_option_policy(policy, dir, configs['distance'], device=device),
            max_executing_time=MAX_OPTION_EXECUTING_TIME if args.max_exec_time is None else args.max_exec_time,
            name=option_name
        )
        options.append(o)
    
    # if args.save_path is not None:
    #     torch.save(options, args.save_path)
    return options, initset

def make_antmaze_options(envname, device, check_can_execute=True):
    configs = ANTMAZE_OPTION_MODELS[envname]
    initset = Initset.load_from_checkpoint(configs['initset_path']).to(device)
    policy = load_policy(configs['policy_path'], device=device)
    options = []
    for dir, option_name in zip(configs['directions'], configs['names']):
        o = Option(
            initiation_classifier=make_initiation_set(initset, option_name, device=device),
            termination_prob=make_termination_probs(dir, configs['distance'], configs['goal_tol'], device=device),
            policy_func_factory=make_option_policy(policy, dir, configs['distance'], device=device),
            max_executing_time=MAX_OPTION_EXECUTING_TIME,
            name=option_name, 
            check_can_execute=check_can_execute
        )
        options.append(o)

    return options, initset


def make_antmaze_with_options(envname, seed=0, options=None, initset=None, device='cpu'):
    env = make_antmaze(envname, seed)
    env = EnvOptionWrapper(options, env)
    env = EnvInitsetWrapper(env, make_initiation_set(initset, option_name=None, device=device))
    return env

def make_antmaze(envname, seed=None):
    env = antmaze.make_env(
                envname,
                start=np.array([-10, -9]),
                goal=np.array([-10., -10.]),
                seed=seed
            )
    env = CastObservationToFloat32(env)
    env = TruncatedWrapper(env)
    return env


def plot_initset_sinks(states, initset):
    init = initset(states)
    plt.figure()
    plt.scatter(states[:, 0], states[:, 1], c=init.sum(-1) > 0, s=3)
    plt.colorbar()
    plt.savefig('sink.png')
    print(f'prob of action available: {(init.sum(-1) > 0).float().mean()}')
    print(f'prob of action unavailable: {(init.sum(-1) == 0).float().mean()}')
    print(f'prob only one action avail {((init.sum(-1) == 1)).float().mean()}')


def test_options():

    '''
        Load environment
        Load options
        Wrap environment
        Test options
    '''
    # env = antmaze.make_env('antmaze-umaze-v2',
    #         start=np.array([8., 0.]),
    #         goal=np.array([0., 0.]),
    #         seed=0)
    # options, initset = create_antmaze_options()
    # env = CastObservationToFloat32(env)
    # env = TruncatedWrapper(env)

    env = make_antmaze('antmaze-umaze-v2', seed=0)
    options, initset = create_antmaze_options()
    env = EnvOptionWrapper(options, env)
    env = EnvInitsetWrapper(env, make_initiation_set(initset, option_name=None))

    # run for 1000 steps
    N = 1000
    
    max_steps = 10
    # generate trajectories
    trajs = []
    states = []
    actions = []
    print(env.action_space)
    initset_fn = make_initiation_set(initset, option_name=None, device='cpu')


    for i in tqdm(range(N)):
        s = env.reset()
        t = []
        timestep = 0
        done = False
        while not done and timestep < max_steps:
            states.append(s)
            iset = initset_fn(s)
            if iset.sum() == 0:
                print('no option available')
            a = np.random.choice(np.nonzero(iset)[0])
            # a = env.action_space.sample()
            actions.append(a)
            ret = env.step(a)
            s, r, done = ret[:3]
            t.append((s, r, done))
            timestep += 1
        trajs.append(t)

    states = torch.from_numpy(np.array(states)).float()
    printarr(states)
    printarr(np.array(actions))
    plot_initset_sinks(states, initset_fn)
    # plot trajectories
    import matplotlib.pyplot as plt
    plt.figure()
    for t in trajs:
        s = np.array(list(zip(*t))[0])
        plt.plot(s[:, 0], s[:, 1], c='k')
    plt.savefig('trajs.png')

if __name__=='__main__':
    test_options()
    # _, initset = make_antmaze_options('antmaze-umaze-v2', device='cpu')
    # plot_initset_sinks(make_initiation_set(initset, option_name=None, device='cpu'))