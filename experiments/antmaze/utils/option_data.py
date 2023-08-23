'''
    Generate the data for learning initiation set for 
    antmaze navigation policies.

    author: Rafael Rodriguez-Sanchez
    date: 17 August 2023
'''

import os
import argparse
import gym
import d4rl
import torch
import numpy as np
import pandas as pd

from experiments.antmaze.utils import TD3, load_td3_agent as load
from envs.antmaze import D4RLAntMazeWrapper
from envs.antmaze import make_env


from dataclasses import dataclass
import matplotlib.pyplot as plt
from src.utils.printarr import printarr
from matplotlib import cm

from typing import List

from joblib import Parallel, delayed
from functools import reduce

from experiments.antmaze.utils import OptionExecution
from tqdm import tqdm

GOAL_DIM = 2
GOAL_THRESHOLD = 0.5
N_SAMPLES = 5
DISTANCE = 1.




def to_df(option_data) -> pd.DataFrame:
    df = pd.DataFrame(option_data)
    return df

def collate_option_execution(batch: List[OptionExecution]) -> OptionExecution:
    batch_s = np.stack([item.s for item in batch])
    batch_next_state = np.stack([item.next_state for item in batch])
    batch_goal = np.stack([item.goal for item in batch])
    batch_success = [item.success for item in batch]
    max_reward_len = max(len(item.reward) for item in batch)
    batch_reward = np.zeros((len(batch), max_reward_len))  # Initialize with zeros
    for i, item in enumerate(batch):
        batch_reward[i, :len(item.reward)] = item.reward
    batch_steps = [item.steps for item in batch]

    return OptionExecution(
        s=batch_s,
        next_state=batch_next_state,
        goal=batch_goal,
        success=batch_success,
        reward=batch_reward,
        steps=batch_steps
    )
    
def execute_option(agent, env, states, goals, time_limit=100):
    data_points = []
    for s, g in zip(states, goals):
        s0 = env.reset()
        s0[:2] = s
        s = s0
        env.set_xy(s)
        env.set_goal(g)
        done = False
        t = 0
        rewards = []
        while not done and t < time_limit:
            _s = np.concatenate([s, g], axis=0)
            a = agent.act(_s)
            s, r, done, _ = env.step(a)
            rewards.append(r)
            t += 1
        success = done and t < time_limit
        data_points.append(OptionExecution(s0, s, g, success, np.array(rewards), t))
    return data_points

def process_option_chunk(name, direction, chunk):
    goals = chunk[:, :2] + np.array(direction)[None] * DISTANCE
    data_points = execute_option(agent, env, chunk, goals)
    return data_points

def execute_option_single(agent, env, s0, g, time_limit=100):
    done = False
    t = 0
    rewards = []
    s = s0
    env.set_goal(g)
    while not done and t < time_limit:
        _s = np.concatenate([s, g], axis=0)
        a = agent.act(_s, evaluation_mode=True)
        s, r, done, _ = env.step(a)
        rewards.append(r)
        t += 1
    success = done and t < time_limit
    return OptionExecution(s0, s, g, success, np.array(rewards), t)


def rollout(agent, env, directions, max_steps=100, time_limit=100):
    '''
        Rollout a trajectory considering the given directions.
    '''

    traj = []
    s = env.reset()
    for i in range(max_steps):
        # sample direction
        dir = np.random.choice(len(directions))
        # take action
        goal = np.array(directions[dir]) * DISTANCE + s[:2]
        option_exec = execute_option_single(agent, env, s, goal, time_limit=time_limit)
        s = option_exec.next_state
        traj.append((dir, option_exec))
    return traj


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='antmaze-umaze-v2')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--save-path', type=str, default='./')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--n-samples', type=int, default=N_SAMPLES)
    parser.add_argument('--max-exec-time', type=int, default=100)
    args = parser.parse_args()

    N_SAMPLES = args.n_samples
    HORIZON = 64
    MAX_EXEC_TIME = 100
    # make environment
    env = make_env(
                args.env,
                start=np.array([8., 0.]),
                goal=np.array([-10., -10.]),
                seed=0
            )

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # load policy
    agent = TD3(obs_size + GOAL_DIM,
        action_size,
        max_action=1.,
        use_output_normalization=False,
        device=torch.device(
            f"cuda:{args.gpu_id}" if args.gpu_id > -1 else "cpu"
        ),
        store_extra_info=False
    )

    assert args.ckpt is not None, 'Must provide a checkpoint to load'
    load(agent, args.ckpt, map_location=f"cuda:{args.gpu_id}" if args.gpu_id > -1 else "cpu")
    print(f'Loaded checkpoint {args.ckpt}')
    # sample initial states
    states = []


    directions = [[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.], [-1., -1.], [1., -1.], [-1., 1.]]
    names = ['right', 'up', 'left', 'down', 'up-right', 'down-left', 'up-left', 'down-right']
    data = []

    n_traj = N_SAMPLES // HORIZON

    # args.n_jobs = min(n_traj, args.n_jobs)

    data = []
    results = Parallel(n_jobs=args.n_jobs)(delayed(rollout)(agent, env, directions, max_steps=HORIZON, time_limit=MAX_EXEC_TIME) for i in tqdm(range(n_traj)))
    results = reduce(lambda x, acc: x + acc, results, [])
        
    ## collate data
    for i, (dir, name) in enumerate(zip(directions, names)):
        _data = [r[1] for r in results if r[0] == i]
        data.append({'data': _data, 'name': name, 'direction': np.array(dir) * DISTANCE})

    data = to_df(data)
    data_save_path = f'{args.save_path}/{args.env}/'
    os.makedirs(data_save_path, exist_ok=True)

    data_size = f"{N_SAMPLES//8//1024}k" if N_SAMPLES >= 1024 else f"{N_SAMPLES//8}"
    data.to_pickle(f'{data_save_path}/data_{data_size}.pkl')
    plt.figure()
    for i in range(len(names)):
        d = data.iloc[i]
        collated_data = collate_option_execution(d.data)
        success = collated_data.success
        states = collated_data.s
        print(f'{d.name} success rate: {np.array(success).mean()}')

        plt.subplot(2, 4, i+1)
        # success = np.array([d.success for d in datum])
        success = np.array(success)
        states = np.array(states)
        # printarr(states, success)
        plt.scatter(states[:, 0], states[:, 1], c=success, cmap=cm.coolwarm, s=5)
        plt.title(d['name'])
        plt.grid()
    # plt.colorbar()
    plt.savefig(f'{data_save_path}/{args.env}_initset.png')


