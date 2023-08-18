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

from td3.utils import load
from td3.TD3AgentClass import TD3
from td3.train import make_env
from envs.antmaze import D4RLAntMazeWrapper

from dataclasses import dataclass
import matplotlib.pyplot as plt
from printarr import printarr
from matplotlib import cm

from typing import List

from joblib import Parallel, delayed
from functools import reduce

from experiments.antmaze.utils import OptionExecution

GOAL_SIZE = 2
GOAL_THRESHOLD = 0.5
N_SAMPLES = 5
DISTANCE = 1.


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
    
def process_option_chunk(name, direction, chunk):
    goals = chunk[:, :2] + np.array(direction)[None] * DISTANCE
    data_points = execute_option(agent, env, chunk, goals)
    return data_points


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='antmaze-umaze-v2')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--save-path', type=str, default='./')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--n-samples', type=int, default=N_SAMPLES)
    args = parser.parse_args()

    N_SAMPLES = args.n_samples
    # make environment
    env = make_env(args.env,
            start=np.array([8., 0.]),
            goal=np.array([0., 0.]),
            seed=0,
            dense_reward=False)

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # load policy
    agent = TD3(obs_size + GOAL_SIZE,
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

    # sample initial states
    states = []

    for _ in range(N_SAMPLES):
        s = env.sample_random_state()
        states.append(s)

    states = np.array(states)
    printarr(states)

    plt.figure()
    plt.scatter(states[:, 0], states[:, 1])
    plt.savefig('initset.png')


    directions = [[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.], [-1., -1.], [1., -1.], [-1., 1.]]
    names = ['right', 'up', 'left', 'down', 'up-right', 'down-left', 'up-left', 'down-right']
    data = []

    args.n_jobs = min(N_SAMPLES, args.n_jobs)

    chunk_size = N_SAMPLES // args.n_jobs  # Number of samples per chunk

    data = []
    for name, dir in zip(names, directions):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_option_chunk)(name, dir, states_chunk)
            for states_chunk in np.array_split(states, args.n_jobs)  # Split the states array into M chunks
        )

        results = reduce(lambda x, acc: x + acc, results, [])
        data.append({'data': results, 'name': name, 'direction': np.array(dir) * DISTANCE})

    data = to_df(data)
    data_save_path = f'{args.save_path}/{args.env}/'
    os.makedirs(data_save_path, exist_ok=True)

    data_size = f"{N_SAMPLES//1000}k" if N_SAMPLES >= 1000 else f"{N_SAMPLES}"
    data.to_pickle(f'{data_save_path}/data_{data_size}.pkl')
    plt.figure()
    for i in range(len(names)):
        d = data.iloc[i]
        success = collate_option_execution(d.data).success
        plt.subplot(2, 4, i+1)
        # success = np.array([d.success for d in datum])
        success = np.array(success)
        plt.scatter(states[:, 0], states[:, 1], c=success, cmap=cm.coolwarm, s=5)
        plt.title(d['name'])
        plt.grid()
    # plt.colorbar()
    plt.savefig(f'{args.env}_initset.png')


