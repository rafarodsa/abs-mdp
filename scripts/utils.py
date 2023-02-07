
"""
    author: Rafael Rodriguez-Sanchez
    email: rrs@brown.edu
    date: January 2023
"""

from src.envs.pinball.pinball_gym import PinballEnvContinuous as Pinball
from src.envs.pinball.pinball_gym import PinballPixelWrapper
from src.envs.pinball.controllers_pinball import create_position_controllers as OptionFactory

import numpy as np
import torch
from tqdm import tqdm
import argparse

from functools import reduce
from collections import defaultdict
import matplotlib.pyplot as plt

from joblib import Parallel, delayed, wrap_non_picklable_objects



####### Auxiliary Functions

def execute_option(env, initial_state, option, obs_type='simple'):
    t = 0
    next_s = initial_state
    can_execute = option.execute(initial_state)
    rewards = []
    done = False
    s = env.reset(initial_state)
    next_s = s

    o = np.array(env.render()) if obs_type == 'pixel' else s
    info = {}
    if can_execute:
        while option.is_executing() and not done and t < 1000:
            action = option.act(next_s)
            if action is None:
                break
            next_s, r, done, _, info = env.step(action)
            rewards.append(r)
            t += 1
    duration = t
    next_o = np.array(env.render()) if obs_type == 'pixel' else next_s

    info = {'state': s, 'next_state': next_s}
    return o, next_o, rewards, can_execute, duration, info


def compute_initiation_masks(state, options):
    return np.array([o.initiation(state) for o in options])

def run_option(env, init_state, options, obs_type='simple', max_exec_time=200):
    dataset = []
    infos = []
    for option_n, option in enumerate(options):
        s = np.array(init_state)
        o, next_o, rewards, executed, duration, info = execute_option(env, s, option, obs_type=obs_type)
        next_s = info['next_state']
        s = info['state']
        infos.append(info)
        initiation_mask_s = compute_initiation_masks(s, options)
        initiation_mask_s_prime = compute_initiation_masks(next_s, options)
        rewards = rewards + [0] * (max_exec_time - len(rewards))
 
        dataset.append((o, option_n, next_o, rewards, executed, duration, np.array([initiation_mask_s, initiation_mask_s_prime])))

    return tuple(dataset), infos