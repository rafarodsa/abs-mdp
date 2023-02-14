"""
    author: Rafael Rodriguez-Sanchez
    email: rrs@brown.edu
    date: January 2023
"""

from envs.pinball.pinball_gym import PinballEnvContinuous as Pinball
from envs.pinball.controllers_pinball import create_position_controllers as OptionFactory

import numpy as np

####### Auxiliary Functions

def execute_option(env, initial_state, option, obs_type='simple', max_exec_time=1000):
    t = 0
    next_s = initial_state
    can_execute = option.execute(initial_state)
    rewards = []
    done = False
    s = env.reset(initial_state)
    next_s = s

    o = np.array(env.render()) if obs_type == 'pixel' else s
    info = {}
    timeout = False
    if can_execute:
        while option.is_executing() and not done and t < max_exec_time:
            action = option.act(next_s)
            if action is None:
                break # option terminated
            next_s, r, done, _, info = env.step(action)
            rewards.append(r)
            t += 1
        if t >= max_exec_time and not done:
            timeout = True # option timed out

    if not timeout:
        duration = t 
        next_o = np.array(env.render()) if obs_type == 'pixel' else next_s
    else:
        duration = 0
        next_o = o
        next_s = s
        rewards = []
        can_execute = False

    info = {'state': s, 'next_state': next_s}
    return o, next_o, rewards, can_execute, duration, info


def compute_initiation_masks(state, options):
    return np.array([o.initiation(state) for o in options])

def run_option(env, init_state, options, obs_type='simple', max_exec_time=200):
    dataset = []
    infos = []
    for option_n, option in enumerate(options):
        s = np.array(init_state)
        o, next_o, rewards, executed, duration, info = execute_option(env, s, option, obs_type=obs_type, max_exec_time=max_exec_time)
        next_s = info['next_state']
        s = info['state']
        infos.append(info)
        initiation_mask_s = compute_initiation_masks(s, options)
        initiation_mask_s_prime = compute_initiation_masks(next_s, options)
        rewards = rewards + [0] * (max_exec_time - len(rewards))
 
        dataset.append((o, option_n, next_o, rewards, executed, duration, np.array([initiation_mask_s, initiation_mask_s_prime])))

    return tuple(dataset), infos