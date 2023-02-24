"""
    author: Rafael Rodriguez-Sanchez
    email: rrs@brown.edu
    date: January 2023
"""

from envs.pinball.pinball_gym import PinballEnvContinuous as Pinball
from envs.pinball.controllers_pinball import create_position_controllers as OptionFactory

import numpy as np

from src.absmdp.datasets import Transition

####### Auxiliary Functions

def execute_option(env, initial_state, option, obs_type='simple', max_exec_time=1000):
    t = 0
    next_s = np.array(initial_state)
    can_execute = option.execute(initial_state)
    rewards = []
    done = False
    s = np.array(env.reset(initial_state))
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

    info = {'state': s, 'next_state': next_s, 'done': done}
    return o, next_o, rewards, can_execute, duration, info


def compute_initiation_masks(state, options):
    return np.array([o.initiation(state) for o in options])

def run_options(env, init_state, options, obs_type='simple', max_exec_time=200):
    '''
        Run all from an initial state.
        env: environment
        init_state: initial state
        options: list of options
        obs_type: 'simple' or 'pixel'
        max_exec_time: maximum execution time for each option (time if runs for longer)
    '''
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
 
        dataset.append(Transition(o, option_n, next_o, rewards, info['done'], executed, duration, np.array([initiation_mask_s, initiation_mask_s_prime]), info, np.float32(True)))

    return list(dataset), infos


def collect_trajectory(env, options, obs_type='simple', max_exec_time=200, horizon=100):
    '''
        Collect samples from sequential option execution. Uniform policy over available options.
    '''
    trajectory = []
    infos = []
    s = env.reset()
    o = np.array(env.render()) if obs_type == 'pixel' else np.array(s)
    executed = True
    done = False
    for t in range(horizon): # execute t options in sequence
        if done: # episode terminated
            break
        if executed:
            initiation_mask_s = compute_initiation_masks(s, options).astype(np.float32)
        else:
            initiation_mask_s[option_n] = 0

        if np.sum(initiation_mask_s) == 0:
            break # no options available

        option_n = np.random.choice(len(options), p=initiation_mask_s/np.sum(initiation_mask_s)) # sample option uniformly
        option = options[option_n]
        o, next_o, rewards, executed, duration, info = execute_option(env, np.array(s), option, obs_type=obs_type, max_exec_time=max_exec_time)
        next_s = next_o if obs_type == 'simple' else info['next_state']
        rewards = rewards + [0] * (max_exec_time - len(rewards))
        done = info['done']

        trajectory.append(Transition(np.array(o), option_n, np.array(next_o), rewards, done, executed, duration, np.array(initiation_mask_s), info, np.float32(t==0)))
        infos.append(info)
        s = next_s
        

    return trajectory