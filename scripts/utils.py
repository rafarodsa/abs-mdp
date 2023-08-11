"""
    author: Rafael Rodriguez-Sanchez
    email: rrs@brown.edu
    date: January 2023
"""

from envs.pinball.pinball_gym import PinballEnvContinuous as Pinball
from envs.pinball.controllers_pinball import create_position_controllers_v0 as OptionFactory

import numpy as np

from src.absmdp.datasets import Transition

import os
import re

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


    duration = t
    next_o = np.array(env.render()) if obs_type == 'pixel' else next_s
    can_execute = not timeout and can_execute

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


def collect_trajectory(env, options, obs_type='simple', max_exec_time=200, horizon=100, with_failures=True):
    '''
        Collect samples from sequential option execution. Uniform policy over available options.
    '''
    trajectory = []
    infos = []
    # print('Collecting trajectory...')
    s = env.reset()
    # print('Sampled initial state...')
    o = np.array(env.render()) if obs_type == 'pixel' else np.array(s)
    executed = True
    done = False
    
    for t in range(horizon): # execute t options in sequence
        if done: # episode terminated
            break
        if executed or with_failures:
            initiation_mask_s = compute_initiation_masks(s, options).astype(np.float32)
        else:
            initiation_mask_s[option_n] = 0

        if np.sum(initiation_mask_s) == 0 and not with_failures:
            print(f'No options available in {s} at time {t}, setting as terminal state.')
            if len(trajectory) > 0:
                trajectory[-1] = trajectory[-1].modify(done=True)
            break # no options available
        if not with_failures:
            option_n = np.random.choice(len(options), p=initiation_mask_s/np.sum(initiation_mask_s)) # sample option uniformly
        else:
            option_n = np.random.choice(len(options))
            
        option = options[option_n]
        try:
            o, next_o, rewards, executed, duration, info = execute_option(env, np.array(s), option, obs_type=obs_type, max_exec_time=max_exec_time)
        except Exception as e:
            print('Error executing option')
            print(e.with_traceback())
            trajectory = []
            break
        next_s = next_o if obs_type == 'simple' else info['next_state']
        rewards = rewards + [0] * (max_exec_time - len(rewards))
        done = info['done']

        trajectory.append(Transition(np.array(o), option_n, np.array(next_o), rewards, done, executed, duration, np.array(initiation_mask_s), info, np.float32(t==0)))
        infos.append(info)
        s = next_s
        

    return trajectory



### File and directory manipulation

CONFIG_SUBDIR = '/logs/infomax-pb'
CKPT_SUBDIR = 'ckpts'
MODEL_SUBDIR = 'phi_train'
LAST_CKPT_REGEX = r'last(-v([0-9])+)?.ckpt'
def get_experiment_info(experiment_dir):
    # config file
    config_file = ''
    ckpt_file = ''
    config_path = f'{experiment_dir}/{MODEL_SUBDIR}/{CONFIG_SUBDIR}'
    if os.path.exists(config_path):
        first_level = next(os.walk(config_path))
        versions = [int(dir.split('_')[-1])  for dir in first_level[1] if 'version' in dir]
        last_version = max(versions)
        config_file = f'{config_path}/version_{last_version}/hparams.yaml'
    
    # ckpt
    ckpt_path = f'{experiment_dir}/{MODEL_SUBDIR}/{CKPT_SUBDIR}'
    if os.path.exists(ckpt_path):
        first_level = next(os.walk(ckpt_path))
        versions = []
        last_versions = [file for file in first_level[-1] if 'last' in file]
        version_numbers = [re.search(LAST_CKPT_REGEX, f)[2] for f in last_versions]
        version_numbers = list(map(lambda s: 0 if s is None else int(s), version_numbers))

        ckpt_name, _ = max(zip(last_versions, version_numbers), key=lambda t: t[-1])
        ckpt_file = f'{ckpt_path}/{ckpt_name}'
    return config_file, ckpt_file


def prepare_outdir(outdir):
    pass
