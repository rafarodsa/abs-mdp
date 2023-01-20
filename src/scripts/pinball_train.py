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

from collections import defaultdict
import matplotlib.pyplot as plt

######## Parameters

configuration_file = "/Users/rrs/Desktop/abs-mdp/src/envs/pinball/configs/pinball_no_obstacles.cfg"
n_samples = 5000
observation_type = 'simple'



###### CMDLINE ARGUMENTS

dataset_file_path = '/Users/rrs/Desktop/abs-mdp/data/'
dataset_name = 'pinball_no_obstacle.pt'

parser = argparse.ArgumentParser()
parser.add_argument('--save-path', type=str, default=dataset_file_path+dataset_name)
parser.add_argument('--env-config', type=str, default=configuration_file)
parser.add_argument('--n-samples', type=int, default=n_samples)
parser.add_argument('--observation', type=str, default=observation_type)
args = parser.parse_args()


####### Auxiliary Functions

def execute_option(env, initial_state, option, obs_type='simple'):
    t = 0
    next_s = initial_state
    can_execute = option.execute(initial_state)
    rewards = []
    done = False
    s = env.reset(initial_state)
    next_s = s

    o = env.render() if obs_type == 'pixel' else s

    if can_execute:
        while option.is_executing() and not done and t < 1000:
            action = option.act(next_s)
            if action is None:
                break
            next_s, r, done, _, info = env.step(action)
            rewards.append(r)
            t += 1
    duration = t
    next_o = env.render() if obs_type == 'pixel' else next_s
    
    info = {'state': initial_state, 'next_state': next_s if 'next_state' in info else next_s}
    return o, next_o, rewards, can_execute, duration, info


def compute_initiation_masks(state, options):
    return np.array([o.initiation(state) for o in options])



######## DATA GENERATION #######

dataset = []  # (o, a, o', rewards, executed, duration, initiation_masks, info)

grid_size = 50

env = Pinball(config=args.env_config, width=grid_size, height=grid_size, render_mode='rgb_array') 

init_states = env.sample_initial_positions(args.n_samples)
options = OptionFactory(env)
max_exec_time = 100


rewards_per_action = defaultdict(list)
for i in tqdm(range(args.n_samples)):
    for j, o in enumerate(options):
        s = np.array(init_states[i])
        o, next_o, rewards, executed, duration, info = execute_option(env, s, o)
        
        next_s = info['next_state']
        s = info['state']
        
        initiation_mask_s = compute_initiation_masks(s, options)
        initiation_mask_s_prime = compute_initiation_masks(next_s, options)
        rewards = rewards + [0] * (max_exec_time - len(rewards))
        rewards_per_action[j].append((s, next_s, rewards))
        
        dataset.append((o, j, next_o, rewards, executed, duration, np.array([initiation_mask_s, initiation_mask_s_prime]), info))


########### SAVE DATASET ###########

torch.save((dataset, rewards_per_action), args.save_path)


######