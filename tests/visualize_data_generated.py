"""
    author: Rafael Rodriguez-Sanchez
    email: rrs@brown.edu
    date: November 2022
"""

from src.envs.pinball.pinball_gym import PinballEnvContinuous as Pinball
from src.envs.pinball.controllers_pinball import create_position_controllers as OptionFactory

import numpy as np
from tqdm import tqdm
import argparse

from collections import defaultdict
import matplotlib.pyplot as plt

######## Parameters

configuration_file = "/Users/rrs/Desktop/abs-mdp/src/envs/pinball/configs/pinball_no_obstacles.cfg"
n_samples = 100
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

def execute_option(env, initial_state, option):
    t = 0
    next_s = initial_state
    can_execute = option.execute(initial_state)
    rewards = []
    done = False
    env.reset(initial_state)
    if can_execute:
        while option.is_executing() and not done and t < 1000:
            action = option.act(next_s) 
            if action is None:
                break
            next_s, r, done, _, _ = env.step(action)
            rewards.append(r)
            t += 1
    duration = t
    return next_s, rewards, can_execute, duration


def compute_initiation_masks(state, options):
    return np.array([o.initiation(state) for o in options])



######## DATA GENERATION #######

dataset = []  # (s', a, s', rewards, executed, duration, initiation_masks)
env = Pinball(config=args.env_config)
init_states = env.sample_initial_positions(args.n_samples)
options = OptionFactory(env)
max_exec_time = 100


rewards_per_action = defaultdict(list)
init_states = np.array([[3,3,0,0],
                         [3,5,0,0],
                         [4,4,0,0],
                         [5,3,0,0],
                         [10,10,0,0]])/10



for i in tqdm(range(4)):
    for j, o in enumerate(options):
        s = np.array(init_states[i])
        next_s, rewards, executed, duration = execute_option(env, s, o)
        initiation_mask_s = compute_initiation_masks(init_states[i], options)
        initiation_mask_s_prime = compute_initiation_masks(next_s, options)
        rewards = rewards + [0] * (max_exec_time - len(rewards))
        rewards_per_action[j].append((s, next_s, rewards))
        dataset.append((init_states[i], j, next_s, rewards, executed, duration, np.array([initiation_mask_s, initiation_mask_s_prime])))



for i in range(4):
    plt.subplot(2,2,i+1)
    plt.scatter(init_states[:, 0], init_states[:, 1], marker='*')
    next_s = np.array(list(zip(*rewards_per_action[i]))[1])
    plt.scatter(next_s[:, 0], next_s[:, 1], marker='x')

plt.show()


######