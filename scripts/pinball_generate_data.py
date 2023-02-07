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

from joblib import Parallel, delayed

from scripts.utils import run_option
from functools import partial
import os

if __name__== "__main__":

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
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--max-exec-time', type=int, default=1000)
    args = parser.parse_args()


    ######## DATA GENERATION #######

    dataset = []  # (o, a, o', rewards, executed, duration, initiation_masks, info)

    grid_size = 50

    env = Pinball(config=args.env_config, width=grid_size, height=grid_size, render_mode='rgb_array') 

    init_states = env.sample_initial_positions(args.n_samples)
    options = OptionFactory(env)
    max_exec_time = args.max_exec_time
    

    results = Parallel(n_jobs=args.n_jobs)(delayed(run_option)(env, tuple(init_states[i]), options, obs_type=args.observation) for i in tqdm(range(args.n_samples)))        

    ##### Print dataset statistics
    dataset = reduce(lambda r, acc: list(r) + list(acc), results, [])
    o, j, next_o, rewards, executed, duration, initiation_masks = zip(*dataset)
    print(f"Mean duration {np.mean(np.array(duration))}")


    ########### SAVE DATASET ###########
    
    dir, name = os.path.split(args.save_path)
    os.makedirs(dir, exist_ok=True)
    torch.save(dataset, args.save_path)