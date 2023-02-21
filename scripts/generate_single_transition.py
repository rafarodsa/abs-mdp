"""
    author: Rafael Rodriguez-Sanchez
    email: rrs@brown.edu
    date: January 2023 (v0)
          February 2023 (v0.1)
"""

from envs.pinball.pinball_gym import PinballEnvContinuous as Pinball
from envs.pinball.controllers_pinball import create_position_controllers as OptionFactory

import numpy as np
import torch
from tqdm import tqdm
import argparse

from functools import reduce

from joblib import Parallel, delayed

from scripts.utils import run_options
import os

if __name__== "__main__":

    ######## Parameters
    np.set_printoptions(precision=3)
    configuration_file = "/Users/rrs/Desktop/abs-mdp/envs/pinball/configs/pinball_simple_single.cfg"
    n_samples = 100
    observation_type = 'simple'

    ###### CMDLINE ARGUMENTS

    dataset_file_path = '/Users/rrs/Desktop/abs-mdp/data/'
    dataset_name = 'pinball_simple_obs.pt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default=dataset_file_path+dataset_name)
    parser.add_argument('--env-config', type=str, default=configuration_file)
    parser.add_argument('--n-samples', type=int, default=n_samples)
    parser.add_argument('--observation', type=str, default=observation_type)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--max-exec-time', type=int, default=1000)
    parser.add_argument('--image-size', type=int, default=100)
    args = parser.parse_args()


    ######## DATA GENERATION #######

    dataset = []  # (o, a, o', rewards, executed, duration, initiation_masks, info)

    grid_size = args.image_size

    env = Pinball(config=args.env_config, width=grid_size, height=grid_size, render_mode='rgb_array') 

    # init_states = env.sample_init_states(args.n_samples) # sample uniformly the velocities.
    init_states = env.sample_initial_positions(args.n_samples) # sample uniform (valid) positions with zero velocities.

    options = OptionFactory(env)
    max_exec_time = args.max_exec_time
    
    options_desc = {i: str(o) for i, o in enumerate(options)}

<<<<<<< HEAD:scripts/generate_single_transition.py
    results = Parallel(n_jobs=args.n_jobs)(delayed(run_options)(env, tuple(init_states[i]), options, obs_type=args.observation, max_exec_time=max_exec_time) for i in tqdm(range(args.n_samples)))        
=======
    results = Parallel(n_jobs=args.n_jobs)(delayed(run_option)(env, tuple(init_states[i]), options, obs_type=args.observation, max_exec_time=max_exec_time) for i in tqdm(range(args.n_samples)))     
>>>>>>> debugging-loss:scripts/pinball_generate_data.py

    ##### Print dataset statistics
    dataset, info = zip(*results)

    transition_samples = reduce(lambda r, acc: list(r) + list(acc), dataset, [])
    info = reduce(lambda r, acc: list(r) + list(acc), info, [])
    o, j, next_o, rewards, done, executed, duration, initiation_masks, _, _ = zip(*transition_samples)
    stats = {}
    _r = np.array(list(map(lambda x: sum(x), rewards)))
    _r_len = list(map(len, rewards))
    
    s = np.array(list(map(lambda x: x['state'], info)))
    next_s = np.array(list(map(lambda x: x['next_state'], info)))
    
    for i in range(len(options)):
        idx = np.array(j) == i
        _executed = np.array(executed)[idx]
        n_executions = _executed.sum()
        _duration = np.array(duration)[idx]
        option_rewards = _r[idx][_executed==1]/_duration[_executed==1]
        avg_duration = np.array(duration)[idx].mean()
       
       
        s_executed = s[idx][_executed==1]
        next_s_executed = next_s[idx][_executed==1]
        state_change = next_s_executed - s_executed
        state_change_min, state_change_max = state_change.min(0), state_change.max(0)
        state_change_mean = state_change.mean(0)

        stats[i] = {
            'prob_executions': n_executions/args.n_samples,
            'avg_duration': avg_duration,
            'avg_reward': option_rewards.mean(),
            'min_reward': option_rewards.min(),
            'max_reward': option_rewards.max(),
            'state_change_min': state_change_min,
            'state_change_max': state_change_max,
            'state_change_mean': state_change_mean
        }

        print(f"--------Option-{i}: {options_desc[i]}---------")
        print(f"Executed {n_executions}/{args.n_samples} times")
        print(f"Average duration {avg_duration}")
        print(f"Average reward {option_rewards.mean()}. Max reward: {option_rewards.max()}. Min reward: {option_rewards.min()}")
        print(f"Reward length: mean: {np.array(_r_len)[idx].mean()}, max: {np.array(_r_len)[idx].max()}, min: {np.array(_r_len)[idx].min()}")
        print(f"State change: mean: {state_change_mean}, max: {state_change_max}, min: {state_change_min}")
    
    debug = {
        'latent_states': info,
        'options': options_desc,
        'stats': stats
    }

    ########### SAVE DATASET ###########
    
    dir, name = os.path.split(args.save_path)
    os.makedirs(dir, exist_ok=True)
    torch.save(dataset, args.save_path)
    print('---------------------------------')
    print(f'Dataset saved at {args.save_path}')
    torch.save(debug, args.save_path.replace('.pt', '_debug.pt'))
    print(f'Debug info saved at {args.save_path.replace(".pt", "_debug.pt")}')