"""
    author: Rafael Rodriguez-Sanchez
    email: rrs@brown.edu
    date: 20 February 2023
"""
import os, io
from functools import reduce

import numpy as np
import torch, argparse
from tqdm import tqdm
import zipfile

from joblib import Parallel, delayed

from utils import collect_trajectory
from envs.pinball.pinball_gym import PinballEnvContinuous as Pinball
from envs.pinball.controllers_pinball import create_position_controllers_v0 as OptionFactory
from envs.pinball.controllers_pinball import create_position_options as OptionFactory2
from envs.pinball.controllers_pinball import PinballGridOptions as GridOptionsFactory

from PIL import Image

from src.absmdp.datasets import ObservationImgFile as ObservationFile

import matplotlib.pyplot as plt
import yaml

def plot_obstacles(obstacles, ax=None):
    ax = ax if ax is not None else plt
    for o in obstacles:
        ax.plot(o[:, 0], o[:, 1], c='k')

def preprocess_img(img_array):
    img = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img)

def save_and_compress(trajectories, zfile):
    trajectories = [save_and_compress_trajectory(trajectory, id, zfile) for id, trajectory in enumerate(trajectories)]
    return trajectories

def save_and_compress_trajectory(trajectory, trajectory_id, zfile):

    if len(trajectory) < 1:
        return trajectory
    
    obs = [preprocess_img(t.obs) for t in trajectory] + [preprocess_img(trajectory[-1].next_obs)]
    fnames = [f'tj_{trajectory_id}_obs_{id}.png' for id in range(len(obs))]

    byte_streams = [io.BytesIO() for fn in fnames]
    for i, bs, fn in zip(obs, byte_streams, fnames):
        i.save(bs, format='png')
        zfile.writestr(fn, bs.getvalue())

    timesteps = list(range(len(obs)))
    obs = [ObservationFile(trajectory_id, i) for i in timesteps[:-1]]
    next_obs = [ObservationFile(trajectory_id, i) for i in timesteps[1:]]
    
    trajectory = [t.modify(obs=o, next_obs=next_o) for t, o, next_o in zip(trajectory, obs, next_obs)]

    return trajectory



if __name__== "__main__":

    ######## Parameters
    np.set_printoptions(precision=3)
    configuration_file = "envs/pinball/configs/pinball_simple_single.cfg"
    num_traj = 100
    observation_type = 'simple'
    DEBUG = False

    ###### CMDLINE ARGUMENTS

    dataset_file_path = '/Users/rrs/Desktop/abs-mdp/data/'
    dataset_name = 'pinball_simple_obs.pt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default=dataset_file_path)
    parser.add_argument('--env-config', type=str, default=configuration_file)
    parser.add_argument('--num-traj', type=int, default=num_traj)
    parser.add_argument('--max-horizon', type=int, default=64)
    parser.add_argument('--observation', type=str, default=observation_type)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--max-exec-time', type=int, default=100)
    parser.add_argument('--image-size', type=int, default=100)
    parser.add_argument('--option_type', type=str, default='continuous-v1')
    parser.add_argument('--grid_size', type=int, default=20)
    parser.add_argument('--uniform', action='store_true', default=False)
    args = parser.parse_args()

    dir, name = os.path.split(args.save_path)
    name, ext = os.path.splitext(name)
    os.makedirs(dir, exist_ok=True)
    zfile = zipfile.ZipFile(args.save_path, 'w')
    ######## DATA GENERATION #######

    trajectories = []  # (o, a, o', rewards, executed, duration, initiation_masks, info)

    grid_size = args.image_size

    env = Pinball(config=args.env_config, width=grid_size, height=grid_size, render_mode='rgb_array') 

    if args.option_type == 'continuous-v1':
        options = OptionFactory2(env, translation_distance=1/args.grid_size, check_can_execute=False)
    elif args.option_type == 'continuous-v0':
        options = OptionFactory(env)
    elif args.option_type == 'grid':
        options = GridOptionsFactory(env, args.grid_size, 1/args.grid_size/5)
    else:
        raise ValueError('Unknown option type')

    max_exec_time = args.max_exec_time
    
    options_desc = {i: str(o) for i, o in enumerate(options)}

    trajectories = Parallel(n_jobs=args.n_jobs)(delayed(collect_trajectory)(env, options, obs_type=args.observation, max_exec_time=max_exec_time, horizon=args.max_horizon, uniform=args.uniform) for i in tqdm(range(args.num_traj)))        
    
    if args.observation == 'pixel':
        trajectories = save_and_compress(trajectories, zfile)


    trajectories_len = np.array(list(map(len, trajectories)))
    idx = list(map(int, np.nonzero(trajectories_len > 0)[0]))
    print(f'Filtering out {len(trajectories) - (trajectories_len > 0).sum()} trajectories that have length 0')
    trajectories = [trajectories[i] for i in idx]

    print(f'Trajectories mean length {trajectories_len.mean()}')
    print(f'Trajectories max length {trajectories_len.max()}')
    print(f'Trajectories min length {trajectories_len.min()}')

    ##### Print dataset statistics
    transition_samples = reduce(lambda x, acc: x + acc, trajectories, [])
    n_samples = len(transition_samples)


    o, option_n, next_o, rewards, done, executed, duration, initiation_mask, info, p0 = zip(*transition_samples)
    stats = {}
    _r = np.array(list(map(lambda x: sum(x), rewards)))
    _r_len = list(map(len, rewards))
    
    s = np.array(list(map(lambda x: x['state'], info)))
    next_s = np.array(list(map(lambda x: x['next_state'], info)))

    __errors = []
    
    for i in range(len(options)):
        idx = np.array(option_n) == i
        _executed = np.array(executed)[idx]
        n_executions = _executed.sum()
        _duration = np.array(duration)[idx]
        option_rewards = _r[idx][_executed==1]/_duration[_executed==1]
        avg_duration = np.array(duration)[idx].mean()
       
       
        s_executed = s[idx][_executed==1]
        next_s_executed = next_s[idx][_executed==1]
        state_change = next_s_executed - s_executed
        state_change_min, state_change_max = state_change.min(0), state_change.max(0)
        state_change_mean, state_change_std = state_change.mean(0), state_change.std(0)

        stats[i] = {
            'prob_executions': float(n_executions/_executed.shape[0]),
            'n_executions_tries': int(_executed.shape[0]),
            'avg_duration': float(avg_duration),
            'avg_reward': option_rewards.mean().item(),
            'min_reward': option_rewards.min().item(),
            'max_reward': option_rewards.max().item(),
            'state_change_min': state_change_min.tolist(),
            'state_change_max': state_change_max.tolist(),
            'state_change_mean': state_change_mean.tolist(),
            'state_change_std': state_change_std.tolist(),
        }
        
        #### PLOT TEST
        _effects = [np.array([[0, -1/args.grid_size]]),
                    np.array([[0, 1/args.grid_size]]),
                    np.array([[-1/args.grid_size, 0]]),
                    np.array([[1/args.grid_size, 0]])]

        s_not_executed = s[idx][_executed==0]
        next_s_not_executed = next_s[idx][_executed==0]
        predicted_next_s = s_not_executed[:, :2] + _effects[i]
        lines = np.vstack([s_not_executed[..., :2][None], predicted_next_s[None]])
        if DEBUG:
            __errors.append((s_not_executed, predicted_next_s, next_s_not_executed))

            print(s_not_executed[:, :2])
            print(predicted_next_s)
            print(next_s_not_executed[:, :2])

            f, ax = plt.subplots()
            plt.scatter(s_not_executed[:, 0], s_not_executed[:, 1], s=5, c='r')
            plt.scatter(predicted_next_s[:, 0], predicted_next_s[:, 1], s=5, c='b')
            plt.scatter(next_s_not_executed[:, 0], next_s_not_executed[:, 1], s=5, c='g')

            for j in range(predicted_next_s.shape[0]):
                circle = plt.Circle(predicted_next_s[j], env.pinball.ball.radius, color='b', fill=False)
                ax.add_artist(circle)

            plot_obstacles(env.expanded_obs, ax)
            # plt.scatter(next_s_not_executed[:, 0], next_s_not_executed[:, 1], s=5, c='g')
            line_type = ['-', '--', ':', '-.']
            for j in range(s_not_executed.shape[0]):
                plt.plot(lines[:, j, 0], lines[:, j, 1], c='k', linestyle=line_type[i])

        print(f'--------Option-{i}: {options_desc[i]}---------')
        print(f"Executed {n_executions}/{_executed.shape[0]} times ({n_executions/_executed.shape[0]})")
        print(f"Average duration {avg_duration}")
        print(f"Average reward {option_rewards.mean()}. Max reward: {option_rewards.max()}. Min reward: {option_rewards.min()}")
        print(f"Reward length: mean: {np.array(_r_len)[idx].mean()}, max: {np.array(_r_len)[idx].max()}, min: {np.array(_r_len)[idx].min()}")
        print(f"State change: mean: {state_change_mean}, max: {state_change_max}, min: {state_change_min}, std {state_change_std}")
    
    debug = {
        'latent_states': info,
        'options': options_desc,
        'stats': stats
    }


    # stats and debug data
    options_stats = {}
    options_stats['Option Stats'] = dict(zip(options_desc.values(), stats.values()))
    options_stats['Trajectory Stats'] = {
            'mean_length': trajectories_len.mean().item(),
            'max_length': trajectories_len.max().item(),
            'min_length': trajectories_len.min().item(),
            'max_execution_length': args.max_exec_time,
            'grid_size': args.grid_size,
            'max_horizon': args.max_horizon,
            'option_type': args.option_type,
            'n_trajectories': len(trajectories),
            'n_samples': n_samples,
    }
    # dump in yaml
    
    with open(f'{dir}/{name}_stats.yaml', 'w') as f:
        yaml.dump(options_stats, f)
    if DEBUG:
        torch.save(__errors, 'errors.pt')
        obstacles = [np.array(o.points + [o.points[0]]) for o in env.get_obstacles()]
        plot_obstacles(env.vertices)
        plt.savefig(f'./failures.png')


    ########### SAVE DATASET ###########
    

    # zfile = zipfile.ZipFile(save_path, 'w')
    bs = io.BytesIO()
    print('---------------------------------')
    print(f'Dataset saved at {args.save_path}/transitions.pt')
    torch.save(trajectories, bs)
    zfile.writestr('transitions.pt', bs.getvalue())

    bs = io.BytesIO()
    torch.save(debug, bs)
    zfile.writestr('debug.pt', bs.getvalue())
    print(f'Debug info saved at {args.save_path}/debug.pt')
    zfile.close()