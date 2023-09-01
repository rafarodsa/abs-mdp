"""
    author: Rafael Rodriguez-Sanchez
    email: rrs@brown.edu
    date: 22 August 2023
"""
import os, io
from functools import reduce

import numpy as np
import torch, argparse
from tqdm import tqdm
import zipfile

from joblib import Parallel, delayed


from src.absmdp.datasets import ObservationImgFile as ObservationFile

import matplotlib.pyplot as plt
import yaml

from make_antmaze_with_options import make_antmaze, create_antmaze_options, make_initiation_set

from src.absmdp.datasets import Transition

def run_option(env, s, option):
    
    timestep = 0
    done = False
    rews = []
    option.check_can_execute = False
    option.execute(s)

    while not done and timestep < option.max_executing_time and option.is_executing():
        # choose action
        action = option.act(s)
        if action is None:
            break
        # execute action
        next_s, r, done, truncated, info = env.step(action)
        rews.append(np.sum(r))
        s = next_s
        timestep += 1

    return next_s, rews, done, info, timestep, timestep < option.max_executing_time

def collect_trajectory(env, options, initset_fn, horizon, uniform=False):

    trajectory = []
    timestep = 0
    s = env.reset()
    done = False
    while not done and timestep < horizon:
        # choose action
        initiation_mask = initset_fn(s)
        if uniform:
            action = np.random.choice(len(options))
        else:
            if initiation_mask.sum() == 0:
                print(f'no option available at state {s[:2]} ')
                break
            avail_options = initiation_mask.nonzero()[0]
            action = np.random.choice(avail_options)

        # execute action
        next_s, r, done, info, duration, success = run_option(env, s, options[action])

        info['next_state'] = next_s
        info['state'] = s

        #Transition(np.array(o), option_n, np.array(next_o), rewards, done, executed, duration, np.array(initiation_mask_s), info, np.float32(t==0))
        trajectory.append(Transition(s, int(action), next_s, r, done, success, duration, initiation_mask.numpy(), info, np.float32(timestep==0)))
        s = next_s
        timestep += 1

    return trajectory



if __name__== "__main__":

    ######## Parameters
    np.set_printoptions(precision=3)
    N_TRAJ = 100
    DEBUG = False

    ###### CMDLINE ARGUMENTS

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--env', type=str, default='antmaze-umaze-v2')
    parser.add_argument('--num-traj', type=int, default=N_TRAJ)
    parser.add_argument('--max-horizon', type=int, default=64)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--max-exec-time', type=int, default=100)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--uniform', action='store_true', default=False)
    args = parser.parse_args()

    dir, name = os.path.split(args.save_path)
    name, ext = os.path.splitext(name)
    os.makedirs(dir, exist_ok=True)
    zfile = zipfile.ZipFile(args.save_path, 'w')


    ######## DATA GENERATION #######

    trajectories = []  # (o, a, o', rewards, executed, duration, initiation_masks, info)

    env = make_antmaze(args.env, seed=0)
    options, initset = create_antmaze_options(args=args)
    initset_fn = make_initiation_set(initset, option_name=None, device='cpu')
    max_exec_time = args.max_exec_time
    
    options_desc = {i: str(o) for i, o in enumerate(options)}

    trajectories = Parallel(n_jobs=args.n_jobs)(delayed(collect_trajectory)(env, options, initset_fn, horizon=args.max_horizon, uniform=args.uniform) for i in tqdm(range(args.num_traj)))


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
        
        if n_executions > 0:
            avg_duration = np.array(duration)[idx][_executed == 1].mean()
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

            print(f'--------Option-{i}: {options[i]}---------')
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
            'max_horizon': args.max_horizon,
            'n_trajectories': len(trajectories),
            'n_samples': n_samples,
    }
    # dump in yaml
    
    with open(f'{dir}/{name}_stats.yaml', 'w') as f:
        yaml.dump(options_stats, f)


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