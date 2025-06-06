"""
    author: Rafael Rodriguez-Sanchez
    email: rrs@brown.edu
    date: 11 April 2023
"""
import os, io
from functools import reduce

import numpy as np
import torch, torchvision
import argparse
from tqdm import tqdm
import zipfile

from joblib import Parallel, delayed

from utils import collect_trajectory
from envs.pinball.pinball_gym import PinballDistractors as Pinball
from envs.pinball.controllers_pinball import create_position_controllers_v0 as OptionFactory
from envs.pinball.controllers_pinball import create_position_options as OptionFactory2

from PIL import Image

from src.absmdp.datasets import ObservationImgFile as ObservationFile

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

def load_mnist_distractors():
    # load MNIST from torchvision
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    dataiter = [d.expand(3,-1,-1,-1).squeeze().permute(2,1,0).numpy() for d, l in iter(trainloader)]
    return dataiter

def load_cifar10_distractors():
    # load CIFAR10 from torchvision
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    dataiter = [d.squeeze().permute(2,1,0).numpy() for d, l in iter(trainloader)]
    return dataiter

if __name__== "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    ######## Parameters
    np.set_printoptions(precision=3)
    configuration_file = "/Users/rrs/Desktop/abs-mdp/envs/pinball/configs/pinball_simple_single.cfg"
    num_traj = 100
    observation_type = 'o'

    ###### CMDLINE ARGUMENTS

    dataset_file_path = '/Users/rrs/Desktop/abs-mdp/data/'
    dataset_name = 'pinball_simple_obs.pt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default=dataset_file_path)
    parser.add_argument('--env-config', type=str, default=configuration_file)
    parser.add_argument('--num-traj', type=int, default=num_traj)
    parser.add_argument('--max-horizon', type=int, default=100)
    parser.add_argument('--observation', type=str, default=observation_type)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--max-exec-time', type=int, default=1000)
    parser.add_argument('--image-size', type=int, default=100)
    parser.add_argument('--distractors', type=str, default='mnist')


    args = parser.parse_args()

    dir, name = os.path.split(args.save_path)
    os.makedirs(dir, exist_ok=True)
    zfile = zipfile.ZipFile(args.save_path, 'w')
    ######## DATA GENERATION #######

    trajectories = []  # (o, a, o', rewards, executed, duration, initiation_masks, info)

    grid_size = args.image_size

    ### Load distractors
    if args.distractors == 'mnist':
        distractors = load_mnist_distractors()
    elif args.distractors == 'cifar10':
        distractors = load_cifar10_distractors()
    else:
        raise ValueError(f'Unknown distractors {args.distractors}')


    env = Pinball(config=args.env_config, 
                  width=grid_size, 
                  height=grid_size, 
                  render_mode='rgb_array',
                  distractors=distractors) 

    options = OptionFactory2(env)
    max_exec_time = args.max_exec_time
    
    options_desc = {i: str(o) for i, o in enumerate(options)}

    trajectories = Parallel(n_jobs=args.n_jobs)(delayed(collect_trajectory)(env, options, obs_type=args.observation, max_exec_time=max_exec_time, horizon=args.max_horizon) for i in tqdm(range(args.num_traj)))        
    
   
    if args.observation == 'pixel':
        trajectories = save_and_compress(trajectories, zfile)


    ##### Print dataset statistics
    transition_samples = reduce(lambda x, acc: x + acc, trajectories, [])
    n_samples = len(transition_samples)


    o, option_n, next_o, rewards, done, executed, duration, initiation_mask, info, p0 = zip(*transition_samples)
    stats = {}
    _r = np.array(list(map(lambda x: sum(x), rewards)))
    _r_len = list(map(len, rewards))
    
    s = np.array(list(map(lambda x: x['state'], info)))
    next_s = np.array(list(map(lambda x: x['next_state'], info)))

    
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
            'prob_executions': n_executions,
            'avg_duration': avg_duration,
            'avg_reward': option_rewards.mean(),
            'min_reward': option_rewards.min(),
            'max_reward': option_rewards.max(),
            'state_change_min': state_change_min,
            'state_change_max': state_change_max,
            'state_change_mean': state_change_mean,
            'state_change_std': state_change_std
        }

        print(f'--------Option-{i}: {options_desc[i]}---------')
        print(f"Executed {n_executions}/{n_samples} times")
        print(f"Average duration {avg_duration}")
        print(f"Average reward {option_rewards.mean()}. Max reward: {option_rewards.max()}. Min reward: {option_rewards.min()}")
        print(f"Reward length: mean: {np.array(_r_len)[idx].mean()}, max: {np.array(_r_len)[idx].max()}, min: {np.array(_r_len)[idx].min()}")
        print(f"State change: mean: {state_change_mean}, max: {state_change_max}, min: {state_change_min}, std {state_change_std}")
    
    debug = {
        'latent_states': info,
        'options': options_desc,
        'stats': stats
    }

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