import argparse
import torch

import numpy as np
from tqdm import tqdm

from functools import reduce
import zipfile, io
from collections import namedtuple

from src.absmdp.datasets import ObservationImgFile

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='data/pinball_simple_obs.zip')
    # parser.add_argument('--save-path', type=str, default=None)

    args = parser.parse_args()

    zfile = zipfile.ZipFile(args.dataset_path, 'a')


    # load dataset
    with zfile.open('transitions.pt', 'r') as transitions_f:
        dataset = torch.load(transitions_f) # trajectories.
    
    print(f'Dataset at {args.dataset_path} loaded')
    transition_samples = reduce(lambda x, acc: x + acc, dataset, [])
    o, option_n, next_o, rewards, executed, done, duration, initiation_masks, _, _ = zip(*transition_samples)
    rewards_per_action = {i: [] for i in range(max(option_n))}

    max_length = max(map(len, rewards))
    rewards = [r + [0] * (max_length - len(r)) for r in rewards]    

    options = np.array(option_n) 
    o = np.array(o)
    next_o = np.array(next_o)
    rewards = np.array(list(rewards))


    for i in tqdm(range(max(option_n)+1)):
        rewards_per_action[i] = (o[options == i], next_o[options == i], rewards[options == i])

    # dataset = list(zip(o, option_n, next_o, rewards, executed, duration, initiation_masks))

    # if args.save_path is None:
    #     print(f'Overwriting original file at {args.dataset_path}')
    #     args.save_path = args.dataset_path

    bs = io.BytesIO()
    torch.save(rewards_per_action, bs)
    zfile.writestr('rewards.pt', bs.getvalue())
    zfile.close()