import argparse
import torch

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='data/pinball_no_obstacle.pt')
    parser.add_argument('--save-path', type=str, default=None)

    args = parser.parse_args()

    # load dataset
    
    dataset = torch.load(args.dataset_path)
    print(f'Dataset at {args.dataset_path} loaded')
    o, j, next_o, rewards, executed, duration, initiation_masks = zip(*dataset)

    rewards_per_action = {i: [] for i in range(max(j))}

    max_length = max(map(len, rewards))
    rewards = [r + [0] * (max_length - len(r)) for r in rewards]    

    options = np.array(j) 
    o = np.array(o)
    next_o = np.array(next_o)
    rewards = np.array(list(rewards))


    for i in tqdm(range(max(j)+1)):
        rewards_per_action[i] = (o[options == i], next_o[options == i], rewards[options == i])

    dataset = list(zip(o, j, next_o, rewards, executed, duration, initiation_masks))

    if args.save_path is None:
        print(f'Overwriting original file at {args.dataset_path}')
        args.save_path = args.dataset_path

    torch.save((dataset, rewards_per_action), args.save_path)
