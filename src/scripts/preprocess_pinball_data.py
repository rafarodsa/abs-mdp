import argparse
import torch

import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='data/pinball_no_obstacle.pt')
    parser.add_argument('--save-path', type=str, default='data/pinball_no_obstacle_rewards.pt')

    args = parser.parse_args()

    # load dataset
    dataset = torch.load(args.dataset_path)
    o, j, next_o, rewards, executed, duration, initiation_masks = zip(*dataset)

    rewards_per_action = {i: [] for i in range(max(j))}

    max_length = max(map(len, rewards))
    rewards = [r + [0] * (max_length - len(r)) for r in rewards]    

    options = np.array(j) 
    o = np.array(o)
    next_o = np.array(next_o)
    rewards = np.array(list(rewards))


    for i in range(max(j)+1):
        rewards_per_action[i] = (o[options == i], next_o[options == i], rewards[options == i])

    dataset = list(zip(o, j, next_o, rewards, executed, duration, initiation_masks))


    torch.save((dataset, rewards_per_action), args.save_path)
