import argparse
import torch
import numpy as np

from src.absmdp.discrete_tpc_critic import DiscreteInfoNCEAbstraction as DiscreteAbstraction
from src.absmdp.mdp import AbstractMDP
import matplotlib.pyplot as plt

from src.utils.printarr import printarr

parser = argparse.ArgumentParser()
parser.add_argument('--save-path', type=str, default='.')
parser.add_argument('--absmdp', type=str)
parser.add_argument('--discrete', action='store_true')
parser.add_argument('--n-traj', type=int, default=100)
parser.add_argument('--length', type=int, default=10)
parser.add_argument('--initset-threshold', type=float, default=0.82)
args = parser.parse_args()

mdp = torch.load(args.absmdp)


trajs = []
n_actions = int(len(mdp.get_actions()))
for _ in range(args.n_traj):
    s = mdp.reset()
    t = []
    for _ in range(args.length):
        # action selection
        available_action_mask = (torch.sigmoid(mdp.initiation_set(torch.from_numpy(s))) > args.initset_threshold).float().numpy()
        sampled_action = np.random.choice(n_actions, p=available_action_mask/available_action_mask.sum(keepdims=True))
        # simulate
        next_s, r, done, info = mdp.step(sampled_action)
        # printarr(next_s, s)
        t.append((s, r, next_s, done, info))
        s = next_s
    trajs.append(t)
    

trajs = [list(zip(*t)) for t in trajs]

for s, _, next_s, _, _ in trajs:
    s = np.array(s)
    next_s = np.array(next_s)
    s = np.concatenate([s, next_s[-1:-2]], axis=0)
    # printarr(s, next_s)
    plt.scatter(s[1:-1, 0], s[1:-1, 1], s=5, c='k')
    plt.scatter(s[0, 0], s[0, 1], s=15, c='r')
    plt.scatter(s[-1, 0], s[-1, 1], s=15, c='g')
    plt.plot(s[:, 0], s[:, 1], c='k')
plt.grid()
plt.savefig(f'{args.save_path}/random_walks.png')

