

import torch
import torch.nn.functional as F
import numpy as np
import scipy
import matplotlib.pyplot as plt

import argparse

from envs.pinball.pinball_gym import PinballEnvContinuous
from src.utils.printarr import printarr

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='pinball_discrete_1__loss.klconst_0__model.latentdim_2/mdp.pt')
parser.add_argument('--obs', type=str, default='full')
args = parser.parse_args()

mdp = torch.load(args.path)
s = mdp.reset()
next_s = s
trajs = []

if args.obs == 'pixel':
    env = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg', render_mode='rgb_array', width=25, height=25)
    s = env.reset()
    obs = env.render()
    goal = obs.transpose(2, 0, 1).mean(0, keepdims=True)[np.newaxis]
else:
    goal = np.array([0.5, 0.06, 0., 0.])

goal = mdp.encode(torch.from_numpy(goal).float()).numpy()

init_s = []
for i in range(10000):
    init_s.append(mdp.reset())

init_s = np.array(init_s)
# convex hull of init_s
ch = scipy.spatial.ConvexHull(init_s)

for traj in range(100):
    t = []
    s = mdp.reset()
    next_s = s
    for i in range(5):
        s = next_s
        initset = torch.sigmoid(mdp.initiation_set(torch.from_numpy(s))).numpy() > 0.8
        actions_avail = np.nonzero(initset)[0]
        sample = np.random.choice(len(actions_avail))
        a = actions_avail[sample]
        # a = 2
        next_s, r, d, info = mdp.step(a)

        t.append((s, a, r, next_s, d, info))
    trajs.append(t)

goals = np.random.randn(100, 2) * 0.1/3 + goal[np.newaxis]

# plt.scatter(goals[:, 0], goals[:, 1], c='g', s=100)
for t in trajs:
    s, a, r, next_s, d, info = list(zip(*t))
    s = np.array(s)
    next_s = np.array(next_s)
    plt.scatter(s[1:-1, 0], s[1:-1, 1], c='b', s=5)
    plt.scatter(s[0, 0], s[0, 1], c='r', s=20)
    plt.scatter(s[-1, 0], s[-1, 1], c='g', s=20)
    plt.plot(s[:, 0], s[:, 1], c='k')

for simplex in ch.simplices:
    plt.plot(init_s[simplex, 0], init_s[simplex, 1], 'k-')
plt.grid()

plt.savefig('traj.png')
plt.show()




