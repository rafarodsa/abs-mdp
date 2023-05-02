

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

path = 'pinball_continuous_09__loss.klconst_0__model.latentdim_2__loss.transitionconst_1/mdp.pt'

mdp = torch.load(path)
s = mdp.reset()
next_s = s
trajs = []
goal = np.array([0.5, 0.06, 0., 0.])
goal = mdp.encode(torch.from_numpy(goal).float()).numpy()
for traj in range(100):
    t = []
    s = mdp.reset()
    next_s = s
    for i in range(1000):
        s = next_s
        initset = torch.sigmoid(mdp.initiation_set(torch.from_numpy(s))).numpy() > 0.5
        actions_avail = np.nonzero(initset)[0]
        
        sample = np.random.choice(len(actions_avail))
        a = actions_avail[sample]
        # a = 1
        next_s, r, d, info = mdp.step(a)

        t.append((s, a, r, next_s, d, info))
    trajs.append(t)

goals = np.random.randn(100, 2) * 0.1/3 + goal[np.newaxis]

plt.scatter(goals[:, 0], goals[:, 1], c='g', s=100)
for t in trajs:
    s, a, r, next_s, d, info = list(zip(*t))
    s = np.array(s)
    next_s = np.array(next_s)
    plt.scatter(s[:, 0], s[:, 1], c='b', s=5)
    plt.plot(s[:, 0], s[:, 1], c='k')

plt.grid()
plt.show()





