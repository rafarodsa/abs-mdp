

import argparse
import torch
import torch.nn as nn
import numpy as np


from envs.pinball.pinball_gym import PinballEnvContinuous
from envs.pinball.controllers_pinball import PinballGridOptions, create_position_options
from envs.env_options import EnvOptionWrapper
from envs.env_goal import EnvGoalWrapper

from src.utils.printarr import printarr
import matplotlib.pyplot as plt
from tqdm import tqdm


from joblib import Parallel, delayed

import os

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--absmdp', type=str)
parser.add_argument('--n-jobs', type=int, default=1)
parser.add_argument('--save-path', type=str, default='.')
parser.add_argument('--dqn-agent', type=str)
parser.add_argument('--n-samples', type=int, default=1000)
parser.add_argument('--horizon', type=int, default=10)

args = parser.parse_args()

def compute_nearest_pos(position, n_pos=20, min_pos=0.05, max_pos=0.95):
    # batch = position.shape[0] if len(position.shape) > 1 else 1
    step = (max_pos - min_pos) / n_pos
    pos_ = (position - min_pos) / (max_pos-min_pos) * n_pos
    min_i, max_i = np.floor(pos_), np.ceil(pos_) # closest grid position to move
    min = np.vstack([min_i, max_i])
    x, y = np.meshgrid(min[:,0], min[:, 1])
    pts = np.stack([x.reshape(-1), y.reshape(-1)], axis=1)
    distances =  np.linalg.norm(pts - pos_[np.newaxis], axis=-1)
    _min_dis = distances.argmax()
    goal = pts[_min_dis] * step + min_pos

    goal = goal.clip(min_pos, max_pos)
    return goal


def _grounding_goal_fn(grounding, phi, goal=[0.55, 0.06, 0., 0.]):
    goal = compute_nearest_pos(np.array(goal[:2]))
    goal = np.concatenate([goal, [0., 0.]])
    goal = torch.Tensor(goal).unsqueeze(0)
    def _goal(state):
        _goal = torch.Tensor(goal)
        # printarr(_goal, state)
        s = phi(torch.from_numpy(state).unsqueeze(0))
        v = torch.tanh(grounding(_goal, s))
        return v > 0.7

    return _goal

def action_mask(state):
    b_dim = state.shape[:-1]
    if len(b_dim) > 0:
        mask = np.array([[o.initiation(state[i]) for o in options] for i in range(b_dim[0])])
    else:
        mask = np.array([o.initiation(state) for o in options])
    return mask

def action_mask_learned(encoded_s):
    return (torch.sigmoid(env_sim.env.init_classifier(encoded_s)) > 0.7).float()



##################

N_SAMPLES = args.n_samples
STEP_SIZE = 1/25
n_actions = 4
latent_dim = 2
goal = [0.55, 0.06, 0., 0.]
goal = compute_nearest_pos(np.array(goal[:2]))
##################

os.makedirs(args.save_path, exist_ok=True)

# load learned env
goal_reward = 1
# env_sim = torch.load(args.absmdp)
# env_sim = EnvGoalWrapper(env_sim, goal_fn=_grounding_goal_fn(env_sim.grounding, lambda x: x), goal_reward=goal_reward)

# create ground env
env_p = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg')
options = create_position_options(env_p, translation_distance=STEP_SIZE)
env = EnvOptionWrapper(options, env_p)
# env = EnvGoalWrapper(env, goal_fn=_grounding_goal_fn(env_sim.env.grounding, env_sim.env.encode), goal_reward=goal_reward)

obstacles = [np.array(o.points + [o.points[0]]) for o in env_p.get_obstacles()]
def plot_obstacles(ax=None):
    ax = ax if ax is not None else plt
    for o in obstacles:
        ax.plot(o[:, 0], o[:, 1], c='k')


states = env_p.sample_init_states(N_SAMPLES).astype(np.float32)
# x_pos = np.logical_and(states[:, 0] > 0.6, states[:, 0] < 1)
# y_pos = np.logical_and(states[:, 1] > 0., states[:, 1] < 0.2)
# states = states[np.logical_and(x_pos, y_pos), :]


length = args.horizon
printarr(states)
plt.figure()
plot_obstacles()
traj = []
for i in tqdm(range(states.shape[0])):
    s = states[i]
    traj = [s]
    env.reset(s) 
    for j in range(length):
        initset = action_mask(s)
        avail_actions = initset.sum(-1)
        if avail_actions > 0:
            a = np.random.choice(a, p=initset/avail_actions)

        next_s, _, _, _, _ = env.step(a)
        traj.append(next_s)
    traj = np.array(traj)
    plt.plot(traj[:, 0], traj[:, 1], linewidth=0.5, c='k')
    plt.scatter(traj[0, 0], traj[0, 1], c='r', s=1)
    plt.scatter(traj[-1, 0], traj[-1, 1], c='g', s=3)
    plt.scatter(traj[1:-1, 0], traj[1:-1, 1], c='k', s=0.5)
    
# plt.grid()
plt.savefig('pinball_traj_debug.png')


