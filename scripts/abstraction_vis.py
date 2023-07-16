

import argparse
import torch
import torch.nn as nn
import numpy as np


from envs.pinball.pinball_gym import PinballEnvContinuous, PinballEnv
from envs.pinball.controllers_pinball import PinballGridOptions, create_position_options
from envs.env_options import EnvOptionWrapper
from envs.env_goal import EnvGoalWrapper

from src.utils.printarr import printarr

from tqdm import tqdm


from joblib import Parallel, delayed

import seaborn as sns
import matplotlib.pyplot as plt


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--absmdp', type=str)
parser.add_argument('--n-jobs', type=int, default=1)
parser.add_argument('--save-path', type=str, default='.')
parser.add_argument('--n-samples', type=int, default=1000)

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
n_actions = 4
obs_dim = 4
goal = [0.55, 0.06, 0., 0.]
goal = compute_nearest_pos(np.array(goal[:2]))
##################

# load learned env
goal_reward = 1
env_sim = torch.load(args.absmdp)
env_sim = EnvGoalWrapper(env_sim, goal_fn=_grounding_goal_fn(env_sim.grounding, lambda x: x), goal_reward=goal_reward)

# create ground env
env_p = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg')
options = create_position_options(env_p)
env = EnvOptionWrapper(options, env_p)

# load agent
latent_dim = env_sim.env.obs_dim
states = env_p.sample_init_states(N_SAMPLES).astype(np.float32)
encoded_s = env_sim.env.encode(torch.from_numpy(states))
actions_semantics = ['-Y', '+Y', '-X', '+X']

states_norm = (states - states.min(0, keepdims=True)) / (states.max(0, keepdims=True) - states.min(0, keepdims=0))
encoded_s_norm = (encoded_s - encoded_s.min(0, keepdims=True)[0]) / (encoded_s.max(0, keepdims=True)[0] - encoded_s.min(0, keepdims=True)[0])
printarr(encoded_s, states, latent_dim, obs_dim, states_norm, encoded_s_norm)

sns.set_theme(style="white")
rs = np.random.RandomState(50)

# Set up the matplotlib figure
f, axes = plt.subplots(obs_dim, latent_dim, figsize=(9, 9), sharex=True, sharey=True)
state_vars = ['x', 'y', 'x_dot', 'y_dot']

# Rotate the starting point around the cubehelix hue circle
for ax, s, p in zip(axes.flat, np.linspace(0, 3, latent_dim * obs_dim), range(latent_dim * obs_dim)):
    # Create a cubehelix colormap to use with kdeplot
    cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)
    i, j = p % latent_dim, p // latent_dim
    x = encoded_s_norm[:, i]
    y = states_norm[:, j]
    sns.kdeplot(
        x=x, y=y,
        clip=(0,1),
        cmap=cmap, 
        fill=True, cut=10,
        thresh=0, levels=15,
        ax=ax,
    )
    ax.set_xlabel(f'z_{i}')
    ax.set_ylabel(f'{state_vars[j]}')
    print(f'{p} = (x_{i}, y_{j})')
    # ax.set_axis_off()

# ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
# f.subplots_adjust(0, 0, 1, 1, .08, .08)
plt.savefig(f'{args.save_path}/joint-prob.png')