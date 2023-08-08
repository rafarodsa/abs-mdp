

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
import matplotlib as mpl
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
parser.add_argument('--ground', action='store_true')

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
    # goal = compute_nearest_pos(np.array(goal[:2]))
    # goal = np.concatenate([goal, [0., 0.]])
    goal = torch.Tensor(goal).unsqueeze(0)
    energy_goal = torch.tanh(grounding(goal, phi(goal)))
    eps = 0.01
    def _goal(state):
        _goal = torch.Tensor(goal)
        # printarr(_goal, state)
        s = phi(torch.from_numpy(state).unsqueeze(0))
        v = torch.tanh(grounding(_goal, s))
        return v > energy_goal-eps

    return _goal

def action_mask(state):
    b_dim = state.shape[:-1]
    if len(b_dim) > 0:
        mask = np.array([[o.initiation(state[i]) for o in options] for i in range(b_dim[0])])
    else:
        mask = np.array([o.initiation(state) for o in options])
    return mask

def action_mask_learned(encoded_s):
    return (torch.sigmoid(env_sim.env.init_classifier(encoded_s)) > 0.5).float()



##################

N_SAMPLES = args.n_samples
n_actions = 4
# goal = [0.55, 0.06, 0., 0.]
goal = torch.Tensor([0.6, 0.5, 0., 0.])
# goal = compute_nearest_pos(np.array(goal[:2]))
##################

os.makedirs(args.save_path, exist_ok=True)

# load learned env
goal_reward = 1
env_sim = torch.load(args.absmdp)
env_sim = EnvGoalWrapper(env_sim, goal_fn=_grounding_goal_fn(env_sim.grounding, lambda x: x), goal_reward=goal_reward)

# create ground env
env_p = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg')
options = create_position_options(env_p)
env = EnvOptionWrapper(options, env_p)
env = EnvGoalWrapper(env, goal_fn=_grounding_goal_fn(env_sim.env.grounding, env_sim.env.encode), goal_reward=goal_reward)

obstacles = [np.array(o.points + [o.points[0]]) for o in env_p.get_obstacles()]
def plot_obstacles(ax=None):
    ax = ax if ax is not None else plt
    for o in obstacles:
        ax.plot(o[:, 0], o[:, 1], c='k')

# load agent
latent_dim = env_sim.env.latent_dim

q_func = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

q_func.load_state_dict(torch.load(args.dqn_agent))

states = env_p.sample_init_states(N_SAMPLES).astype(np.float32)
encoded_s = env_sim.env.encode(torch.from_numpy(states))
actions_semantics = ['-Y', '+Y', '-X', '+X']

with torch.no_grad():
    q_values = q_func(encoded_s) if not args.ground else q_func(torch.from_numpy(states))
q_values = q_values.numpy()
printarr(states, encoded_s, q_values, goal)

# cmap = mpl.cm.viridis

### plot q-values

plt.figure()
action_mask_t = action_mask(states)
action_mask_l = action_mask_learned(encoded_s)
for a in range(n_actions):
    plt.subplot(2,4,a+1)
    plt.scatter(states[:, 0], states[:, 1], c=action_mask_t[:, a], s=5)
    plt.scatter(goal[0], goal[1], s=10, c='r')
    plot_obstacles()
    plt.colorbar()
    plt.grid()
    plt.title(f'Action {actions_semantics[a]}')

# plt.savefig(f'{args.save_path}/action_mask_ground.png')

for a in range(n_actions):
    plt.subplot(2,4,4+a+1)
    plt.scatter(states[:, 0], states[:, 1], c=action_mask_l[:, a], s=5)
    plt.scatter(goal[0], goal[1], s=10, c='r')
    plot_obstacles()
    plt.colorbar()
    plt.grid()
    plt.title(f'Action {actions_semantics[a]}')

plt.savefig(f'{args.save_path}/action_mask.png')

plt.figure()
plt.scatter(states[:, 0], states[:, 1], c=action_mask_l.sum(-1), s=5)
plot_obstacles()
plt.colorbar()
plt.grid()
plt.savefig(f'{args.save_path}/action_mask_agg.png')

q_min = q_values.min()
masked_values_t = torch.from_numpy((1-action_mask_t)) * q_min
q_masked_values = torch.from_numpy((1-action_mask_t)) * -1e12 + q_values
printarr(masked_values_t, q_min, action_mask_t, q_values)
q_masked_values_plot = q_values * action_mask_t + masked_values_t.numpy()
for a in range(n_actions):
    plt.subplot(2,2,a+1)
    plt.scatter(states[:, 0], states[:, 1], c=q_masked_values_plot[:, a], s=5)
    plt.scatter(goal[0], goal[1], s=10, c='r')
    plot_obstacles()
    plt.colorbar()
    plt.grid()
    plt.title(f'Action {actions_semantics[a]}')

plt.savefig(f'{args.save_path}/q-heatmap.png')

q_max = q_masked_values_plot.max(-1)

printarr(q_max)
plt.figure()
plt.scatter(states[:, 0], states[:, 1], c=q_max, s=5)
plt.scatter(goal[0], goal[1], s=10, c='r')
plot_obstacles()
plt.colorbar()
plt.grid()
plt.savefig(f'{args.save_path}/q-greedy-heatmap.png')

### greedy policy
# action_mask_t  = action_mask(states)
# masked_values_t = torch.from_numpy((1-action_mask_t)) * -1e2
# q_masked_values = masked_values_t + q_values
greedy_policy = q_masked_values.argmax(-1)

action_vel_x = np.array([0., 0., -1., 1.]) * 1/15
action_vel_y = np.array([-1., 1., 0., 0.]) * 1/15

action_vel_x = action_vel_x[greedy_policy]
action_vel_y = action_vel_y[greedy_policy]


printarr(masked_values_t, greedy_policy, action_vel_y, action_vel_x)

plt.figure()
f, axes = plt.subplots(1, 3, sharex=True, sharey=True)

ax = axes.flat[0]
ax.quiver(states[:, 0], states[:, 1], action_vel_x, action_vel_y, greedy_policy)
ax.scatter(goal[0], goal[1], s=10, c='r')
plot_obstacles(ax)
ax.grid()
ax.set_title('w/True initset')

# plt.savefig(f'{args.save_path}/q-greedy-policy.pdf')


### greedy policy with learned initset

masked_values_l = (1-action_mask_l) * -1e2
q_masked_values = masked_values_l + q_values
greedy_policy = q_masked_values.argmax(-1)

action_vel_x = np.array([0., 0., -1., 1.]) * 1/15
action_vel_y = np.array([-1., 1., 0., 0.]) * 1/15
action_vel_x = action_vel_x[greedy_policy]
action_vel_y = action_vel_y[greedy_policy]
printarr(masked_values_l, greedy_policy, action_vel_y, action_vel_x)

ax = axes.flat[1]
ax.quiver(states[:, 0], states[:, 1], action_vel_x, action_vel_y, greedy_policy)
ax.scatter(goal[0], goal[1], s=10, c='r')
plot_obstacles(ax)
ax.grid()
ax.set_title('w/Learned initset')

## plot greedy unmasked
greedy_policy = q_values.argmax(-1)
action_vel_x = np.array([0., 0., -1., 1.]) * 1/15
action_vel_y = np.array([-1., 1., 0., 0.]) * 1/15
action_vel_x = action_vel_x[greedy_policy]
action_vel_y = action_vel_y[greedy_policy]

ax = axes.flat[2]
ax.quiver(states[:, 0], states[:, 1], action_vel_x, action_vel_y, greedy_policy)
ax.scatter(goal[0], goal[1], s=10, c='r')
plot_obstacles(ax)
ax.grid()
ax.set_title('Policy Unmasked')

plt.savefig(f'{args.save_path}/q-greedy-policy.pdf')

action_mask_t = action_mask_t.astype(np.float32)
action_mask_l = action_mask_l.numpy()
true_pred = (action_mask_l == action_mask_t)
printarr(action_mask_t, action_mask_l, true_pred)
initset_acc = true_pred.mean(0)
tpr = ((action_mask_l * action_mask_t) == 1).sum(0) / (action_mask_l == 1).sum(0)
fpr = (action_mask_l * (1-action_mask_t) == 1).sum(0)/ (action_mask_l == 0).sum(0)
print(initset_acc, tpr, fpr)

def softmax(a, dim=-1, beta=1.):
    x = a * beta 
    return torch.exp(x - torch.logsumexp(x, dim=dim))

#### Plot Trajectories.
def sample_real_trajectory(env, encoder, q_func, initiation_mask, max_len=50, greedy=True, s0=None, epsilon=0.0):
    s = env.reset(s0)
    done = False
    z = encoder
    timestep = 0
    traj = []
    while not done and timestep < max_len:
        # act greedily.
        q_values = q_func(z(torch.from_numpy(s))) 
        initset = initiation_mask(s)
        q_masked_values = (1-initset) * -1e12 + q_values
        if greedy: 
            if np.random.random() > epsilon:
                action = q_masked_values.argmax(-1) # act greedily
            else:
                avail_actions = np.nonzero(initset)
                action = np.random.choice(avail_actions[0])
        else:
            probs = softmax(q_masked_values, dim=-1, beta=10).detach().numpy()
            action = np.random.choice(n_actions, p=probs)
            # print(probs, initset, action, q_values)
        next_s, r, done, info = env.step(action)
        # update
        traj.append((s, action, next_s))
        s = next_s 
        timestep += 1
        
    return traj


def plot_trajectory(s, ax=None):
    ax = plt if ax is None else ax
    ax.plot(s[:, 0], s[:, 1], c='k', linewidth=0.5)
    ax.scatter(s[0, 0], s[0, 1], s=1, c='b')
    ax.scatter(s[1:-1, 0], s[1:-1, 1], s=0.5, c='k')
    ax.scatter(s[-1, 0], s[-1, 1], s=2, c='g')


### Plot trajectories!

N_TRAJS = 15
z = env_sim.env.encode
start_states = env_p.sample_init_states(N_TRAJS)

GREEDY = True

z = z if not args.ground else lambda s: s
# Generate Trajectories with ground initset
for i in range(N_TRAJS):
    traj = sample_real_trajectory(env, z, q_func, lambda s : torch.from_numpy(action_mask(s).astype(np.float32)), greedy=GREEDY, s0=start_states[i])
    s, a, next_s = zip(*traj)
    s = np.array(s + (next_s[-1],))
    printarr(s)

    ax = axes.flat[0]
    plot_trajectory(s, ax)
    plot_obstacles(ax)
    ax.set_title('w/Ground Initset')
    
# Generate Trajectories with learned initset
for i in range(N_TRAJS):
    traj = sample_real_trajectory(env, z, q_func, lambda s: action_mask_learned(z(torch.from_numpy(s))), greedy=GREEDY, s0=start_states[i])
    s, a, next_s = zip(*traj)
    s = np.array(s + (next_s[-1],))
    printarr(s)

    ax = axes.flat[1]
    plot_trajectory(s, ax)
    plot_obstacles(ax)
    ax.set_title('w/Learned initset')

# Generate Trajectories no mask
for i in range(N_TRAJS):
    traj = sample_real_trajectory(env, z, q_func, lambda s: torch.ones(n_actions), greedy=GREEDY, s0=start_states[i])
    s, a, next_s = zip(*traj)
    s = np.array(s + (next_s[-1],))
    printarr(s)

    ax = axes.flat[2]
    plot_trajectory(s, ax)
    plot_obstacles(ax)
    ax.set_title('No initset')
    

plt.savefig(f'{args.save_path}/trajs-greedy.pdf')

# f, axes = plt.subplots(1, 3, sharex=True, sharey=True)
# GREEDY = False
# # Generate Trajectories with ground initset
# for i in range(N_TRAJS):
#     traj = sample_real_trajectory(env, z, q_func, lambda s : torch.from_numpy(action_mask(s).astype(np.float32)), greedy=GREEDY, s0=start_states[i])
#     s, a, next_s = zip(*traj)
#     s = np.array(s + (next_s[-1],))
#     printarr(s)

#     ax = axes.flat[0]
#     plot_trajectory(s, ax)
    
# # Generate Trajectories with learned initset
# for i in range(N_TRAJS):
#     traj = sample_real_trajectory(env, z, q_func, lambda s: action_mask_learned(z(torch.from_numpy(s))), greedy=GREEDY, s0=start_states[i])
#     s, a, next_s = zip(*traj)
#     s = np.array(s + (next_s[-1],))
#     printarr(s)

#     ax = axes.flat[1]
#     plot_trajectory(s, ax)
    

# # Generate Trajectories no mask
# for i in range(N_TRAJS):
#     traj = sample_real_trajectory(env, z, q_func, lambda s: torch.ones(n_actions), greedy=GREEDY, s0=start_states[i])
#     s, a, next_s = zip(*traj)
#     s = np.array(s + (next_s[-1],))
#     printarr(s)

#     ax = axes.flat[2]
#     plot_trajectory(s, ax)

# plt.savefig(f'{args.save_path}/trajs-softmax.pdf')