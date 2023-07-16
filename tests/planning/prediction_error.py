
import argparse
import torch
import numpy as np


from envs.pinball.pinball_gym import PinballEnvContinuous, PinballEnv
from envs.pinball.controllers_pinball import PinballGridOptions, create_position_options
from envs.env_options import EnvOptionWrapper
from envs.env_goal import EnvGoalWrapper

from src.utils.printarr import printarr
import matplotlib.pyplot as plt
from tqdm import tqdm


from joblib import Parallel, delayed


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--absmdp', type=str)
parser.add_argument('--n-jobs', type=int, default=1)
parser.add_argument('--save-path', type=str, default='.')
parser.add_argument('--discrete', action='store_true')
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


# load learned env
goal_reward = 1
env_sim = torch.load(args.absmdp)
env_sim = EnvGoalWrapper(env_sim, goal_fn=_grounding_goal_fn(env_sim.grounding, lambda x: x), goal_reward=goal_reward)

# create ground env
env = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg')
options = create_position_options(env) if not args.discrete else PinballGridOptions(env)
env = EnvOptionWrapper(options, env)


def action_mask(state):
    mask = np.array([o.initiation(state) for o in options])
    return mask


def random_selection(obs):
    _mask = action_mask(obs) + 1e-12
    _mask = _mask / _mask.sum(axis=-1, keepdims=True)
    n_actions = _mask.shape[-1]
    selection =  np.random.choice(n_actions, p=_mask)
    return selection


# sample real trajectory

def sample_trajectory(env, s0, actions, H):
    env.reset(s0)
    states = [s0]
    for a in actions:
        ret = env.step(a)
        states.append(ret[0])
    return states
        
def sample_real_trajectories(ground_env, N=100, H=50):
    s0 = ground_env.reset()
    actions = []
    states = [s0]
    for _ in range(H):
        a = random_selection(s0)
        next_s, _, _, _, _ = ground_env.step(a)
        actions.append(a)
        states.append(next_s)
    trajs = [np.array(states)]
    trajs = trajs + [sample_trajectory(ground_env, s0, actions, H) for _ in range(N-1)]
    return np.array(trajs), actions, s0

def compute_error_curve(N, H):
    ground_s, actions, s0 = sample_real_trajectories(env, N=N, H=H)
    encoded_s = env_sim.env.encode(torch.from_numpy(ground_s.reshape(-1, STATE_DIM))).reshape(N, H+1, -1).numpy()
    z0 = env_sim.env.encode(torch.from_numpy(s0)).numpy()
    latent_z = np.array([[z0] + sample_trajectory(env_sim, s0, actions, H=H)[1:] for _ in range(N)])
    error = ((encoded_s - latent_z) ** 2).sum(-1)
    errors.append(error)
    mse = error.mean(0)
    std = error.std(0)
    return mse, std, error
# sample trajs
N = 100
H = 50
STATE_DIM = 4

m = 8
M = m ** 2
plt.subplot(m, m, 1)
errors = []

results = Parallel(n_jobs=args.n_jobs)(delayed(compute_error_curve)(N=N, H=H) for _ in tqdm(range(M)))
for j, (mse, std, error) in enumerate(results): 
    errors.append(error)
    t = np.arange(H+1)
    printarr(mse, std, t)
    plt.subplot(m,m,j+1)
    plt.plot(t, mse)
    plt.fill_between(t, mse-std, mse + std, alpha=0.5)
    plt.grid()
plt.savefig(f'{args.save_path}/mse-common-init.pdf')
plt.figure()

errors = np.array(errors).reshape(-1, H+1)
mse = errors.mean(0)
std = errors.std(0)
plt.plot(t, mse)
plt.fill_between(t, mse-std, mse + std, alpha=0.5)
plt.grid()
plt.savefig(f'{args.save_path}/mse.png')

