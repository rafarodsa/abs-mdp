import argparse

import numpy as np
import torch
import torch.nn as nn

import pfrl
from pfrl import agents, explorers, experiments
from pfrl import nn as pnn
from pfrl import replay_buffers, utils
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead, DuelingDQN
from pfrl.wrappers import atari_wrappers

from envs.pinball.pinball_gym import PinballEnvContinuous, PinballEnv
from envs.pinball.controllers_pinball import PinballGridOptions, create_position_options
from envs.env_options import EnvOptionWrapper
from envs.env_goal import EnvGoalWrapper
from src.agents.abstract_ddqn import AbstractDoubleDQN as AbstractDDQN, AbstractLinearDecayEpsilonGreedy
from src.agents.abstract_ddqn import AbstractDDQNGrounded

from src.agents.train_agent import train_agent_with_evaluation
from src.agents.evaluator import eval_performance
from src.utils.printarr import printarr

class SingleSharedBias(nn.Module):
    """Single shared bias used in the Double DQN paper.

    You can add this link after a Linear layer with nobias=True to implement a
    Linear layer with a single shared bias parameter.

    See http://arxiv.org/abs/1509.06461.
    """

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([1], dtype=torch.float32))

    def __call__(self, x):
        return x + self.bias.expand_as(x)


def parse_arch(arch, n_actions, state_dim):
    if arch == 'mlp':
        return nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            DiscreteActionValueHead()
        )
    else:
        raise RuntimeError("Not supported architecture: {}".format(arch))


def parse_agent(agent):
    return {"DQN": agents.DQN, "DoubleDQN": agents.DoubleDQN, "PAL": agents.PAL}[agent]


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


def check_init_state(state, goal, step_size, tol):
    n_steps = np.abs(state[:2]-goal[:2])/step_size
    min_n = np.floor(n_steps)
    max_n = np.ceil(n_steps)
    return np.all(np.logical_or(np.abs(min_n-n_steps) <= tol, np.abs(max_n-n_steps) <= tol)) and state[1] < 0.1

def init_state_sampler(goal, step_size=1/15, tol=0.015):
    config = 'envs/pinball/configs/pinball_simple_single.cfg'
    pinball_env = PinballEnvContinuous(config)
    
    def __sampler():
        accepted = False
        while not accepted:
            s = pinball_env.sample_initial_positions(1)[0].astype(np.float32)
            accepted = np.linalg.norm(goal[:2]-s[:2]) >= 0.2
        return s
    return __sampler

# goal
def goal_fn(phi, goal=[0.2, 0.9, 0., 0.], goal_tol=0.01):
    goal = phi(torch.Tensor(goal)).numpy()
    # goal = compute_nearest_pos(goal)
    def _goal(state):
        distance =  np.linalg.norm(state[:2] - goal)
        # print(distance, state[:2], goal)
        return distance <= goal_tol
    return _goal

def grounding_goal_fn(grounding, phi, goal=[0.2, 0.9, 0., 0.]):
    # goal = compute_nearest_pos(np.array(goal[:2]))
    print(goal)
    # goal = np.concatenate([goal, [0., 0.]])
    goal = torch.Tensor(goal).unsqueeze(0)
    def _goal(state):
        _goal = torch.Tensor(goal)
        # printarr(_goal, state)
        s = torch.from_numpy(state).unsqueeze(0)
        v = torch.tanh(grounding(_goal, s))
        return v > 0.7

    return _goal

goal = [0.5, 0.5, 0., 0.]

def make_env(test=False, test_seed=127, train_seed=255, args=None):
        goal_reward = 1
        goal_tol = 5
        
        print('================ CREATE ENVIRONMENT ================', goal_reward)
        if not test:
            env = torch.load(args.absmdp)
            # env = EnvGoalWrapper(env, goal_fn=goal_fn(env.encoder, goal_tol=goal_tol), goal_reward=goal_reward)
            env = EnvGoalWrapper(env, goal_fn=grounding_goal_fn(env.grounding, env.encoder, goal=goal), goal_reward=goal_reward, init_state_sampler=init_state_sampler(goal))
            env.seed(train_seed)
        else:
            env = torch.load(args.absmdp)
            # env = EnvGoalWrapper(env, goal_fn=goal_fn(env.encoder, goal_tol=goal_tol), goal_reward=goal_reward)
            env = EnvGoalWrapper(env, goal_fn=grounding_goal_fn(env.grounding, env.encoder, goal=goal), goal_reward=goal_reward, init_state_sampler=init_state_sampler(goal))
            env.seed(test_seed)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

def make_ground_env(test=False, test_seed=0, train_seed=1, args=None):
    goal_reward = 1
    goal_tol = 1/10
    enc = lambda x: x[..., :2]
    print('================ CREATE GROUND ENVIRONMENT ================', goal_reward)
    
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


    if not test:
        env = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg')
        if args.render:
            env.render_mode = 'human'
        # print('grid options')
        # options = PinballGridOptions(env)
        options = create_position_options(env)
        env_sim = torch.load(args.absmdp)
        env = EnvOptionWrapper(options, env)
        # env = EnvGoalWrapper(env, goal_fn=goal_fn(enc, goal_tol=goal_tol), goal_reward=goal_reward)
        env = EnvGoalWrapper(env, goal_fn=_grounding_goal_fn(env_sim.grounding, env_sim.encoder, goal=goal), goal_reward=goal_reward, init_state_sampler=init_state_sampler(goal))
        env.seed(train_seed)
    else:
        # evaluation in real env
        env = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg')
        if args.render:
            env.render_mode = 'human'
        print('cont options')
        options = create_position_options(env)
        # print('grid options')
        # options = PinballGridOptions(env)
        env = EnvOptionWrapper(options, env)
        env_sim = torch.load(args.absmdp)
        # env = EnvGoalWrapper(env, goal_fn=goal_fn(enc, goal_tol=goal_tol), goal_reward=goal_reward)
        env = EnvGoalWrapper(env, goal_fn=_grounding_goal_fn(env_sim.grounding, env_sim.encoder, goal=goal), goal_reward=goal_reward, init_state_sampler=init_state_sampler(goal))
        env.seed(test_seed)
    if args.monitor:
        env = pfrl.wrappers.Monitor(
            env, args.outdir, mode="evaluation" if test else "training"
        )
    if args.render:
        env = pfrl.wrappers.Render(env)
    return env


class RandomAgent(pfrl.agent.Agent):
    def __init__(self, action_selection, encoder):
        self.action_selection = action_selection
        self.encoder = encoder
    def act(self, obs):
        z = self.encoder(torch.from_numpy(obs))
        return self.action_selection(z)
    def load(self, dirname):
        pass
    def get_statistics(self):
        pass
    def observe(self, *args):
        pass
    def save(self, dirname):
        pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--absmdp',
        type=str,
        default='./mdp.pt'
    )

    parser.add_argument(
        '--absgroundmdp',
        action='store_true',
        default=False
    )

    parser.add_argument(
        "--env",
        type=str,
        default="BreakoutNoFrameskip-v4",
        help="OpenAI Atari domain to perform algorithm on.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--seed", type=int, default=31, help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument(
        "--final-exploration-frames",
        type=int,
        default=4*10**5,
        help="Timesteps after which we stop " + "annealing exploration rate",
    )
    parser.add_argument(
        "--final-epsilon",
        type=float,
        default=0.01,
        help="Final value of epsilon during training.",
    )
    parser.add_argument(
        "--eval-epsilon",
        type=float,
        default=0.001,
        help="Exploration epsilon used during eval episodes.",
    )
    parser.add_argument("--noisy-net-sigma", type=float, default=None)

    parser.add_argument(
        "--steps",
        type=int,
        default=10**6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,  
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=1000,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1000,
        help="Frequency (in timesteps) at which " + "the target network is updated.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Frequency (in timesteps) of evaluation phase.",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=5,
        help="Frequency (in timesteps) of network updates.",
    )
    parser.add_argument("--eval-n-runs", type=int, default=20)
    parser.add_argument("--no-clip-delta", dest="clip_delta", action="store_false")
    parser.add_argument("--num-step-return", type=int, default=1)
    parser.set_defaults(clip_delta=True)
    parser.add_argument(
        "--agent", type=str, default="DoubleDQN", choices=["DQN", "DoubleDQN", "PAL"]
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--prioritized",
        action="store_true",
        default=False,
        help="Use prioritized experience replay.",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=None,
        help="Frequency at which agents are stored.",
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="mlp",
        choices=["nature", "nips", "dueling", "doubledqn", "mlp"],
        help="Network architecture to use.",
    )
    
    parser.add_argument(
        '--eval-random-agent',
        action='store_true'
    )
    parser.add_argument(
        '--init-thresh',
        type=float,
        default=0.7
    )
    
    parser.add_argument(
        '--finetune',
        action='store_true'
    )

    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2**31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir, make_backup=False)
    print("Output files are saved in {}".format(args.outdir))
    
    if args.finetune:
        env = make_ground_env(test=False, train_seed=train_seed, test_seed=test_seed, args=args)
        eval_env = make_ground_env(test=True, train_seed=train_seed, test_seed=test_seed, args=args)
        env_sim = make_env(test=False, train_seed=train_seed, test_seed=test_seed, args=args)
        state_dim = env_sim.env.latent_dim
        encoder = env_sim.env.encode
    elif (not args.absgroundmdp and not args.demo):
        env = make_env(test=False, train_seed=train_seed, test_seed=test_seed, args=args)
        eval_env = make_env(test=True, train_seed=train_seed, test_seed=test_seed, args=args)
        state_dim = env.env.latent_dim
        encoder = env.env.encode
    elif args.demo:
        env = make_env(test=False, train_seed=train_seed, test_seed=test_seed, args=args)
        eval_env = make_ground_env(test=True, train_seed=train_seed, test_seed=test_seed, args=args)
        state_dim = env.env.latent_dim
        encoder = env.env.encode
    elif args.absgroundmdp:
        print('absgroundmdp')
        env = make_ground_env(test=False, train_seed=train_seed, test_seed=test_seed, args=args)
        eval_env = make_ground_env(test=True, train_seed=train_seed, test_seed=test_seed, args=args)
        state_dim = 4
        

    n_actions = eval_env.action_space.n
    q_func = parse_arch(args.arch, n_actions, state_dim=state_dim)


    
    if not args.absgroundmdp and not args.finetune and not args.demo:
        init = env.env.init_classifier.to(f'cuda:{args.gpu}') if args.gpu >= 0 else env.env.initiation_set
        def action_mask(s):
            probs = torch.sigmoid(init(s.float()))
            return (probs > args.init_thresh).float()
    else:
        options = eval_env.env.options
        def action_mask(state):
            state = state.cpu()
            device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
            mask = torch.from_numpy(np.array([o.initiation(state[0].numpy()) for o in options]).T).float().unsqueeze(0).to(device)
            return mask


    def random_selection(obs):
        _mask = action_mask(obs) + 1e-12
        _mask = _mask / _mask.sum(-1, keepdim=True)
        n_actions = _mask.shape[-1]
        # printarr(_mask, n_actions)
        selection =  np.random.choice(n_actions, p=_mask.squeeze().cpu().numpy())
        # printarr(selection)
        return torch.tensor(selection)

    if args.noisy_net_sigma is not None:
        pnn.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        # Turn off explorer
        explorer = explorers.Greedy()
    else:

        explorer = AbstractLinearDecayEpsilonGreedy(
            1.0,
            args.final_epsilon,
            args.final_exploration_frames,
            lambda obs: random_selection(obs),
        )

    opt = torch.optim.Adam(q_func.parameters(), lr=args.lr, eps=1.5e-4)
    # Select a replay buffer to use
    if args.prioritized:
        # Anneal beta from beta0 to 1 throughout training
        betasteps = args.steps / args.update_interval
        rbuf = replay_buffers.PrioritizedReplayBuffer(
            5*10**5,
            alpha=0.6,
            beta0=0.4,
            betasteps=betasteps,
            num_steps=args.num_step_return,
        )
    else:
        rbuf = replay_buffers.ReplayBuffer(10**6, args.num_step_return)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    Agent = AbstractDDQN #parse_agent(args.agent)
    agent = Agent(
        action_mask,
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        clip_delta=args.clip_delta,
        update_interval=args.update_interval,
        batch_accumulator="sum",
        phi=lambda x: x,
        minibatch_size=32
    )
    

    if args.load:
        agent.load(args.load)
    
    if args.demo:
        
        if args.eval_random_agent:
            agent = RandomAgent(random_selection, encoder)
        else:
            agent = AbstractDDQNGrounded(encoder, agent, action_mask)
        eval_stats = eval_performance(
            env=eval_env, 
            agent=agent, 
            n_steps=None, 
            n_episodes=args.eval_n_runs, 
            max_episode_len=50, 
            discounted=True
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        if args.finetune:   
            agent = AbstractDDQNGrounded(encoder, agent)
        train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=None,
            checkpoint_freq=args.checkpoint_frequency,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=False,
            eval_env=eval_env,
            use_tensorboard=True,
            train_max_episode_len=args.max_steps,
            eval_max_episode_len=50
        )


if __name__ == "__main__":
    main()