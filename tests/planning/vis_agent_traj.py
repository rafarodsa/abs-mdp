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
from src.options.policies import SoftmaxCategoricalHeadInitiation, OptionPolicyInit
from src.absmdp.policies import AbstractPolicyWrapper
from src.agents.abstract_ddqn import AbstractDoubleDQN as AbstractDDQN, AbstractLinearDecayEpsilonGreedy

from src.agents.train_agent import train_agent_with_evaluation
from src.agents.evaluator import eval_performance
from src.utils.printarr import printarr

import matplotlib.pyplot as plt


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


def parse_arch(arch, n_actions):
    if arch == "nature":
        return nn.Sequential(
            pnn.LargeAtariCNN(),
            init_chainer_default(nn.Linear(512, n_actions)),
            DiscreteActionValueHead(),
        )
    elif arch == "doubledqn":
        return nn.Sequential(
            pnn.LargeAtariCNN(),
            init_chainer_default(nn.Linear(512, n_actions, bias=False)),
            SingleSharedBias(),
            DiscreteActionValueHead(),
        )
    elif arch == "nips":
        return nn.Sequential(
            pnn.SmallAtariCNN(),
            init_chainer_default(nn.Linear(256, n_actions)),
            DiscreteActionValueHead(),
        )
    elif arch == 'mlp':
        return nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            DiscreteActionValueHead()
        )
    elif arch == "dueling":
        return DuelingDQN(n_actions)
    else:
        raise RuntimeError("Not supported architecture: {}".format(arch))


def parse_agent(agent):
    return {"DQN": agents.DQN, "DoubleDQN": agents.DoubleDQN, "PAL": agents.PAL}[agent]



# goal
def goal_fn(phi, goal=[0.55, 0.06, 0., 0.], goal_tol=0.01):
    goal = phi(torch.Tensor(goal)).numpy()
    def _goal(state):
        distance = np.linalg.norm(state[:2] - goal)
        # print(distance, goal_tol)
        # print(state, goal)
        return  distance <= goal_tol
    return _goal

def grounding_goal_fn(grounding, phi, goal=[0.55, 0.06, 0., 0.]):
    goal = torch.Tensor(goal).unsqueeze(0)
    def _goal(state):
        _goal = torch.Tensor(goal)
        # printarr(_goal, state)
        s = torch.from_numpy(state).unsqueeze(0)
        v = torch.tanh(grounding(_goal, s))
        return v > 0.

    return _goal

def make_env(test=False, test_seed=0, train_seed=1, args=None):
        goal_reward = 10000
        goal_tol = 3
        print('================ CREATE ENVIRONMENT ================', goal_reward)
        if not test:
            env = torch.load(args.absmdp)
            env = EnvGoalWrapper(env, goal_fn=goal_fn(env.encoder, goal_tol=goal_tol), goal_reward=goal_reward)
            # env = EnvGoalWrapper(env, goal_fn=grounding_goal_fn(env.grounding, env.encoder), goal_reward=goal_reward)
            env.seed(train_seed)
        else:
            # # evaluation in real env
            # env = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg')
            # if args.render:
            #     env.render_mode = 'human'
            # options = create_position_options(env)
            # env = EnvOptionWrapper(options, env)
            # def _goal_fn(state, goal=[0.55, 0.06], goal_tol=1/18):
            #     return np.sqrt(np.linalg.norm(state[:2] - goal)) <= goal_tol
            # env = EnvGoalWrapper(env, goal_fn=_goal_fn, goal_reward=goal_reward)
            env = torch.load(args.absmdp)
            env = EnvGoalWrapper(env, goal_fn=goal_fn(env.encoder, goal_tol=goal_tol), goal_reward=goal_reward)
            # env = EnvGoalWrapper(env, goal_fn=grounding_goal_fn(env.grounding, env.encoder), goal_reward=goal_reward)
            # env.seed(test_seed)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--absmdp',
        type=str,
        default='./mdp.pt'
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument(
        "--final-exploration-frames",
        type=int,
        default=10**5,
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
        default=10**5,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
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
        default=500,
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
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate.")
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

    
    
    env = make_env(test=False, train_seed=train_seed, test_seed=test_seed, args=args)
    eval_env = make_env(test=True, train_seed=train_seed, test_seed=test_seed, args=args)

    n_actions = env.action_space.n
    q_func = parse_arch(args.arch, n_actions)

    def action_mask(s):
        return (torch.sigmoid(env.env.initiation_set(s.float())) > 0.8).float()

    if args.noisy_net_sigma is not None:
        pnn.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        # Turn off explorer
        explorer = explorers.Greedy()
    else:
        explorer = AbstractLinearDecayEpsilonGreedy(
            1.0,
            args.final_epsilon,
            args.final_exploration_frames,
            lambda obs: np.random.choice(action_mask(obs).numpy().argmax(axis=-1)),
        )

    # Use the Nature paper's hyperparameters
    # opt = pfrl.optimizers.RMSpropEpsInsideSqrt(
    #     q_func.parameters(),
    #     lr=args.lr,
    #     alpha=0.95,
    #     momentum=0.0,
    #     eps=1e-2,
    #     centered=True,
    # )

    # opt = torch.optim.RMSprop(q_func.parameters(), lr=args.lr)
    opt = torch.optim.Adam(q_func.parameters(), lr=args.lr, eps=1.5e-4)
    # Select a replay buffer to use
    if args.prioritized:
        # Anneal beta from beta0 to 1 throughout training
        betasteps = args.steps / args.update_interval
        rbuf = replay_buffers.PrioritizedReplayBuffer(
            10**6,
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
        minibatch_size=256
    )

    if args.load:
        agent.load(args.load)

        eval_stats = eval_performance(
            env=eval_env, agent=agent, n_steps=None, n_episodes=args.eval_n_runs, max_episode_len=100
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )

        trajs = []

        goal_tol = 3
        goal = eval_env.env.encoder(torch.Tensor([0.55, 0.06, 0., 0.]))

        with agent.eval_mode():
            for i in range(args.eval_n_runs):
                terminate = False
                obs = eval_env.reset()
                t = []
                timestep = 0
                while not terminate:
                    action = agent.act(obs)
                    next_obs, reward, done, info = eval_env.step(action)
                    t.append((obs, action, reward, next_obs, done, info))
                    obs = next_obs
                    eval_env.render()
                    timestep += 1
                    if done or timestep > 100:
                        terminate = True
                        break
                if done:
                    trajs.append(t)

        fig, ax = plt.subplots()
        for t in trajs:
            s = np.array([_t[0] for _t in t] + [t[-1][3]])
            printarr(s)
            plt.plot(s[:, 0], s[:, 1], c='k')
            plt.scatter(s[:, 0], s[:, 1], s=10, c='b')
            plt.scatter(s[-1,0], s[-1, 1], s=10, c='r')
        plt.scatter(goal[0], goal[1], s=20, c='g')
        c = plt.Circle((goal[0], goal[1]), goal_tol, color='g', fill=False)
        ax.add_patch(c)
            
        plt.grid()
        plt.show()
        
    else:
        ValueError('No agent to load')

if __name__ == "__main__":
    main()