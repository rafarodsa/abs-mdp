from envs.pinball.pinball_gym import PinballEnvContinuous, PinballEnv
from envs.pinball.controllers_pinball import PinballGridOptions
from envs.env_options import EnvOptionWrapper
from envs.env_goal import EnvGoalWrapper
from src.options.policies import SoftmaxCategoricalHeadOptions, OptionPolicy

import numpy as np

import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO


import argparse, functools
import torch
import torch.nn as nn



# goal
def goal_fn(state, goal=[0.5, 0.06], goal_tol=1/20):
    return np.linalg.norm(state[:2] - goal) <= goal_tol

def make_env(idx, test, process_seeds, args):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2**32 - 1 - process_seed if test else process_seed
        env = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg')
        if args.render:
            env.render_mode = 'human'
        options = PinballGridOptions(env)
        env = EnvOptionWrapper(options, env)
        env = EnvGoalWrapper(env, goal_fn=goal_fn)
        env.seed(env_seed)
        
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU device ID. Set to -1 to use CPUs only."
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of env instances run in parallel.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--steps", type=int, default=10**7, help="Total time steps for training."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=250,
        help="Interval (in timesteps) between evaluation phases.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=10,
        help="Number of episodes ran in an evaluation phase.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run demo episodes, not training.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help=(
            "Directory path to load a saved agent data from"
            " if it is a non-empty string."
        ),
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
    parser.add_argument(
        "--update-interval",
        type=int,
        default=100,
        help="Interval (in timesteps) between PPO iterations.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=256,
        help="Size of minibatch (in timesteps).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs used for each PPO iteration.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Interval (in timesteps) of printing logs.",
    )

    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=None,
        help="Frequency at which agents are stored.",
    )
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2**32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    

    def make_batch_env(test):
        vec_env = pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test, process_seeds, args)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )
        # if not args.no_frame_stack:
        #     vec_env = pfrl.wrappers.VectorFrameStack(vec_env, 4)
        return vec_env

    sample_env = make_batch_env(test=False)
    print("Observation space", sample_env.observation_space)
    print("Action space", sample_env.action_space)
    env = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg')
    options = PinballGridOptions(env)

    n_actions = sample_env.action_space.n
    obs_n_channels = sample_env.observation_space.low.shape[0]
    del sample_env, env
  

    model_features = nn.Sequential(
        lecun_init(nn.Linear(4, 128)),
        nn.ReLU(),
        lecun_init(nn.Linear(128, 128)),
        nn.ReLU()
    )

    model = OptionPolicy(
                        model_features,
                        lecun_init(nn.Linear(128, n_actions)),
                        lecun_init(nn.Linear(128, 1)),
                        options
                )
    

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)


    agent = PPO(
        model,
        opt,
        gpu=args.gpu,
        phi=lambda x: x.astype(np.float32),
        update_interval=args.update_interval,
        minibatch_size=args.batchsize,
        epochs=args.epochs,
        clip_eps=0.1,
        clip_eps_vf=None,
        standardize_advantages=True,
        entropy_coef=1e-1,
        recurrent=False,
        max_grad_norm=0.5,
        gamma=0.99,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=1000
        )
        print(
            "n_runs: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        step_hooks = []

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = value

        step_hooks.append(
            experiments.LinearInterpolationHook(args.steps, args.lr, 0, lr_setter)
        )
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            checkpoint_freq=args.checkpoint_frequency,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            save_best_so_far_agent=True,
            step_hooks=step_hooks,
            use_tensorboard=True,
            max_episode_len=100
        )




if __name__=='__main__':
    main()