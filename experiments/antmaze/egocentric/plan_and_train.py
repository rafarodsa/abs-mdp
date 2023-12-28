'''
    Plan with DDQN and train world model
    author: Rafael Rodriguez-Sanchez
    date: 26 August 2023
'''
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pfrl import explorers

from omegaconf import OmegaConf as oc
from experiments.antmaze.plan import make_ground_env, parse_oc_args, gaussian_ball_goal_fn, GOALS, GOAL_TOL, DATA_PATH
from src.absmdp.absmdp import AbstractMDPGoal, AbstractMDP
from src.models import ModuleFactory
from src.agents.abstract_ddqn import AbstractDDQNGrounded, AbstractDoubleDQN, AbstractLinearDecayEpsilonGreedy
from src.agents.rainbow import Rainbow, AbstractRainbow
from src.absmdp.datasets_traj import PinballDatasetTrajectory_
import pfrl
from pfrl.q_functions import DiscreteActionValueHead
from pfrl import replay_buffers, utils

import lightning as L

from src.agents.train_agent_online import train_agent_with_evaluation

from envs.pinball.pinball_gym import PinballPixelWrapper
from envs.pinball import PinballEnvContinuous

from director.embodied.envs import loconav as nav
from experiments.antmaze.egocentric.env_adaptor import EmbodiedEnv
from envs.env_options import EnvOptionWrapper
from envs.env_goal import EnvGoalWrapper


def random_selection_initset(initset_s):
    # TODO this works only with single samples.
    if initset_s.sum() == 0: # TODO this doesnt make sense. it should be a mistake to not finish the episode.
        return torch.randint(0, initset_s.shape[-1], (1,)).item()
    avail_action = torch.nonzero(initset_s.squeeze())
    selection = torch.randint(0, avail_action.shape[0], (1,))
    a =  avail_action[selection].squeeze()
    return a

def make_ddqn_agent(agent_cfg, experiment_cfg, world_model):
    
    agent_cfg.q_func.input_dim = world_model.latent_dim
    q_func = ModuleFactory.build(agent_cfg.q_func)
    q_func = torch.nn.Sequential(q_func, DiscreteActionValueHead())

    training_steps = experiment_cfg.steps // experiment_cfg.train_every
    agent_total_steps = agent_cfg.agent.rollout_len * training_steps

    betasteps = agent_total_steps / agent_cfg.agent.update_interval
    rbuf = replay_buffers.PrioritizedReplayBuffer(
        agent_cfg.agent.replay_buffer_size,
        alpha=0.6,
        beta0=0.4,
        betasteps=betasteps,
        num_steps=agent_cfg.agent.num_step_return,
    )

    explorer = AbstractLinearDecayEpsilonGreedy(
            1.0,
            agent_cfg.agent.final_epsilon,
            agent_total_steps * agent_cfg.agent.final_exploration_steps,
            random_selection_initset,
        )

    opt = torch.optim.Adam(q_func.parameters(), lr=agent_cfg.agent.lr, eps=1.5e-4)
    agent = AbstractDoubleDQN(
        world_model.initset,
        q_func,
        opt,
        rbuf,
        gpu=experiment_cfg.gpu,
        gamma=agent_cfg.env.gamma,
        explorer=explorer,
        replay_start_size=agent_cfg.agent.replay_start_size,
        target_update_interval=agent_cfg.agent.target_update_interval,
        clip_delta=True,
        update_interval=agent_cfg.agent.update_interval,
        batch_accumulator="sum",
        phi=lambda x: x,
        minibatch_size=32
    )

    device = f'cuda:{experiment_cfg.gpu}' if experiment_cfg.gpu >= 0 else 'cpu'

    grounded_agent = AbstractDDQNGrounded(agent=agent, encoder=world_model.encoder, action_mask=world_model.initset, device=device)

    return agent, grounded_agent


def make_rainbow_agent(agent_cfg, experiment_cfg, world_model):
    agent_cfg.q_func_rainbow.input_dim = world_model.latent_dim
    q_func = ModuleFactory.build(agent_cfg.q_func_rainbow)

    training_steps = experiment_cfg.steps // experiment_cfg.train_every
    agent_total_steps = agent_cfg.agent.rollout_len * training_steps

    betasteps = agent_total_steps / agent_cfg.agent.update_interval


    explorer = explorers.LinearDecayEpsilonGreedy(
            1.0,
            agent_cfg.agent.final_epsilon,
            agent_total_steps * agent_cfg.agent.final_exploration_steps,
            lambda: np.random.choice(agent_cfg.q_func.n_actions)
    )

    agent = Rainbow(
        q_func, # TODO change config for this
        n_actions=agent_cfg.q_func.n_actions,
        betasteps=betasteps,
        lr=agent_cfg.agent.lr,
        n_steps=agent_cfg.agent.num_step_return,
        replay_start_size=agent_cfg.agent.replay_start_size,
        target_update_interval=agent_cfg.agent.update_interval,
        gamma=agent_cfg.env.gamma,
        gpu=experiment_cfg.gpu,
        update_interval=agent_cfg.agent.update_interval,
        explorer=explorer
    )
    device = f'cuda:{experiment_cfg.gpu}' if experiment_cfg.gpu >= 0 else 'cpu'

    grounded_agent = AbstractRainbow(agent=agent, encoder=world_model.encoder, action_mask=world_model.initset, device=device)

    return agent, grounded_agent

def make_dummy_options(act_space, n_options=4):
    from src.options.option import Option
    options = []
    dummy_init = lambda *args: 1.
    dummy_termination = lambda *args: lambda *args: 0.9
    for i in range(n_options):
        action = act_space.sample()
        dummy_policy = lambda obs: lambda *args: action
        opt = Option(dummy_init, dummy_policy, dummy_termination)
        options.append(opt)
    return options

def get_pose(env):
    physics = env._env._env._physics
    return list(map(np.array, env._walker.get_pose(physics)))
        
def make_reward_function(goal, env, tol=0.5):
    target_pos = env._arena.grid_to_world_positions([goal])[0][:2]
    def reward_fn(obs):
        pos = np.array(get_pose(env)[0])[:2]
        return ((pos - target_pos) ** 2).sum() < tol ** 2
    return reward_fn

GOALS = {
    'ant_maze_xl': [[14, 14], [8, 14], [2, 14], [8, 8], [2, 8]],
    'ant_empty': [[14, 14], [8, 14], [2, 14], [8, 8], [2, 8]]
}

def make_egocentric_maze(name, goal):
    
    base_env = nav.LocoNav(name) 
    env = EmbodiedEnv(base_env)
    # dummy options to test.

    options = make_dummy_options(env.action_space)

    # goal space.
    assert name in GOALS and len(GOALS[name]) > goal
    goal = GOALS[name][goal]
    task_reward = make_reward_function(goal, base_env)

    # i need to save global position to easily design reward function
    # array of goals for all mazes?
    # randomize initial position ??
    # make env
    env = EnvOptionWrapper(options, env)
    env = EnvGoalWrapper(env, task_reward)
    return env

    
def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/antmaze/online_planner.yaml')
    parser.add_argument('--exp-id', type=str, default=None)
    parser.add_argument('--learn-task-reward', action='store_true')
    parser.add_argument('--no-initset', action='store_true')
    parser.add_argument('--agent', type=str, default='ddqn', choices=['rainbow', 'ddqn'])
    
    args, unknown = parser.parse_known_args()
    cli_args = parse_oc_args(unknown)
    
    print(f'Loading experiment config from {args.config}')

    # load configs  
    cfg = oc.load(args.config)
    cfg = oc.merge(cfg, cli_args)
    agent_cfg = cfg.planner
    world_model_cfg = cfg.world_model
    # make envs.
    train_seed = cfg.experiment.seed
    test_seed = 2**31 - 1 - cfg.experiment.seed

    device = f'cuda:{cfg.experiment.gpu}' if cfg.experiment.gpu >= 0 else 'cpu'

    train_env = make_egocentric_maze('ant_maze_xl', goal=1)
    test_env = make_egocentric_maze('ant_maze_xl', goal=1)

    # initialize fabric for logging and checkpointing
   
    # load world model
    goal_cfg = world_model_cfg.model.goal_class

    mdp_constructor = AbstractMDPGoal if args.learn_task_reward else AbstractMDP


    if world_model_cfg.ckpt != 'none':
        print(f'Loading world model from checkpoint at {world_model_cfg.ckpt}')
        world_model = mdp_constructor.load_from_old_checkpoint(world_model_cfg=world_model_cfg)
    else:
        world_model = mdp_constructor(world_model_cfg, goal_cfg=goal_cfg)
    
    # # make task_reward_funcion
    # goal = GOALS[agent_cfg.env.envname][agent_cfg.env.goal]
    # if args.learn_task_reward:
    #     pos_samples, neg_samples = get_goal_examples(goal, n_samples=5000)
    #     world_model.preload_task_reward_samples(pos_samples, neg_samples)


    world_model.freeze(world_model_cfg.fixed_modules)

    # make grounded agent
    use_initset = True
    if args.agent == 'rainbow':
        agent, grounded_agent = make_rainbow_agent(agent_cfg, experiment_cfg=cfg.experiment, world_model=world_model)
        use_initset = False
        # world_model.set_no_initset()
    elif args.agent == 'ddqn':
        agent, grounded_agent = make_ddqn_agent(agent_cfg, experiment_cfg=cfg.experiment, world_model=world_model)
    else:
        raise ValueError(f'Agent {args.agent} not implemented')
    
    if args.no_initset:
        use_initset = False
        world_model.set_no_initset()

    # train agent
    train_agent_with_evaluation(
                                grounded_agent, 
                                train_env,
                                test_env, 
                                world_model, 
                                task_reward=None, # handcrafted reward
                                max_steps=cfg.experiment.steps,
                                config=cfg,
                                use_initset=use_initset,
                                learning_reward=args.learn_task_reward
                            )

if __name__ == '__main__':
    main()