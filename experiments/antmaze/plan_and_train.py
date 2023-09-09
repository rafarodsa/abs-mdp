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


from omegaconf import OmegaConf as oc
from experiments.antmaze.plan import make_ground_env, parse_oc_args, gaussian_ball_goal_fn, GOALS, GOAL_TOL
from src.absmdp.absmdp import AbstractMDPGoal, AbstractMDP
from src.models import ModuleFactory
from src.agents.abstract_ddqn import AbstractDDQNGrounded, AbstractDoubleDQN, AbstractLinearDecayEpsilonGreedy

import pfrl
from pfrl.q_functions import DiscreteActionValueHead
from pfrl import replay_buffers, utils

import lightning as L

from src.agents.train_agent_online import train_agent_with_evaluation

from envs.pinball.pinball_gym import PinballPixelWrapper
from envs.pinball import PinballEnvContinuous

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

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/antmaze/online_planner.yaml')
    parser.add_argument('--exp-id', type=str, default=None)
    parser.add_argument('--learn-task-reward', action='store_true')
    
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

    # make ground environment
    train_env = make_ground_env(
                                    test=False,
                                    args=agent_cfg.env,
                                    gamma=agent_cfg.env.gamma,
                                    train_seed=train_seed,
                                    device=device,
                                )

    test_env = make_ground_env(
                                    test=True,
                                    args=agent_cfg.env,
                                    gamma=agent_cfg.env.gamma,
                                    train_seed=test_seed,
                                    device=device,
                                )

    # initialize fabric for logging and checkpointing
   
    # load world model
    goal_cfg = world_model_cfg.model.goal_class

    mdp_constructor = AbstractMDPGoal if args.learn_task_reward else AbstractMDP

    if world_model_cfg.ckpt is not None:
        print(f'Loading world model from checkpoint at {world_model_cfg.ckpt}')
        world_model = mdp_constructor.load_from_old_checkpoint(world_model_cfg=world_model_cfg)
    else:
        world_model = mdp_constructor(world_model_cfg, goal_cfg=goal_cfg)
    
    # make task_reward_funcion
    goal = GOALS[agent_cfg.env.envname][agent_cfg.env.goal]

    def make_task_reward(envname):
        goal_fn = gaussian_ball_goal_fn(world_model.encoder, goal, goal_tol=GOAL_TOL, device=device, envname=agent_cfg.env.envname)
        def __r(s):
            r = goal_fn(s)
            return r, r > 0
        return __r

    task_reward = make_task_reward(agent_cfg.env.envname)
    # world_model.set_task_reward(task_reward)

    world_model.freeze(world_model_cfg.fixed_modules)

    # make grounded agent
    agent, grounded_agent = make_ddqn_agent(agent_cfg, experiment_cfg=cfg.experiment, world_model=world_model)

    # train agent
    train_agent_with_evaluation(
                                grounded_agent, 
                                train_env,
                                test_env, 
                                world_model, 
                                task_reward,
                                max_steps=cfg.experiment.steps,
                                config=cfg
                            )

if __name__ == '__main__':
    main()