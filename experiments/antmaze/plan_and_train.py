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
from src.absmdp.absmdp import AbstractMDPGoalWTermination as AbstractMDPGoal, AbstractMDP
from src.absmdp.absmdp2 import AMDP
from src.models import ModuleFactory
from src.agents.abstract_ddqn import AbstractDDQNGrounded, AbstractDoubleDQN, AbstractLinearDecayEpsilonGreedy
from src.agents.rainbow import Rainbow, AbstractRainbow
from src.absmdp.datasets_traj import PinballDatasetTrajectory_
import pfrl
from pfrl.q_functions import DiscreteActionValueHead
from pfrl import replay_buffers, utils


import lightning as L

from src.agents.train_agent_online import train_agent_with_evaluation
from src.agents.multiprocess_env import MultiprocessVectorEnv

from envs.pinball.pinball_gym import PinballPixelWrapper
from envs.pinball import PinballEnvContinuous

from functools import partial

import pprint

def random_selection_initset(initset_s):
    # TODO this works only with single samples.
    if initset_s.sum() == 0: # TODO this doesnt make sense. it should be a mistake to not finish the episode.
        return torch.randint(0, initset_s.shape[-1], (1,)).item()
    avail_action = torch.nonzero(initset_s.squeeze())
    selection = torch.randint(0, avail_action.shape[0], (1,))
    a =  avail_action[selection].squeeze()
    return a.item()

def make_ddqn_agent(agent_cfg, experiment_cfg, world_model):
    
    agent_cfg.q_func.input_dim = world_model.n_feats
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
    print('Building Rainbow agent')
    agent_cfg.q_func_rainbow.input_dim = world_model.n_feats
    q_func = ModuleFactory.build(agent_cfg.q_func_rainbow)

    training_steps = experiment_cfg.steps // experiment_cfg.train_every
    agent_total_steps = agent_cfg.agent.rollout_len * training_steps

    betasteps = agent_total_steps / agent_cfg.agent.update_interval


    # explorer = explorers.LinearDecayEpsilonGreedy(
    #         1.0,
    #         agent_cfg.agent.final_epsilon,
    #         agent_total_steps * agent_cfg.agent.final_exploration_steps,
    #         lambda: np.random.choice(agent_cfg.q_func.n_actions)
    # )

    agent = Rainbow(
        q_func, # TODO change config for this
        n_actions=agent_cfg.q_func.n_actions,
        betasteps=betasteps,
        lr=agent_cfg.agent.lr,
        n_steps=agent_cfg.agent.num_step_return,
        replay_start_size=agent_cfg.agent.replay_start_size,
        target_update_interval=agent_cfg.agent.target_update_interval,
        gamma=agent_cfg.env.gamma,
        gpu=experiment_cfg.gpu,
        update_interval=agent_cfg.agent.update_interval,
        explorer=None
    )
    device = f'cuda:{experiment_cfg.gpu}' if experiment_cfg.gpu >= 0 else 'cpu'

    grounded_agent = AbstractRainbow(agent=agent, encoder=world_model.encoder, action_mask=world_model.initset, device=device)

    return agent, grounded_agent

def make_ppo_agent(agent_cfg, experiment_cfg, world_model):
    from pfrl.policies import SoftmaxCategoricalHead
    from src.agents.ppo import AbstractPPO, PPO
    
    def ortho_init(layer, gain=1):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
        return layer
    agent_cfg.ppo_func.input_dim = world_model.n_feats
    model_feats = ModuleFactory.build(agent_cfg.ppo_func, ortho_init)


    model = torch.nn.Sequential(
        model_feats,
        nn.Tanh(),
        pfrl.nn.Branched(
            nn.Sequential(
                ortho_init(nn.Linear(agent_cfg.ppo_func.output_dim, agent_cfg.ppo_func.n_actions), gain=1e-2),
                SoftmaxCategoricalHead()
            ),
            ortho_init(nn.Linear(agent_cfg.ppo_func.output_dim, 1))
        )
    )
    

    opt = torch.optim.Adam(model.parameters(), lr=agent_cfg.agent.lr, eps=1e-5)

    agent = PPO(
        model,
        opt,
        gpu=experiment_cfg.gpu,
        phi=lambda x: x,
        update_interval=256,
        minibatch_size=128,
        epochs=10,
        clip_eps=0.1,
        clip_eps_vf=None,
        standardize_advantages=True,
        entropy_coef=1e-2,
        recurrent=False,
        max_grad_norm=0.5,
        gamma=agent_cfg.env.gamma
    )
    device = f'cuda:{experiment_cfg.gpu}' if experiment_cfg.gpu >= 0 else 'cpu'

    encoder = world_model.encode if not world_model.recurrent else (world_model.encode, world_model.transition)
    grounded_agent = AbstractPPO(agent=agent, encoder=encoder, action_mask=world_model.initset, device=device, recurrent=world_model.recurrent)

    return agent, grounded_agent

def make_mpc_agent(agent_cfg, experiment_cfg, world_model):
    from src.agents.mpc import AbstractMPC
    print('Building MPC agent')

    device = f'cuda:{experiment_cfg.gpu}' if experiment_cfg.gpu >= 0 else 'cpu'

    grounded_agent = AbstractMPC(world_model=world_model, action_mask=world_model.initset, device=device)

    return grounded_agent, grounded_agent


def get_goal_examples(goal, n_samples=10000, device='cpu', envname='antmaze-umaze-v2', abstract_tol=0.1):        
    goal = np.array(goal).astype(np.float32)

    
    # load sample datasets
    dataset = PinballDatasetTrajectory_(DATA_PATH[envname], length=64)

    states = [s[0] for i in range(len(dataset)) for s in dataset.trajectories[i]]
    states = np.array(states).astype(np.float32)

    distances = ((states[..., :2] - goal) ** 2).sum(-1) < GOAL_TOL ** 2

    #plot to test
    import matplotlib.pyplot as plt
    plt.scatter(states[:, 0], states[:, 1], c=distances)
    plt.savefig('samples_goal.png')
    return states[distances], states[~distances]

def make_batched_ground_env(make_env, num_envs, seeds):
    vec_env = MultiprocessVectorEnv(
        [
            partial(make_env, seed=seeds[i])
            for i in range(num_envs)
        ]
    )
    return vec_env

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/antmaze/online_planner.yaml')
    parser.add_argument('--exp-id', type=str, default=None)
    parser.add_argument('--learn-task-reward', action='store_true')
    parser.add_argument('--no-initset', action='store_true')
    parser.add_argument('--agent', type=str, choices=['rainbow', 'ddqn', 'ppo', 'mpc'])
    parser.add_argument('--mpc', action='store_true')
    
    args, unknown = parser.parse_known_args()
    cli_args = parse_oc_args(unknown)
    
    print(f'Loading experiment config from {args.config}')

    # load configs  
    cfg = oc.load(args.config)
    cfg = oc.merge(cfg, cli_args)

    pprint.pprint(oc.to_container(cfg))

    agent_cfg = cfg.planner
    world_model_cfg = cfg.world_model
    # make envs.
    train_seed = cfg.experiment.seed
    test_seed = 2**31 - 1 - cfg.experiment.seed
    process_seeds = np.arange(cfg.planner.env.n_envs).astype(np.int64) + cfg.experiment.seed * cfg.planner.env.n_envs
    print(f'Process seeds: {process_seeds}')
    assert process_seeds.max() < 2**32
    process_seeds = [int(s) for s in process_seeds]

    device = f'cuda:{cfg.experiment.gpu}' if cfg.experiment.gpu >= 0 else 'cpu'

    # make ground environment
    # train_env = make_ground_env(
    #                                 test=False,
    #                                 args=agent_cfg.env,
    #                                 gamma=agent_cfg.env.gamma,
    #                                 train_seed=train_seed,
    #                                 device=device,
    #                             )

    train_env = make_batched_ground_env(lambda *args, **kwargs: make_ground_env(args=agent_cfg.env, gamma=cfg.planner.env.gamma, test=True, device=device), seeds=process_seeds, num_envs=cfg.planner.env.n_envs)

    test_env = make_ground_env(
                                    test=True,
                                    args=agent_cfg.env,
                                    gamma=agent_cfg.env.gamma,
                                    seed=test_seed,
                                    device=device,
                                )

    # initialize fabric for logging and checkpointing
   
    # load world model
    goal_cfg = world_model_cfg.model.goal_class

    # mdp_constructor = AbstractMDPGoal if args.learn_task_reward else AbstractMDP
    mdp_constructor = AMDP


    # if world_model_cfg.ckpt is not None:
    #     print(f'Loading world model from checkpoint at {world_model_cfg.ckpt}')
    #     world_model = mdp_constructor.load_from_old_checkpoint(world_model_cfg=world_model_cfg)
    # else:
    #     world_model = mdp_constructor(world_model_cfg, goal_cfg=goal_cfg)
    
    
    world_model = mdp_constructor(world_model_cfg)

    # make task_reward_funcion
    goal = GOALS[agent_cfg.env.envname][agent_cfg.env.goal]
    # if args.learn_task_reward:
    #     pos_samples, neg_samples = get_goal_examples(goal, n_samples=5000)
    #     world_model.preload_task_reward_samples(pos_samples, neg_samples)


    def make_task_reward(envname, abstract_goal_tol):
        goal_fn = gaussian_ball_goal_fn(world_model.encoder, goal, goal_tol=GOAL_TOL, device=device, envname=agent_cfg.env.envname, abstract_tol=abstract_goal_tol)
        def __r(s):
            r = goal_fn(s)
            return r, r > 0
        return __r

    # task_reward = make_task_reward(agent_cfg.env.envname, agent_cfg.env.abstract_goal_tol)
    # world_model.set_task_reward(task_reward)

    # world_model.freeze(world_model_cfg.fixed_modules)

    # make grounded agent
    use_initset = True
    if args.agent == 'rainbow':
        agent, grounded_agent = make_rainbow_agent(agent_cfg, experiment_cfg=cfg.experiment, world_model=world_model)
        use_initset = False
        # world_model.set_no_initset()
    elif args.agent == 'ppo':
        agent, grounded_agent = make_ppo_agent(agent_cfg, experiment_cfg=cfg.experiment, world_model=world_model)
        use_initset = False
    elif args.agent == 'ddqn':
        use_initset = False
        agent, grounded_agent = make_ddqn_agent(agent_cfg, experiment_cfg=cfg.experiment, world_model=world_model)
    elif args.agent == 'mpc':
        _, grounded_agent = make_mpc_agent(agent_cfg, experiment_cfg=cfg.experiment, world_model=world_model)
    else:
        raise ValueError(f'Agent {args.agent} not implemented')
    
    if args.no_initset:
        use_initset = False
        world_model.set_no_initset()

    # train agent
    print(world_model)
    train_agent_with_evaluation(
                                grounded_agent, 
                                train_env,
                                test_env, 
                                world_model, 
                                True,
                                max_steps=cfg.experiment.steps,
                                config=cfg,
                                use_initset=use_initset,
                                learning_reward=args.learn_task_reward,
                                args=args, 
                                device=device
                            )

if __name__ == '__main__':
    main()