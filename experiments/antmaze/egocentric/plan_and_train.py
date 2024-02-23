
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pfrl import explorers

from omegaconf import OmegaConf as oc
from experiments.antmaze.plan import make_ground_env, parse_oc_args, gaussian_ball_goal_fn, GOALS, GOAL_TOL, DATA_PATH
from src.absmdp.absmdp import AbstractMDPGoal, AbstractMDPGoalWTermination, TrueStateAbstractMDP, TrueStateAbstractMDPWTermination
from src.models import ModuleFactory
from src.agents.abstract_ddqn import AbstractDDQNGrounded, AbstractDoubleDQN, AbstractLinearDecayEpsilonGreedy
from src.agents.rainbow import Rainbow, AbstractRainbow
from src.absmdp.datasets_traj import PinballDatasetTrajectory_
import pfrl
from pfrl.q_functions import DiscreteActionValueHead
from pfrl import replay_buffers

from dm_control.utils.transformations import quat_rotate

from src.agents.train_agent_online import train_agent_with_evaluation

import pprint
from functools import partial
from experiments.antmaze.egocentric.env_config import make_egocentric_maze, make_egocentric_maze_ground_truth

def random_selection_initset(initset_s):
    # TODO this works only with single samples.
    if initset_s.sum() == 0: # TODO this doesnt make sense. it should be a mistake to not finish the episode.
        return torch.randint(0, initset_s.shape[-1], (1,)).item()
    avail_action = torch.nonzero(initset_s.squeeze())
    selection = torch.randint(0, avail_action.shape[0], (1,))
    a =  avail_action[selection].squeeze()
    return a.cpu()

def make_ddqn_agent(agent_cfg, experiment_cfg, world_model):
    
    agent_cfg.q_func.input_dim = world_model.latent_dim
    q_func = ModuleFactory.build(agent_cfg.q_func)
    q_func = torch.nn.Sequential(q_func, DiscreteActionValueHead())

    training_steps = experiment_cfg.steps // agent_cfg.train_every
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
            # agent_total_steps * agent_cfg.agent.final_exploration_steps
            agent_cfg.agent.final_exploration_steps,
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

    agent_cfg.q_func_rainbow.input_dim = world_model.n_feats
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
        q_func, 
        n_actions=agent_cfg.q_func.n_actions,
        betasteps=betasteps,
        lr=agent_cfg.agent.lr,
        n_steps=agent_cfg.agent.num_step_return,
        replay_start_size=agent_cfg.agent.replay_start_size,
        replay_buffer_size=agent_cfg.agent.replay_buffer_size,
        target_update_interval=agent_cfg.agent.target_update_interval,
        gamma=agent_cfg.env.gamma,
        gpu=experiment_cfg.gpu,
        update_interval=agent_cfg.agent.update_interval,
        explorer=None #explorer
    )
    device = f'cuda:{experiment_cfg.gpu}' if experiment_cfg.gpu >= 0 else 'cpu'

    encoder = world_model.encoder if not world_model.recurrent else (world_model.encoder, world_model.transition)
    grounded_agent = AbstractRainbow(agent=agent, encoder=encoder, action_mask=world_model.initset, device=device, recurrent=world_model.recurrent)

    return agent, grounded_agent

def make_ppo_agent(agent_cfg, experiment_cfg, world_model):
    from pfrl.policies import SoftmaxCategoricalHead
    from src.agents.ppo import AbstractPPO, PPO

    agent_cfg.q_func_rainbow.input_dim = world_model.n_feats
    model_feats = ModuleFactory.build(agent_cfg.q_func_rainbow)

    model = torch.nn.Sequential(
        model_feats,
        nn.Tanh(),
        pfrl.nn.Branched(
            nn.Sequential(
                nn.Linear(agent_cfg.q_func_rainbow.output_dim, agent_cfg.q_func_rainbow.n_actions),
                SoftmaxCategoricalHead()
            ),
            nn.Linear(agent_cfg.q_func_rainbow.output_dim, 1)
        )
    )
    

    opt = torch.optim.Adam(model.parameters(), lr=agent_cfg.agent.lr, eps=1e-5)

    agent = PPO(
        model,
        opt,
        gpu=experiment_cfg.gpu,
        phi=lambda x: x,
        update_interval=128,
        minibatch_size=8,
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

    encoder = world_model.encoder if not world_model.recurrent else (world_model.encoder, world_model.transition)
    grounded_agent = AbstractPPO(agent=agent, encoder=encoder, action_mask=world_model.initset, device=device, recurrent=world_model.recurrent)

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

def make_batched_ground_env(make_env, num_envs, seeds):
    from src.agents.multiprocess_env import MultiprocessVectorEnv
    vec_env = MultiprocessVectorEnv(
        [
            partial(make_env, train_seed=seeds[i])
            for i in range(num_envs)
        ]
    )
    return vec_env

def encoder(state, device):
    keys = ['log_global_pos', 'log_global_orientation']
    state = torch.cat([state[k] for k in keys], dim=-1)
    return state.float().to(device)


def compute_orientation(quat, device):
    shape = quat.shape
    if len(quat.shape) > 1:
        quat = quat.reshape(-1, shape[-1])
        bsize = quat.shape[0]
    else:
        bsize = 1
        quat = quat[None]
    x = np.stack([quat_rotate(quat[i], np.array([1., 0., 0.])) for i in range(bsize)])
    angle = np.arctan2(x[..., 1], x[..., 0])[..., None]
    z = torch.from_numpy(np.concatenate((np.cos(angle), np.sin(angle)), axis=-1)).to(device)
    return z.reshape(*shape[:-1], -1)

def simpler_encoder(state, device):
    quat = state['log_global_orientation']
    quat = quat.cpu().numpy() if isinstance(quat, torch.Tensor) else quat
    state = torch.cat([state['log_global_pos'][..., :2], compute_orientation(quat, device)], dim=-1)
    return state.float().to(device)
    
def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/antmaze/egocentric/online_planner.yaml')
    parser.add_argument('--exp-id', type=str, default=None)
    parser.add_argument('--no-initset', action='store_true')
    parser.add_argument('--ground_truth_encoder', action='store_true')
    parser.add_argument('--model_termination', action='store_true')
    parser.add_argument('--agent', type=str, default='rainbow', choices=['rainbow', 'ddqn', 'ppo'])
    parser.add_argument('--offline_model', action='store_true')
    parser.add_argument('--data_path', type=str, default=None)
    
    args, unknown = parser.parse_known_args()
    cli_args = parse_oc_args(unknown)
    
    print(f'Loading experiment config from {args.config}')
    torch.set_float32_matmul_precision('medium')

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
    reward_scale = cfg.planner.env.reward_scale


    # initialize fabric for logging and checkpointing
    
    if not args.ground_truth_encoder:
        train_env = make_batched_ground_env(lambda *args, **kwargs: make_egocentric_maze(cfg.planner.env.envname, goal=cfg.planner.env.goal, gamma=cfg.planner.env.gamma, test=False, reward_scale=reward_scale), seeds=process_seeds, num_envs=cfg.planner.env.n_envs)
        test_env = make_egocentric_maze(cfg.planner.env.envname, goal=cfg.planner.env.goal, gamma=cfg.planner.env.gamma, test=True, reward_scale=reward_scale)
        mdp_constructor = AbstractMDPGoal if not args.model_termination else AbstractMDPGoalWTermination
        goal_cfg = world_model_cfg.model.goal_class

        if world_model_cfg.ckpt != 'none':
            print(f'Loading world model from checkpoint at {world_model_cfg.ckpt}')
            world_model = mdp_constructor.load_from_old_checkpoint(world_model_cfg=world_model_cfg)
        else:
            world_model = mdp_constructor(world_model_cfg, goal_cfg=goal_cfg)
            # world_model = mdp_constructor(world_model_cfg)
    else:

        train_env = make_batched_ground_env(lambda *args, **kwargs: make_egocentric_maze_ground_truth(cfg.planner.env.envname, goal=cfg.planner.env.goal, gamma=cfg.planner.env.gamma, test=False, reward_scale=reward_scale), seeds=process_seeds, num_envs=cfg.planner.env.n_envs)
        test_env = make_egocentric_maze_ground_truth(cfg.planner.env.envname, goal=cfg.planner.env.goal, gamma=cfg.planner.env.gamma, test=True, reward_scale=reward_scale)
        mdp_constructor = TrueStateAbstractMDP if not args.model_termination else TrueStateAbstractMDPWTermination
        goal_cfg = world_model_cfg.model.goal_class
        world_model = mdp_constructor(world_model_cfg, goal_cfg=goal_cfg, encoder_fn=partial(simpler_encoder, device=device), ground_truth_state_dim=4)


    world_model.freeze(world_model_cfg.fixed_modules)
    world_model.to(device)
    # make grounded agent
    print(world_model)
    
    use_initset = True
    if args.agent == 'rainbow':
        agent, grounded_agent = make_rainbow_agent(agent_cfg, experiment_cfg=cfg.experiment, world_model=world_model)
        use_initset = False
        # world_model.set_no_initset()
    elif args.agent == 'ddqn':
        agent, grounded_agent = make_ddqn_agent(agent_cfg, experiment_cfg=cfg.experiment, world_model=world_model)
    elif args.agent == 'ppo':
        agent, grounded_agent = make_ppo_agent(agent_cfg, experiment_cfg=cfg.experiment, world_model=world_model)
        use_initset = False
    else:
        raise ValueError(f'Agent {args.agent} not implemented')
    
    if args.no_initset:
        use_initset = False
        world_model.set_no_initset()


    if args.offline_model:
        cfg.planner.train_every = int(1e10)
        cfg.world_model.train_every = 1
        cfg.experiment.checkpoint_frequency = 500
        assert os.path.exists(args.data_path)

    # train agent
    train_agent_with_evaluation(
                                grounded_agent, 
                                train_env,
                                test_env, 
                                world_model, 
                                task_reward=True, # handcrafted reward
                                max_steps=cfg.experiment.steps,
                                config=cfg,
                                use_initset=use_initset,
                                learning_reward=True,
                                offline_data_path=args.data_path
                            )

if __name__ == '__main__':
    main()