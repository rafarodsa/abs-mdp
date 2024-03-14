'''
    Train Abstract World Model
    author: Rafael Rodriguez-Sanchez
'''

import argparse

import torch
import torch.nn as nn
import numpy as np

from omegaconf import OmegaConf as oc
from experiments.antmaze.plan import make_ground_env, parse_oc_args, GOAL_TOL, DATA_PATH
from src.absmdp.trainer import Trainer
from src.absmdp.absmdp2 import AMDP
from src.models import ModuleFactory
from src.agents.abstract_ddqn import AbstractDDQNGrounded, AbstractDoubleDQN, AbstractLinearDecayEpsilonGreedy
from src.agents.rainbow import Rainbow, AbstractRainbow
from src.absmdp.datasets_traj import PinballDatasetTrajectory_
import pfrl
from pfrl.q_functions import DiscreteActionValueHead
from pfrl import replay_buffers

from src.agents.multiprocess_env import MultiprocessVectorEnv
from src.agents.pfrl_trainer import PFRLAgent
from src.absmdp.trainer import Trainer

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

def make_ppo_agent(cfg, world_model):
    from pfrl.policies import SoftmaxCategoricalHead
    from src.agents.ppo import PPO
    
    def ortho_init(layer, gain=1):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
        return layer
    
    agent_cfg = cfg.planner
    agent_cfg.ppo_func.input_dim = world_model.n_feats


    policy = torch.nn.Sequential(
        ModuleFactory.build(agent_cfg.ppo_func, ortho_init),
        nn.Tanh(),
        ortho_init(nn.Linear(agent_cfg.ppo_func.output_dim, agent_cfg.ppo_func.n_actions), gain=1e-2),
        SoftmaxCategoricalHead()   
    )

    vf = torch.nn.Sequential(
        ModuleFactory.build(agent_cfg.ppo_func, ortho_init),
        nn.Tanh(),
        ortho_init(nn.Linear(agent_cfg.ppo_func.output_dim, 1), gain=0)
    )
    
    model = pfrl.nn.Branched(policy, vf)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.planner.agent.lr, eps=1e-5)

    agent = PPO(
        model,
        opt,
        gpu=cfg.experiment.gpu,
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
        gamma=cfg.planner.env.gamma
    )


    agent = PFRLAgent(cfg, agent)
    return agent

def make_dist_ac(cfg):
    from src.agents.ac import DistributionalActorCritic
    agent = DistributionalActorCritic(cfg)
    return agent

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
    parser.add_argument('--agent', type=str, choices=['rainbow', 'ddqn', 'ppo', 'dist-ac'])
    parser.add_argument('--data-path', type=str, default=None)
    
    args, unknown = parser.parse_known_args()
    cli_args = parse_oc_args(unknown)
    
    print(f'Loading experiment config from {args.config}')

    # load configs  
    oc.register_new_resolver("eval", eval, replace=True)
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

    train_env = make_batched_ground_env(lambda *args, **kwargs: make_ground_env(args=agent_cfg.env, gamma=cfg.planner.env.gamma, test=True, device=device), seeds=process_seeds, num_envs=cfg.planner.env.n_envs)
    test_env = make_ground_env(
                                    test=True,
                                    args=agent_cfg.env,
                                    gamma=agent_cfg.env.gamma,
                                    seed=test_seed,
                                    device=device,
                                )

   


    world_model = AMDP(world_model_cfg)

    # make grounded agent
    use_initset = True
    if args.agent == 'rainbow':
        agent, grounded_agent = make_rainbow_agent(agent_cfg, experiment_cfg=cfg.experiment, world_model=world_model)
        use_initset = False
        # world_model.set_no_initset()
    elif args.agent == 'ppo':
        agent = make_ppo_agent(cfg, world_model=world_model)
        use_initset = False
    elif args.agent == 'ddqn':
        use_initset = False
        agent, grounded_agent = make_ddqn_agent(agent_cfg, experiment_cfg=cfg.experiment, world_model=world_model)
    elif args.agent == 'dist-ac':
        agent = make_dist_ac(cfg)
    else:
        raise ValueError(f'Agent {args.agent} not implemented')
    
    
    trainer = Trainer(
                        cfg,
                        train_env=train_env, 
                        test_env=test_env, 
                        agent=agent, 
                        world_model=world_model,
                        offline_data=args.data_path
                    )
    
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()