import os
import pfrl
from pfrl import explorers
import numpy as np
import jax
import torch
import argparse

from src.models import ModuleFactory
from src.agents.rainbow import Rainbow
from experiments.antmaze.utils import parse_and_merge
from functools import partial
from experiments.antmaze.egocentric.env_config import make_egocentric_maze, make_egocentric_maze_ground_truth

from src.agents.train_agent_batch import train_agent_batch_with_evaluation

def make_rainbow_agent(agent_cfg, experiment_cfg, obs):
    print(agent_cfg)
    print(experiment_cfg)
    q_func = ModuleFactory.build(agent_cfg.q_func)

    with torch.no_grad():
        q_func(obs) # dummy forward pass

    training_steps = experiment_cfg.steps
    agent_total_steps = training_steps

    betasteps = agent_total_steps / experiment_cfg.update_interval

    # explorer = explorers.LinearDecayEpsilonGreedy(
    #         1.0,
    #         experiment_cfg.final_epsilon,
    #         agent_total_steps * experiment_cfg.final_exploration_steps,
    #         lambda: np.random.choice(agent_cfg.q_func.n_actions)
    # )

    agent = Rainbow(
        q_func, 
        n_actions=agent_cfg.q_func.n_actions,
        betasteps=betasteps,
        lr=experiment_cfg.lr,
        n_steps=experiment_cfg.num_step_return,
        replay_start_size=experiment_cfg.replay_start_size,
        replay_buffer_size=experiment_cfg.replay_buffer_size,
        target_update_interval=experiment_cfg.target_update_interval,
        gamma=experiment_cfg.gamma,
        gpu=experiment_cfg.gpu,
        update_interval=experiment_cfg.update_interval,
        v_max=1,
        v_min=-1,
        explorer=None, #explorer
    )
    device = f'cuda:{experiment_cfg.gpu}' if experiment_cfg.gpu >= 0 else 'cpu'

    return agent, q_func, device

def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise ValueError(f'x is {type(x)} not (torch.Tensor, np.ndarray)')
def encoder(state, device):
    keys = ['log_global_pos', 'log_global_orientation']
    state = torch.cat([to_tensor(state[k]) for k in keys], dim=-1)
    return state.float().to(device)


def make_rainbow_agent_gt(agent_cfg, experiment_cfg):
    print(agent_cfg)
    print(experiment_cfg)
    q_func = ModuleFactory.build(agent_cfg.q_func)

    # with torch.no_grad():
    #     q_func(obs) # dummy forward pass

    training_steps = experiment_cfg.steps
    agent_total_steps = training_steps

    betasteps = agent_total_steps / experiment_cfg.update_interval

    # explorer = explorers.LinearDecayEpsilonGreedy(
    #         1.0,
    #         agent_cfg.experiment.final_epsilon,
    #         agent_total_steps * agent_cfg.experiment.final_exploration_steps,
    #         lambda: np.random.choice(agent_cfg.q_func.n_actions)
    # )
    device = f'cuda:{experiment_cfg.gpu}' if experiment_cfg.gpu >= 0 else 'cpu'
    agent = Rainbow(
        q_func, 
        n_actions=agent_cfg.q_func.n_actions,
        betasteps=betasteps,
        lr=experiment_cfg.lr,
        n_steps=experiment_cfg.num_step_return,
        replay_start_size=experiment_cfg.replay_start_size,
        replay_buffer_size=experiment_cfg.replay_buffer_size,
        target_update_interval=experiment_cfg.target_update_interval,
        gamma=experiment_cfg.gamma,
        gpu=experiment_cfg.gpu,
        update_interval=experiment_cfg.update_interval,
        v_max=1,
        v_min=-1,
        explorer=None, #explorer
        phi=partial(encoder, device=device)
    )
    

    return agent, q_func, device

def make_batched_ground_env(make_env, num_envs, seeds):
    from src.agents.multiprocess_env import MultiprocessVectorEnv
    from functools import partial
    vec_env = MultiprocessVectorEnv(
        [
            partial(make_env, train_seed=seeds[i])
            for i in range(num_envs)
        ]
    )
    return vec_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/antmaze/egocentric/baseline_cfg.yaml')
    parser.add_argument('--agent', type=str, default='rainbow', choices=['rainbow', 'ddqn', 'ppo'])
    parser.add_argument('--exp_id', type=str, default='')
    parser.add_argument('--gt_encoder', action='store_true')
    parser.add_argument('--include_stop', action='store_true')
    args, cfg = parse_and_merge(parser)


    # make env

    process_seeds = np.arange(cfg.env.n_envs).astype(np.int64) + cfg.seed * cfg.env.n_envs
    print(f'Process seeds: {process_seeds}')
    assert process_seeds.max() < 2**32
    process_seeds = [int(s) for s in process_seeds]
    print(f'Building {cfg.env.n_envs} environments')
    
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # run 
    print(args)
    if args.gt_encoder:
       
        train_env = make_batched_ground_env(lambda *arg, **kwargs: make_egocentric_maze_ground_truth(name=cfg.env.envname, goal=cfg.env.goal, gamma=cfg.env.gamma, test=True, reward_scale=0., include_stop=args.include_stop), seeds=process_seeds, num_envs=cfg.env.n_envs)
        # test_env = make_batched_ground_env(lambda *args, **kwargs: make_egocentric_maze(cfg.env.envname, goal=cfg.env.goal, gamma=cfg.env.gamma, test=True, reward_scale=0.), seeds=process_seeds, num_envs=cfg.env.n_envs)
        test_env = make_egocentric_maze_ground_truth(cfg.env.envname, goal=cfg.env.goal, gamma=cfg.env.gamma, test=True, reward_scale=0., include_stop=args.include_stop)
        agent, q_func, device = make_rainbow_agent_gt(cfg.rainbow_gt, cfg.experiment)
    else:
        train_env = make_batched_ground_env(lambda *arg, **kwargs: make_egocentric_maze(cfg.env.envname, goal=cfg.env.goal, gamma=cfg.env.gamma, test=True, reward_scale=0., include_stop=args.include_stop), seeds=process_seeds, num_envs=cfg.env.n_envs)
        # test_env = make_batched_ground_env(lambda *args, **kwargs: make_egocentric_maze(cfg.env.envname, goal=cfg.env.goal, gamma=cfg.env.gamma, test=True, reward_scale=0.), seeds=process_seeds, num_envs=cfg.env.n_envs)
        test_env = make_egocentric_maze(cfg.env.envname, goal=cfg.env.goal, gamma=cfg.env.gamma, test=True, reward_scale=0., include_stop=args.include_stop)
        obs = test_env.reset()
        obs = jax.tree_map(lambda s: torch.from_numpy(np.array(s)).float(), obs)
        agent, q_func, device = make_rainbow_agent(cfg.rainbow, cfg.experiment, obs)

    # pfrl trainer

    train_agent_batch_with_evaluation(
        agent=agent,
        env=train_env,
        steps=cfg.trainer.steps,
        eval_n_steps=None,
        checkpoint_freq=cfg.trainer.checkpoint_frequency,
        eval_n_episodes=cfg.trainer.eval_n_runs,
        eval_interval=cfg.trainer.eval_interval,
        outdir=f'{cfg.trainer.outdir}/{args.exp_id}',
        save_best_so_far_agent=True,
        eval_envs={'ground': test_env},
        use_tensorboard=cfg.trainer.log_tensorboard,
        train_max_episode_len=cfg.trainer.max_episode_len,
        eval_max_episode_len=cfg.trainer.max_episode_len,
        discounted=False,
        log_interval=cfg.trainer.log_interval
    )

if __name__=='__main__':
    main()