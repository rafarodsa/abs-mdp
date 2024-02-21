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

from experiments.antmaze.egocentric.env_config import make_egocentric_maze

from src.agents.train_agent_batch import train_agent_batch_with_evaluation

def make_rainbow_agent(agent_cfg, obs):
    q_func = ModuleFactory.build(agent_cfg.q_func)

    with torch.no_grad():
        q_func(obs) # dummy forward pass

    training_steps = agent_cfg.experiment.steps
    agent_total_steps = training_steps

    betasteps = agent_total_steps / agent_cfg.experiment.update_interval

    # explorer = explorers.LinearDecayEpsilonGreedy(
    #         1.0,
    #         agent_cfg.experiment.final_epsilon,
    #         agent_total_steps * agent_cfg.experiment.final_exploration_steps,
    #         lambda: np.random.choice(agent_cfg.q_func.n_actions)
    # )

    agent = Rainbow(
        q_func, 
        n_actions=agent_cfg.q_func.n_actions,
        betasteps=betasteps,
        lr=agent_cfg.experiment.lr,
        n_steps=agent_cfg.experiment.num_step_return,
        replay_start_size=agent_cfg.experiment.replay_start_size,
        replay_buffer_size=agent_cfg.experiment.replay_buffer_size,
        target_update_interval=agent_cfg.experiment.target_update_interval,
        gamma=agent_cfg.experiment.gamma,
        gpu=agent_cfg.experiment.gpu,
        update_interval=agent_cfg.experiment.update_interval,
        v_max=1,
        v_min=-1,
        explorer=None, #explorer
    )
    device = f'cuda:{agent_cfg.experiment.gpu}' if agent_cfg.experiment.gpu >= 0 else 'cpu'

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
    args, cfg = parse_and_merge(parser)


    # make env

    process_seeds = np.arange(cfg.env.n_envs).astype(np.int64) + cfg.seed * cfg.env.n_envs
    print(f'Process seeds: {process_seeds}')
    assert process_seeds.max() < 2**32
    process_seeds = [int(s) for s in process_seeds]
    print(f'Building {cfg.env.n_envs} environments')
    train_env = make_batched_ground_env(lambda *args, **kwargs: make_egocentric_maze(cfg.env.envname, goal=cfg.env.goal, gamma=cfg.env.gamma, test=False, reward_scale=0.), seeds=process_seeds, num_envs=cfg.env.n_envs)
    # test_env = make_batched_ground_env(lambda *args, **kwargs: make_egocentric_maze(cfg.env.envname, goal=cfg.env.goal, gamma=cfg.env.gamma, test=True, reward_scale=0.), seeds=process_seeds, num_envs=cfg.env.n_envs)
    test_env = make_egocentric_maze(cfg.env.envname, goal=cfg.env.goal, gamma=cfg.env.gamma, test=True, reward_scale=0.)
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # run 
    obs = test_env.reset()
    obs = jax.tree_map(lambda s: torch.from_numpy(np.array(s)).float(), obs)
    agent, q_func, device = make_rainbow_agent(cfg.rainbow, obs)

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