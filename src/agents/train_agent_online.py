'''
    Train abstract world model online
    author: Rafael Rodriguez-Sanchez
    date: 24 August 2023
'''


import os
import torch
import pfrl

from src.agents.pfrl_trainer import GroundedPFRLTrainer

from src.agents.abstract_ddqn import AbstractDDQNGrounded
import time, datetime
from collections import deque
import numpy as np

import pathlib


class Every:

  def __init__(self, every, initial=True):
    self._every = every
    self._initial = initial
    self._prev = None

  def __call__(self, step):
    step = int(step)
    if self._every < 0:
      return True
    if self._every == 0:
      return False
    if self._prev is None:
      self._prev = (step // self._every) * self._every
      return self._initial
    if step >= self._prev + self._every:
      self._prev += self._every
      return True
    return False


''' 
    ground_env: ground environment is an environment with options
    world_model: abstract world model
    task_reward: reward function for the task
    config: configuration object
'''

def rollout(grounded_agent, ground_env, world_model, n_steps=1, max_rollout_len=64):
    """
        act in ground environment
        store transitions in world model
        
        args:
            grounded_agent: agent that acts in ground environment
            ground_env: ground environment (goal-based, task)
            world_model: abstract world model
            n_steps: number of steps to rollout

    """
    timestep = 0
    episode_len = 0
    s = ground_env.reset()
    while timestep < n_steps:
        a = grounded_agent.act(s)
        # TODO this is loading reward without the reward scale from the env.
        next_s, r, done, info = ground_env.step(a)
        # transform action one hot here? or change in the dataset loading?
        world_model.observe(s, a, r, next_s, done, info['tau'], info['success'], info=info)
        s = next_s
        timestep += 1
        episode_len += 1
        if done or episode_len >= max_rollout_len:
            s = ground_env.reset()
            episode_len = 0
            world_model.end_episode()
        # checkpointing and logging

    world_model.end_episode()
    return timestep


def makeoutdirs(config):
    basedir = f'{config.experiment_cwd}/{config.experiment_name}'
    os.makedirs(basedir, exist_ok=True)
    # make world model outdir
    world_model_outdir = f'{basedir}/{config.exp_id}/world_model/'
    os.makedirs(world_model_outdir, exist_ok=True)
    # make agent outdir
    agent_outdir = f'{basedir}/{config.exp_id}/agent'
    os.makedirs(agent_outdir, exist_ok=True)
    agent_outdir = pfrl.experiments.prepare_output_dir(None, agent_outdir, exp_id=config.planner.agent_id, make_backup=False)
    return basedir, world_model_outdir, agent_outdir

def train_agent_with_evaluation(
                                grounded_agent, 
                                ground_env,
                                test_env, 
                                world_model, 
                                task_reward,
                                max_steps,
                                config,
                                init_state_sampler=None,
                                use_initset=True, 
                                learning_reward=False,
                                offline_data_path=None,
                                args=None,
                                device='cpu'
                            ):
    
    ## Boilerplate: set up logging, evaluator, and checkpointing 

    # set up outdir here!
    basedir, world_model_outdir, agent_outdir = makeoutdirs(config)

    # set trainer in imagination

    warmup_steps = world_model.warmup_steps

    if (pathlib.Path(world_model_outdir) / 'checkpoints/world_model.ckpt').exists():
        world_model = world_model.load_checkpoint(pathlib.Path(world_model_outdir) / 'checkpoints/world_model.ckpt')
        grounded_agent.encoder = world_model.encode
        grounded_agent.action_mask = world_model.initset
        if args.mpc:
            grounded_agent.world_model = world_model
        print(f"Loading checkpoint at {pathlib.Path(world_model_outdir) / 'checkpoints/world_model.ckpt'}")

    if not args.mpc:
        agent_trainer = GroundedPFRLTrainer(
                                        agent=grounded_agent,
                                        env=world_model,
                                        steps=max_steps,
                                        eval_n_steps=None,
                                        eval_n_episodes=config.experiment.eval_n_runs,
                                        eval_interval=config.experiment.eval_interval,
                                        outdir=agent_outdir,
                                        checkpoint_freq=config.experiment.checkpoint_frequency,
                                        train_max_episode_len=config.experiment.max_episode_len,
                                        step_offset=0,
                                        eval_max_episode_len=config.experiment.max_episode_len,
                                        eval_envs={'ground': test_env},
                                        discounted=config.experiment.discounted,
                                        use_tensorboard=config.experiment.log_tensorboard,
                                        use_initset=use_initset
        )
    
    world_model.set_outdir(world_model_outdir)
    world_model.setup_trainer(config)
    # world_model.set_task_reward(task_reward)
    world_model.setup_replay(offline_data_path)
    world_model.sample_transition = config.world_model.sample_transition

    # world_model.set_init_state_sampler(test_env.init_state_sampler if init_state_sampler is None else init_state_sampler)

    ## main training loop

    episode_len = 0
    episode_count = 0
    ss = ground_env.reset()
    max_rollout_len = config.experiment.max_rollout_len
    train_every = config.experiment.train_every
    checkpoint_freq = config.experiment.checkpoint_frequency
    prefill = config.experiment.prefill
    episode_return = []

    times_per_loop = deque(maxlen=10)
    tic = time.perf_counter()
    init_time = tic
    train_agent_in_sim = not learning_reward
    timestep = world_model.timestep   
    # timestep=0

    should_train_wm = Every(config.world_model.train_every)
    should_train_agent = Every(config.planner.train_every)
    should_checkpoint = Every(checkpoint_freq)
    n_steps = config.experiment.gradient_steps if 'gradient_steps' in config.experiment else 1

    while timestep < max_steps:
        if not args.mpc:
            grounded_agent.agent.batch_last_episode = None
        ## rollout agent in ground environment
        # if config.experiment.explore_ground:
        #     with torch.no_grad():
        #         actions = grounded_agent.act(ss)
        # else:  
        with grounded_agent.eval_mode():
            actions = grounded_agent.act(ss)

        next_ss, rs, dones, infos = ground_env.step(actions) # this might return tuples/lists of steps per env if env is batched.
        # print(actions, rs, infos)

        if len(episode_return) == 0:
            episode_return = [0. for _ in range(len(rs))]
            episode_len = [0. for _ in range(len(rs))]
        episode_return = [rs + r for rs, r in zip(episode_return, rs)]
        episode_len = [ep_len + 1 for ep_len in episode_len]
        env_rewards = [info['env_reward'] for info in infos]
        taus = [info['tau'] for info in infos]
        successes = [info['success'] for info in infos]
        # print(dones, episode_len, max_rollout_len, episode_return)
        last = np.logical_or(dones, np.array(episode_len) >= max_rollout_len)
        world_model.observe(ss, actions, env_rewards, next_ss, dones, taus, successes, info=infos, last=last)

        ss = ground_env.reset(~last)
        if world_model.recurrent:
            grounded_agent.batch_reset(~last)
        # episode_count += last.sum()

        for i in range(len(episode_len)):
            timestep += 1
            if last[i]:
                episode_count += 1
                ground_log = {
                    'ground_env/episode_return': episode_return[i],
                    'ground_env/episode_length': episode_len[i],
                    'ground_env/success': float(infos[i]['goal_reached'])
                }
                # log 
                print(f'[rollout] timestep {timestep} episodes {len(world_model.data)}, length {episode_len[i]}, return {episode_return[i]}')
                world_model.log(ground_log, timestep)
                episode_len[i] = 0
                episode_return[i] = 0

        if not args.mpc:
            agent_trainer.update_ground_step(len(ss))
        

        if learning_reward: 
            train_agent_in_sim = world_model.n_gradient_steps > warmup_steps
            
        if should_train_wm(timestep) and len(world_model.data) > prefill:
            ## train world model for n gradient steps
            world_model.train_world_model(timestep=timestep, steps=n_steps)
            ## train agent
            
            ### timing
            toc = time.perf_counter()
            times_per_loop.append(toc-tic)
            tic = toc
            avg_time_per_loop = sum(times_per_loop) / len(times_per_loop)
            estimate = avg_time_per_loop * (max_steps - timestep) / train_every
            print(f'\tAverage time per loop: {avg_time_per_loop} seconds. Estimated time to complete: {datetime.timedelta(seconds=estimate)}')

        if train_agent_in_sim and should_train_agent(timestep) and not args.mpc:
            grounded_agent.agent.batch_last_episode = None
            eval_history, ep_logs = agent_trainer.train(steps_budget=config.planner.agent.rollout_len)
            world_model.log(ep_logs, step=timestep)
            print(f'[simulation stats] {" | ".join([f"{k}: {v}" for k,v in ep_logs.items()])}')


        if should_checkpoint(timestep):
            # save world model.
            world_model.save_checkpoint()
            print('Checkpointing world model...')
        
    # final steps
    if not args.mpc:
        agent_trainer.train(steps_budget=config.planner.agent.rollout_len) # train & evaluate finally
        agent_trainer.save(t=timestep)
    world_model.save_checkpoint()


    print(f'Finished training. Total time: {datetime.timedelta(seconds=time.perf_counter()-init_time)}')