'''
    Train abstract world model online
    author: Rafael Rodriguez-Sanchez
    date: 24 August 2023
'''

import logging
import os
import torch
import pfrl
from pfrl.experiments.evaluator import save_agent
from pfrl.utils.ask_yes_no import ask_yes_no

from src.agents.evaluator import Evaluator
from src.utils.printarr import printarr

from src.agents.abstract_ddqn import AbstractDDQNGrounded
from tqdm import tqdm
import time, datetime
from collections import deque
import numpy as np

import pathlib

from jax import tree_map

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


class GroundedPFRLTrainer:
    def __init__(
                    self, 
                    agent,
                    env,
                    steps,
                    eval_n_steps,
                    eval_n_episodes,
                    eval_interval,
                    outdir,
                    checkpoint_freq=None,
                    train_max_episode_len=None,
                    step_offset=0,
                    eval_max_episode_len=None,
                    eval_envs=None,
                    successful_score=None,
                    step_hooks=(),
                    evaluation_hooks=(),
                    save_best_so_far_agent=True,
                    use_tensorboard=False,
                    eval_during_episode=False,
                    logger=None,
                    discounted=False,
                    use_initset=True
                 ):
        
        # assert isinstance(agent, AbstractDDQNGrounded) # TODO create abstract class for these agents.
        self.grounded_agent = agent # grounded agent
        self.agent = agent.agent # abstract agent
        self.env = env
        self.steps = steps
        self.eval_n_steps = eval_n_steps
        self.eval_n_episodes = eval_n_episodes
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.checkpoint_freq = checkpoint_freq
        self.max_episode_len = train_max_episode_len
        self.step_offset = step_offset
        self.eval_max_episode_len = eval_max_episode_len
        self.eval_envs = eval_envs
        self.successful_score = successful_score
        self.step_hooks = step_hooks
        self.evaluation_hooks = evaluation_hooks
        self.save_best_so_far_agent = save_best_so_far_agent
        self.eval_during_episode = eval_during_episode,
        self.use_initset = use_initset

        self.logger = logger or logging.getLogger(__name__)
        # self.logger.setLevel('INFO')
        # make pfrl evaluators
        evaluators = []
        for name, eval_env in eval_envs.items():
            
            os.makedirs(f'{outdir}/{name}', exist_ok=True)
            if 'sim' in name:
                agent = self.agent
            elif 'ground' in name:
                agent = self.grounded_agent
            else:
                raise ValueError(f'Env name must indicate whether the env is sim or ground')

            evaluator = Evaluator(
                agent=agent,
                n_steps=eval_n_steps,
                n_episodes=eval_n_episodes,
                eval_interval=eval_interval,
                outdir=f'{outdir}/{name}',
                max_episode_len=eval_max_episode_len,
                env=eval_env,
                step_offset=step_offset,
                evaluation_hooks=evaluation_hooks,
                save_best_so_far_agent=save_best_so_far_agent,
                use_tensorboard=use_tensorboard,
                logger=logger,
                discounted=discounted
            )
            evaluators.append(evaluator)
        self.evaluators = evaluators
        self.ground_step = 0
        self.current_step = 0 # n of step in simulation

    def compute_metrics(self, ep):
        ep = np.array([o for o in ep])
        norm2 = (ep ** 2).sum(-1)
        residual = ((ep[1:]-ep[:-1])**2).sum(-1)

        logs = {
            'sim_rollout/norm2_mean': norm2.mean(),
            'sim_rollout/norm2_std': norm2.std(),
            'sim_rollout/norm2_max': norm2.max(),
            'sim_rollout/norm2_min': norm2.min(),
            'sim_rollout/residual_norm2_mean': residual.mean(),
            'sim_rollout/residual_norm2_std': residual.std(),
            'sim_rollout/residual_norm2_max': residual.max(),
            'sim_rollout/residual_norm2_min': residual.min(),
        }
        return logs

    def train(self, steps_budget):
        '''
            Run training for a number of steps.
            return: evaluation statistics if evaluation is done
        '''

        episode_r = 0
        episode_idx = 0

        # o_0, r_0

        obs = self.env.reset()

        t = 0

        eval_stats_history = []  # List of evaluation episode stats dict
        episode_len = 0
        
        ep = [obs]
        ep_logs = []
        try:
            while t < steps_budget:

                # a_t
                if self.use_initset:
                    initset_s = self.env.last_initset
                    action = self.agent.act(obs, initset_s)
                else:
                    action = self.agent.act(obs)

                # o_{t+1}, r_{t+1}
                last_obs = obs
                obs, r, done, info = self.env.step(action)
                
                # track
                ep.append(obs)

                t += 1
                episode_r += r
                episode_len += 1
                reset = episode_len == self.max_episode_len or info.get("needs_reset", False)
            
                if 'tau' not in info:
                    info['tau'] = 1
                self.agent.observe(obs, r, done, reset, info)

                for hook in self.step_hooks:
                    hook(self.env, self.agent, t)

                episode_end = done or reset or t == steps_budget

                if episode_end:
                    print(f'agent training: episode {episode_idx} length {episode_len} return {episode_r}, statistics: {self.agent.get_statistics()}')
                    self.logger.info(
                        "outdir:%s step:%s episode:%s R:%s",
                        self.outdir,
                        t,
                        episode_idx,
                        episode_r,
                    )
                    stats = self.agent.get_statistics()
                    self.logger.info("statistics:%s", stats)
                    episode_idx += 1

                    ep_logs.append(self.compute_metrics(ep))

                if self.evaluators is not None and (episode_end or self.eval_during_episode):
                    eval_results = [self.evaluate(evaluator, self.ground_step, episode_idx, self.agent, self.successful_score) for evaluator in self.evaluators]
                    eval_dicts, success = list(zip(*eval_results))
                    eval_stats_history.append(eval_dicts)
                    # if any in success is True
                    if any(success):
                        break

                if episode_end:
                    if t == steps_budget:
                        break
                    # Start a new episode
                    episode_r = 0
                    episode_len = 0
                    obs = self.env.reset()
                    ep = [obs]

                # if self.checkpoint_freq and self.ground_step % self.checkpoint_freq == 0:
                #     save_agent(self.agent, self.ground_step, self.outdir, self.logger, suffix="_checkpoint")

        except (Exception, KeyboardInterrupt):
            # Save the current model before being killed
            save_agent(self.agent, self.ground_step, self.outdir, self.logger, suffix="_except")
            raise

        # Save the final model
        # save_agent(self.agent, self.ground_step, self.outdir, self.logger, suffix="_finish")

        self.current_step += t

        return eval_stats_history, tree_map(lambda *ep_stats: np.median(np.array(ep_stats)), *ep_logs)
    
    def update_ground_step(self, t=1):
        self.ground_step += t

    def evaluate(self, evaluator, t, episode_idx, agent, successful_score=None):
        '''
            Evaluate the agent and return the evaluation score.
            return:
                eval_stats: dict of evaluation statistics
                success: bool indicating whether the agent has reached the successful score
        '''
        eval_stats = {}
        eval_score = evaluator.evaluate_if_necessary(t=t, episodes=episode_idx)
        if eval_score is not None:
            eval_stats = dict(agent.get_statistics())
            eval_stats["eval_score"] = eval_score
        if (
            successful_score is not None
            and evaluator.max_score >= successful_score
        ):
            return eval_stats, True
        return eval_stats, False


    def save_agent_replay_buffer(self, agent, t, outdir, suffix="", logger=None):
        logger = logger or logging.getLogger(__name__)
        filename = os.path.join(outdir, "{}{}.replay.pkl".format(t, suffix))
        agent.replay_buffer.save(filename)
        logger.info("Saved the current replay buffer to %s", filename)


    def ask_and_save_agent_replay_buffer(self, agent, t, outdir, suffix=""):
        if hasattr(agent, "replay_buffer") and ask_yes_no(
            "Replay buffer has {} transitions. Do you save them to a file?".format(
                len(agent.replay_buffer)
            )
        ):  # NOQA
            self.save_agent_replay_buffer(agent, t, outdir, suffix=suffix)

    def save(self, t):
        """Save the current model to a file."""
        save_agent(self.agent, t, self.outdir, self.logger, suffix="_finish")


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
                                args=None
                            ):
    
    ## Boilerplate: set up logging, evaluator, and checkpointing 

    # set up outdir here!
    basedir, world_model_outdir, agent_outdir = makeoutdirs(config)

    # set trainer in imagination

    
    warmup_steps = world_model.warmup_steps

    if (pathlib.Path(world_model_outdir) / 'checkpoints/world_model.ckpt').exists():
        world_model = world_model.load_checkpoint(pathlib.Path(world_model_outdir) / 'checkpoints/world_model.ckpt', goal_cfg=world_model.goal_cfg)
        grounded_agent.encoder = world_model.encoder
        grounded_agent.action_mask = world_model.initset
        if args.mpc:
            grounded_agent.world_model = world_model
        print(f"Loading checkpoint at {pathlib.Path(world_model_outdir) / 'checkpoints/world_model.ckpt'}")

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
    world_model.set_task_reward(task_reward)
    world_model.setup_replay(offline_data_path)
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


    while timestep < max_steps:
        grounded_agent.agent.batch_last_episode = None
        ## rollout agent in ground environment
        if config.experiment.explore_ground:
            with torch.no_grad():
                actions = grounded_agent.act(ss)
        else:  
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


        agent_trainer.update_ground_step(len(ss))
        
        
        if learning_reward: 
            train_agent_in_sim = timestep > warmup_steps
            

        if should_train_wm(timestep) and len(world_model.data) > prefill:
            ## train world model for n gradient steps
            world_model.train_world_model(timestep=timestep)
            ## train agent
            
            ### timing
            toc = time.perf_counter()
            times_per_loop.append(toc-tic)
            tic = toc
            avg_time_per_loop = sum(times_per_loop) / len(times_per_loop)
            estimate = avg_time_per_loop * (max_steps - timestep) / train_every
            print(f'\tAverage time per loop: {avg_time_per_loop} seconds. Estimated time to complete: {datetime.timedelta(seconds=estimate)}')

        if train_agent_in_sim and should_train_agent(timestep):
            grounded_agent.agent.batch_last_episode = None
            eval_history, ep_logs = agent_trainer.train(steps_budget=config.planner.agent.rollout_len)
            world_model.log(ep_logs, step=timestep)
            print(f'[simulation stats] {" | ".join([f"{k}: {v}" for k,v in ep_logs.items()])}')


        if should_checkpoint(timestep):
            # save world model.
            world_model.save_checkpoint()
            print('Checkpointing world model...')
        
    # final steps

    agent_trainer.train(steps_budget=config.planner.agent.rollout_len) # train & evaluate finally
    agent_trainer.save(t=timestep)
    world_model.save_checkpoint()


    print(f'Finished training. Total time: {datetime.timedelta(seconds=time.perf_counter()-init_time)}')