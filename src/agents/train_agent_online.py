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
                    discounted=False
                 ):
        
        assert isinstance(agent, AbstractDDQNGrounded) # TODO create abstract class for these agents.
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
        self.eval_during_episode = eval_during_episode

        self.logger = logger or logging.getLogger(__name__)
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

        try:
            while t < steps_budget:

                # a_t
                initset_s = self.env.last_initset

                action = self.agent.act(obs, initset_s)

                # o_{t+1}, r_{t+1}
                obs, r, done, info = self.env.step(action.cpu())
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
                    print(f'agent training: episode {episode_idx} length {episode_len} return {episode_r.item()}, statistics: {self.agent.get_statistics()}')
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
                if self.checkpoint_freq and self.ground_step % self.checkpoint_freq == 0:
                    save_agent(self.agent, self.ground_step, self.outdir, self.logger, suffix="_checkpoint")

        except (Exception, KeyboardInterrupt):
            # Save the current model before being killed
            save_agent(self.agent, self.ground_step, self.outdir, self.logger, suffix="_except")
            raise

        # Save the final model
        # save_agent(self.agent, t, self.outdir, self.logger, suffix="_finish")

        self.current_step += t

        return eval_stats_history
    
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
    world_model_outdir = f'{basedir}/world_model/{config.exp_id}'
    os.makedirs(world_model_outdir, exist_ok=True)
    # make agent outdir
    agent_outdir = f'{basedir}/agent'
    os.makedirs(agent_outdir, exist_ok=True)
    agent_outdir = pfrl.experiments.prepare_output_dir(None, agent_outdir, exp_id=config.exp_id, make_backup=False)
    return basedir, world_model_outdir, agent_outdir

def train_agent_with_evaluation(
                                grounded_agent, 
                                ground_env,
                                test_env, 
                                world_model, 
                                task_reward,
                                max_steps,
                                config
                            ):
    
    ## Boilerplate: set up logging, evaluator, and checkpointing 

    # set up outdir here!
    basedir, world_model_outdir, agent_outdir = makeoutdirs(config)

    # set trainer in imagination
    # TODO add "real steps" to agent trainer to log evaluation correctly.
    # TODO trigger evaluation by hand or align with steps in real env.
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
                                        )
    world_model.set_outdir(world_model_outdir)
    world_model.setup_trainer(config)
    world_model.set_task_reward(task_reward)

    ## main training loop

    episode_len = 0
    episode_count = 0
    s = ground_env.reset()
    max_rollout_len = config.experiment.max_rollout_len
    train_every = config.experiment.train_every
    checkpoint_freq = config.experiment.checkpoint_frequency
    episode_return = 0

    times_per_loop = deque(maxlen=50)
    tic = time.perf_counter()
    init_time = tic
    for timestep in range(max_steps):

        ## rollout agent in ground environment
        with grounded_agent.eval_mode():
        # with torch.no_grad():
            a = grounded_agent.act(s)

        next_s, r, done, info = ground_env.step(a)
        episode_return += r

        world_model.observe(s, a, info['env_reward'], next_s, done, info['tau'], info['success'], info=info)
        s = next_s
        episode_len += 1
        if done or episode_len >= max_rollout_len:
            ground_log = {
                'ground_env/episode_return': episode_return,
                'ground_env/episode_length': episode_len,
                'ground_env/success': float(info['goal_reached'])
            }
            # log 
            print(f'[rollout] timestep {timestep} episode {episode_count}, length {episode_len}, return {episode_return}')
            s = ground_env.reset()
            episode_len = 0
            episode_return = 0
            episode_count += 1
            world_model.end_episode(ground_log, step=timestep)

        if timestep % train_every == 0:
            ## train world model for n gradient steps
            world_model.train_world_model(timestep=timestep)
            ## train agent
            agent_trainer.train(steps_budget=config.planner.agent.rollout_len)
            
            ### timing
            toc = time.perf_counter()
            times_per_loop.append(toc-tic)
            tic = toc
            avg_time_per_loop = sum(times_per_loop) / len(times_per_loop)
            estimate = avg_time_per_loop * (max_steps - timestep) / train_every
            print(f'\t\tAverage time per loop: {avg_time_per_loop} seconds. Estimated time to complete: {datetime.timedelta(seconds=estimate)}')
        
        if timestep % checkpoint_freq == 0:
            # save world model.
            world_model.save_checkpoint()
        
        agent_trainer.update_ground_step()

    # final steps

    agent_trainer.train(steps_budget=config.planner.agent.rollout_len) # train & evaluate finally
    agent_trainer.save(t=timestep)
    world_model.save_checkpoint()


    print(f'Finished training. Total time: {datetime.timedelta(seconds=time.perf_counter()-init_time)}')