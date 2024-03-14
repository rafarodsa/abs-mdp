
import os

import logging
import statistics
import time


from pfrl.experiments.evaluator import save_agent
from pfrl.utils.ask_yes_no import ask_yes_no
from src.agents.evaluator import PFRLEvaluator, Evaluator


from src.agents.evaluator import record_stats, record_tb_stats, create_tb_writer, write_header
from src.agents.agent import Agent

import numpy as np
from jax import tree_map

from contextlib import contextmanager


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
        ):
            self.save_agent_replay_buffer(agent, t, outdir, suffix=suffix)

    def save(self, t):
        """Save the current model to a file."""
        save_agent(self.agent, t, self.outdir, self.logger, suffix="_finish")

class PFRLAgent(Agent):

    def __init__(self, cfg, agent, step_hooks=(), evaluation_hooks=()):
        self.agent = agent # abstract agent
        self.cfg = cfg
        self.max_episode_len = self.cfg.experiment.max_episode_len
        self.eval_max_episode_len = self.cfg.experiment.max_episode_len
        self.step_hooks = step_hooks
        self.evaluation_hooks = evaluation_hooks
        self.outdir = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel('INFO')
        self.ground_step = 0
        self.imagination_steps = 0 
        self.n_episodes = 0
        self.start_time = time.time()

    def setup(self, outdir):
        self.outdir = outdir
        self.tb_writer = None
        if self.cfg.experiment.log_tensorboard:
            self.tb_writer = create_tb_writer(self.outdir)

        write_header(self.outdir, self.agent, None)

    def act(self, obs, encoder):
        with self.agent.eval_mode():
            if isinstance(obs, list):
                # acting in multivect env
                z = [encoder(s) for s in obs]
                action =  self.agent.batch_act(z)
            else:
                z = encoder(obs)
                action = self.agent.act(z)
            return action
        
    def act_in_imagination(self, z, initset=None):
        return self.agent.act(z)
    
    def update_ground_step(self, t=1):
        self.ground_step += t

    def compute_metrics(self, episodes):
        logs = []
        for ep in episodes:
            obss, actions, rewards, next_obss, dones, infos = zip(*ep)
            ep = np.array(obss + next_obss[-1:])
            norm2 = (ep ** 2).sum(-1, keepdims=True)
            residual = ((ep[1:]-ep[:-1])**2).sum(-1, keepdims=True)


            logs.append({
                'sim_rollout/norm2_mean': norm2.mean(),
                'sim_rollout/norm2_std': norm2.std(),
                'sim_rollout/norm2_max': norm2.max(),
                'sim_rollout/norm2_min': norm2.min(),
                'sim_rollout/residual_norm2_mean': residual.mean(),
                'sim_rollout/residual_norm2_std': residual.std(),
                'sim_rollout/residual_norm2_max': residual.max(),
                'sim_rollout/residual_norm2_min': residual.min(),
            })  
        stats = tree_map(lambda *ep_stats: np.median(np.array(ep_stats)), *logs)
        stats['n_episodes'] = len(episodes)
        return stats
    
    def compute_eval_stats(self, episodes):
        scores, lengths, exec_time, discounted_scores = [], [], [], []
        for ep in episodes:
            obss, actions, rewards, next_obss, dones, infos = zip(*ep)
            scores.append(sum(rewards))
            lengths.append(len(actions))
            taus = [info['tau'] for info in infos]
            discounts =  self.agent.gamma ** np.cumsum([0] + taus)
            discounted_rewards = np.array(rewards) * discounts[:-1]
            exec_time.append(sum(taus))
            discounted_scores.append(float(discounted_rewards.sum()))
        
        print(f'Evaluation: mean return {sum(scores) / len(scores)}, mean length {sum(lengths) / len(lengths)}')
        return scores, lengths, exec_time, discounted_scores

    def rollout(self, env, policy_fn, max_episode_len, n_episodes=None, n_steps=None, name='eval', stats_fn=None):
        logger = self.logger
        timestep = 0
        tau_total = 0
        reset = True
        assert (n_steps is None) != (n_episodes is None)
        episodes = []
        terminate = False
        n_steps = n_steps if n_steps else -1
        n_episodes = n_episodes if n_episodes else -1
        
        assert n_steps > 0 or n_episodes > 0

        while not terminate:
            if reset:
                obs = env.reset()
                done = False
                test_r = 0
                discounted_test_r = 0
                episode_len = 0
                info = {}
                tau_total = 0
                ep = []
            
            initset = getattr(env, 'last_initset', None)
            a = policy_fn(obs, initset)
            next_obs, r, done, info = env.step(a)
            
            ep.append([obs, a, r, next_obs, done, info])
            obs = next_obs
            
            episode_len += 1
            timestep += 1
            reset = done or episode_len == max_episode_len or info.get("needs_reset", False)
            if 'tau' not in info:
                info['tau'] = 1 

            self.agent.observe(obs, r, done, reset, info)
            test_r += r
            discounted_test_r += (self.agent.gamma ** tau_total) * r
            tau_total += info['tau']
            if reset:
                logger.info(
                    f"{name} episode {len(episodes)} length:{episode_len} R:{test_r}"
                )
                episodes.append(ep)
            terminate = len(episodes) == n_episodes or timestep == n_steps
                
        # If all steps were used for a single unfinished episode
        if len(episodes) == 0:
            episodes.append(ep) # non terminated episode
        return None if not stats_fn else stats_fn(episodes)

    def train(self, world_model, steps_budget):
        self.agent.batch_last_episode = None
        stats = self.rollout(
                            world_model, 
                            self.act_in_imagination, 
                            self.max_episode_len, 
                            n_steps=steps_budget, 
                            stats_fn=self.compute_metrics
                        )
        self.imagination_steps += steps_budget
        self.n_episodes += stats['n_episodes']
        del stats['n_episodes']
        return stats
    
    def evaluate(self, env, encoder, n_episodes):
        self.agent.batch_last_episode = None
        with self.agent.eval_mode():
            eval_stats = self.rollout(
                                        env,
                                        lambda obs, initset: self.act(obs, encoder),
                                        self.eval_max_episode_len,
                                        n_episodes=n_episodes,
                                        stats_fn=self.compute_eval_stats
                                    )
        scores, lengths, exec_times, discounted_scores = eval_stats
        option_duration_mean = np.array(exec_times)/np.array(lengths)

        stats = dict(
            episodes=len(scores),
            mean=statistics.mean(scores),
            median=statistics.median(scores),
            stdev=statistics.stdev(scores) if len(scores) >= 2 else 0.0,
            max=np.max(scores),
            min=np.min(scores),
            length_mean=statistics.mean(lengths),
            length_median=statistics.median(lengths),
            length_stdev=statistics.stdev(lengths) if len(lengths) >= 2 else 0,
            length_max=np.max(lengths),
            length_min=np.min(lengths),
            option_len_mean = statistics.mean(option_duration_mean),
            option_len_median = statistics.median(option_duration_mean),
            option_len_std = statistics.stdev(option_duration_mean) if len(option_duration_mean) >= 2 else 0.0,
            option_len_min = option_duration_mean.min(),
            option_len_max = option_duration_mean.max(),
            discounted_mean = statistics.mean(discounted_scores),
            discounted_median = statistics.median(discounted_scores),
            discounted_std = statistics.stdev(discounted_scores) if len(discounted_scores) >= 2 else 0.0,
            discounted_min = np.min(discounted_scores),
            discounted_max = np.max(discounted_scores),
        )

        agent_stats = self.agent.get_statistics()
        custom_values = tuple(tup[1] for tup in agent_stats)

        mean = stats["mean"]
        elapsed = time.time() - self.start_time
        values = (
            (
                self.ground_step,
                self.n_episodes,
                elapsed,
                mean,
                stats["median"],
                stats["stdev"],
                stats["max"],
                stats["min"],
            )
            + custom_values
        )

        record_stats(self.outdir, values)
        if self.tb_writer:
            record_tb_stats(self.tb_writer, agent_stats, stats, [], self.ground_step)

        return stats
    
    @contextmanager
    def eval_mode(self):
        orig_mode = self.agent.training
        try:
            self.agent.training = False
            yield
        finally:
            self.agent.training = orig_mode