'''
    Adaptation of PFRL's train_agent.py to work with abstract MDPs.
    author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
    date: 27 June 2023
'''

import logging
import os

from pfrl.experiments.evaluator import save_agent
from pfrl.utils.ask_yes_no import ask_yes_no

from src.agents.evaluator import Evaluator
from src.utils.printarr import printarr

import time, datetime
from collections import deque

import numpy as np

import time, datetime


def save_agent_replay_buffer(agent, t, outdir, suffix="", logger=None):
    logger = logger or logging.getLogger(__name__)
    filename = os.path.join(outdir, "{}{}.replay.pkl".format(t, suffix))
    agent.replay_buffer.save(filename)
    logger.info("Saved the current replay buffer to %s", filename)


def ask_and_save_agent_replay_buffer(agent, t, outdir, suffix=""):
    if hasattr(agent, "replay_buffer") and ask_yes_no(
        "Replay buffer has {} transitions. Do you save them to a file?".format(
            len(agent.replay_buffer)
        )
    ):  # NOQA
        save_agent_replay_buffer(agent, t, outdir, suffix=suffix)


def evaluate(evaluator, t, episode_idx, agent, successful_score=None):
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



def train_agent_batch(
    agent,
    env,
    steps,
    outdir,
    checkpoint_freq=None,
    log_interval=None,
    max_episode_len=None,
    step_offset=0,
    evaluator=None,
    successful_score=None,
    step_hooks=(),
    return_window_size=100,
    logger=None,
):
    """Train an agent in a batch environment.

    Args:
        agent: Agent to train.
        env: Environment to train the agent against.
        steps (int): Number of total time steps for training.
        outdir (str): Path to the directory to output things.
        checkpoint_freq (int): frequency at which agents are stored.
        log_interval (int): Interval of logging.
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        logger (logging.Logger): Logger used in this function.
    Returns:
        List of evaluation episode stats dict.
    """

    logger = logger or logging.getLogger(__name__)
    recent_returns = deque(maxlen=return_window_size)

    num_envs = env.num_envs
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_idx = np.zeros(num_envs, dtype="i")
    episode_len = np.zeros(num_envs, dtype="i")

    # o_0, r_0
    obss = env.reset()

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    eval_stats_history = []  # List of evaluation episode stats dict
    init_tic = time.time()
    time_per_step = deque(maxlen=25)
    try:
        tic = time.time()
        while True:
            
            # a_t
            initset_ss = env.last_initset
            actions = agent.batch_act(obss, initset_ss)
            # o_{t+1}, r_{t+1}
            obss, rs, dones, infos = env.step(actions) # TODO add .cpu() to actions
            episode_r += rs
            episode_len += 1

            # Compute mask for done and reset
            if max_episode_len is None:
                resets = np.zeros(num_envs, dtype=bool)
            else:
                resets = episode_len == max_episode_len
            resets = np.logical_or(
                resets, [info.get("needs_reset", False) for info in infos]
            )
            # Agent observes the consequences
            agent.batch_observe(obss, rs, dones, resets, infos)

            # Make mask. 0 if done/reset, 1 if pass
            end = np.logical_or(resets, dones)
            not_end = np.logical_not(end)

            # For episodes that ends, do the following:
            #   1. increment the episode count
            #   2. record the return
            #   3. clear the record of rewards
            #   4. clear the record of the number of steps
            #   5. reset the env to start a new episode
            # 3-5 are skipped when training is already finished.
            episode_idx += end
            recent_returns.extend(episode_r[end])

            for _ in range(num_envs):
                t += 1
                if checkpoint_freq and t % checkpoint_freq == 0:
                    save_agent(agent, t, outdir, logger, suffix="_checkpoint")

                for hook in step_hooks:
                    hook(env, agent, t)

            if (
                log_interval is not None
                and t >= log_interval
                and t % log_interval < num_envs
            ):
                logger.info(
                    "outdir:{} step:{} episode:{} last_R: {} average_R:{}".format(  # NOQA
                        outdir,
                        t,
                        np.sum(episode_idx),
                        recent_returns[-1] if recent_returns else np.nan,
                        np.mean(recent_returns) if recent_returns else np.nan,
                    )
                )
                logger.info("statistics: {}".format(agent.get_statistics()))

            if evaluator:
                eval_score = evaluator.evaluate_if_necessary(
                    t=t, episodes=np.sum(episode_idx)
                )
                if eval_score is not None:
                    eval_stats = dict(agent.get_statistics())
                    eval_stats["eval_score"] = eval_score
                    eval_stats_history.append(eval_stats)
                    if (
                        successful_score is not None
                        and evaluator.max_score >= successful_score
                    ):
                        break
            

            if (
                log_interval is not None
                and t >= log_interval
                and t % log_interval < num_envs
            ):
                toc = time.time()
                time_per_step.append((toc-tic)/log_interval)
                logger.info(f'Iteration time: {(toc-tic)/log_interval}s | Estimated time left: {datetime.timedelta(seconds=(steps-t)*sum(time_per_step)/len(time_per_step))}')
                tic = toc

            if t >= steps:
                break

            # Start new episodes if needed
            episode_r[end] = 0
            episode_len[end] = 0
            obss = env.reset(not_end)

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix="_except")
        env.close()
        if evaluator:
            evaluator.env.close()
        raise
    else:
        # Save the final model
        save_agent(agent, t, outdir, logger, suffix="_finish")
    print(f'Total time: {datetime.timedelta(seconds=time.perf_counter()-init_tic)}.')
    return eval_stats_history


def train_agent_batch_with_evaluation(
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
    return_window_size=100,
    eval_envs=None,
    log_interval=None,
    successful_score=None,
    step_hooks=(),
    evaluation_hooks=(),
    save_best_so_far_agent=True,
    use_tensorboard=False,
    logger=None,
    discounted=False,
):
    """Train an agent while regularly evaluating it.

    Args:
        agent: Agent to train.
        env: Environment train the againt against.
        steps (int): Number of total time steps for training.
        eval_n_steps (int): Number of timesteps at each evaluation phase.
        eval_n_runs (int): Number of runs for each time of evaluation.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output things.
        log_interval (int): Interval of logging.
        checkpoint_freq (int): frequency with which to store networks
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If set to None, max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
        logger (logging.Logger): Logger used in this function.
    Returns:
        agent: Trained agent.
        eval_stats_history: List of evaluation episode stats dict.
    """

    logger = logger or logging.getLogger(__name__)

    for hook in evaluation_hooks:
        if not hook.support_train_agent_batch:
            raise ValueError(
                "{} does not support train_agent_batch_with_evaluation().".format(hook)
            )

    os.makedirs(outdir, exist_ok=True)

    if eval_envs is None:
        eval_envs = {'train': env}

    if eval_max_episode_len is None:
        eval_max_episode_len = train_max_episode_len

    evaluators = []
    for name, eval_env in eval_envs.items():
        
        os.makedirs(f'{outdir}/{name}', exist_ok=True)

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

    eval_stats_history = train_agent_batch(
        agent,
        env,
        steps,
        outdir,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=train_max_episode_len,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        return_window_size=return_window_size,
        log_interval=log_interval,
        step_hooks=step_hooks,
        logger=logger,
    )

    return agent, eval_stats_history