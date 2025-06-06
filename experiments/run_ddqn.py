import argparse

import numpy as np
import torch
import torch.nn as nn

import logging


from src.agents.abstract_ddqn import AbstractDoubleDQN as AbstractDDQN, AbstractLinearDecayEpsilonGreedy
from src.agents.abstract_ddqn import AbstractDDQNGrounded

from src.agents.train_agent import train_agent_with_evaluation
from src.agents.train_agent_batch import train_agent_batch_with_evaluation
from src.agents.evaluator import eval_performance
from src.utils.printarr import printarr

import pfrl
from pfrl import replay_buffers, utils
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.experiments.evaluation_hooks import EvaluationHook

from collections import defaultdict


class RandomAgent(pfrl.agent.Agent):
    def __init__(self, action_selection, encoder):
        self.action_selection = action_selection
        self.encoder = encoder
        self.gamma = 0.99
    def act(self, obs):
        z = self.encoder(torch.from_numpy(obs))
        return self.action_selection(z)
    def load(self, dirname):
        pass
    def get_statistics(self):
        pass
    def observe(self, *args):
        pass
    def save(self, dirname):
        pass


class LogDiscountedReturn(EvaluationHook):
    header = [
              'steps', 
              'discounted_mean', 
              'discounted_median', 
              'discounted_std', 
              'discounted_min', 
              'discounted_max', 
              'option_len_mean', 
              'option_len_median', 
              'option_len_std', 
              'option_len_min', 
              'option_len_max'
            ]

    def __init__(self):
        self.support_train_agent = True
        self.header_created = defaultdict(lambda: False)
        

    def __call__(self, env, agent, evaluator, step, eval_stats, agent_stats, env_stats):
        if not self.header_created[evaluator.outdir]:
            with open(f'{evaluator.outdir}/discounted_return.txt', 'w') as f:
                f.write('\t'.join(self.header) + '\n')
            self.header_created[evaluator.outdir] = True
        if eval_stats is not None:
            with open(f'{evaluator.outdir}/discounted_return.txt', 'a+') as f:
                f.write('\t'.join([str(step)] + [str(eval_stats[k]) for k in self.header[1:]]) + '\n')
    

def random_selection_initset(initset_s):
    # TODO this works only with single samples.
    avail_action = torch.nonzero(initset_s.squeeze())
    selection = torch.randint(0, avail_action.shape[0], (1,))
    a =  avail_action[selection].squeeze()
    return a

def random_selection_from_obs(obs, iniset_fn):
    return random_selection_initset(iniset_fn(obs))

def run_abstract_ddqn(envs, q_func, encoder, agent_args, experiment_args, finetuning_args=None, normalizer=lambda x: x, device='cpu', tune=False, trial=None, batched=False, example_env=None):

    '''
        envs: Dict of envs (e.g. {'train': train_env, 'eval': {'eval_env1', 'eval_env2', ...}})
        args: Namespace with DDQN config params
    '''

    if experiment_args.finetune:
        assert finetuning_args is not None

    logging.basicConfig(level=experiment_args.log_level)
    utils.set_random_seed(experiment_args.seed) # set random seed

    env = envs['train']
    eval_envs = envs['eval']


    # Exploration
    if not experiment_args.finetune:
        explorer = AbstractLinearDecayEpsilonGreedy(
            1.0,
            agent_args.final_epsilon,
            agent_args.final_exploration_steps,
            random_selection_initset,
        )
    else:
        explorer = AbstractLinearDecayEpsilonGreedy(
            1.0,
            finetuning_args.final_epsilon,
            finetuning_args.final_exploration_steps,
            random_selection_initset,
        )


    opt = torch.optim.Adam(q_func.parameters(), lr=agent_args.lr, eps=1.5e-4)


    ### REPLAY BUFFER
    if experiment_args.finetune:
        betasteps = finetuning_args.steps / agent_args.update_interval  
    else:
        betasteps = agent_args.steps / agent_args.update_interval


    rbuf = replay_buffers.PrioritizedReplayBuffer(
        agent_args.replay_buffer_size,
        alpha=0.6,
        beta0=0.4,
        betasteps=betasteps,
        num_steps=agent_args.num_step_return,
    )

    ## MAKE AGENT

    _initset_fn = env.initset if not batched else example_env.initset

    agent = AbstractDDQN(
        _initset_fn,
        q_func,
        opt,
        rbuf,
        gpu=experiment_args.gpu,
        gamma=agent_args.gamma,
        explorer=explorer,
        replay_start_size=agent_args.replay_start_size,
        target_update_interval=agent_args.target_update_interval,
        clip_delta=True,
        update_interval=agent_args.update_interval,
        batch_accumulator="sum",
        phi=normalizer,
        minibatch_size=32
    )

    ### TRAINING
    if agent_args.load:
        agent.load(agent_args.load)
        if experiment_args.finetune:
            agent.optimizer = torch.optim.Adam(q_func.parameters(), lr=finetuning_args.lr, eps=1.5e-4)
     
    if experiment_args.demo:
        if experiment_args.eval_random_agent:
            agent = RandomAgent(random_selection_from_obs, encoder)
        # elif not args.absgroundmdp:
        #     agent = AbstractDDQNGrounded(encoder, agent, env.initset, device)
        eval_stats = eval_performance(
            env=eval_envs, 
            agent=agent, 
            n_steps=None, 
            n_episodes=experiment_args.eval_n_runs, 
            max_episode_len=agent_args.max_episode_len, 
            discounted=False
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                experiment_args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        if experiment_args.finetune:   
            agent = AbstractDDQNGrounded(encoder, agent, action_mask=env.initset, device=device)
        steps = agent_args.steps if not experiment_args.finetune else finetuning_args.steps
        eval_hooks = [LogDiscountedReturn()] if not batched else []
        if tune:
            eval_hooks.append(pfrl.experiments.evaluation_hooks.OptunaPrunerHook(trial=trial))
        trainer_fn = train_agent_with_evaluation if not batched else train_agent_batch_with_evaluation
        _, eval_stats = trainer_fn(
                                    agent=agent,
                                    env=env,
                                    steps=steps,
                                    eval_n_steps=None,
                                    checkpoint_freq=experiment_args.checkpoint_frequency,
                                    eval_n_episodes=experiment_args.eval_n_runs,
                                    eval_interval=experiment_args.eval_interval,
                                    outdir=experiment_args.outdir,
                                    save_best_so_far_agent=True,
                                    eval_envs=eval_envs,
                                    use_tensorboard=experiment_args.log_tensorboard,
                                    train_max_episode_len=experiment_args.max_episode_len,
                                    eval_max_episode_len=experiment_args.max_episode_len,
                                    discounted=False,
                                    evaluation_hooks=eval_hooks,
                                    log_interval=1000
                                )

    return eval_stats