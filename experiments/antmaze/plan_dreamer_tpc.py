'''
    Learn with Dreamer
    author: Rafael Rodriguez-Sanchez
    date: 12 Nov 2023
'''
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pfrl import explorers

from omegaconf import OmegaConf as oc
from experiments.antmaze.plan import make_ground_env, parse_oc_args, gaussian_ball_goal_fn, GOALS, GOAL_TOL, DATA_PATH
from src.absmdp.absmdp import AbstractMDPGoal, AbstractMDP
from src.models import ModuleFactory
from src.agents.abstract_ddqn import AbstractDDQNGrounded, AbstractDoubleDQN, AbstractLinearDecayEpsilonGreedy
from src.agents.rainbow import Rainbow, AbstractRainbow
from src.absmdp.datasets_traj import PinballDatasetTrajectory_
import pfrl
from pfrl.q_functions import DiscreteActionValueHead
from pfrl import replay_buffers, utils

import lightning as L


# from dreamerv2.common.envs import GymWrapper

import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings

try:
    import rich.traceback
    rich.traceback.install()
except ImportError:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import src.agents.dreamer as agent

from src.agents.dreamer import GymWrapperOptions as GymWrapper
from dreamerv2 import common

def random_selection_initset(initset_s):
    # TODO this works only with single samples.
    if initset_s.sum() == 0: # TODO this doesnt make sense. it should be a mistake to not finish the episode.
        return torch.randint(0, initset_s.shape[-1], (1,)).item()
    avail_action = torch.nonzero(initset_s.squeeze())
    selection = torch.randint(0, avail_action.shape[0], (1,))
    a =  avail_action[selection].squeeze()
    return a

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


def train_dreamer(train_env, eval_env, config, pretrain_path=None, model_pretraining=False):
    # configs = yaml.safe_load((
    #     pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    # parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
    # config = common.Config(configs['defaults'])
    # for name in parsed.configs:
    #     config = config.update(configs[name])
    # config = common.Flags(config).parse(remaining)
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)

    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(not config.jit)
    message = 'No GPU found. To actually train on CPU remove this assert.'
    assert tf.config.experimental.list_physical_devices('GPU'), message
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        import tensorflow.keras.mixed_precision as prec
        prec.set_global_policy(prec.Policy('mixed_float16'))

    train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
    eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
        capacity=config.replay.capacity // 10,
        minlen=1,
        maxlen=config.dataset.length))
    step = common.Counter(train_replay.stats['total_steps'])
    outputs = [
        common.TerminalOutput(),
        common.JSONLOutput(logdir),
        common.TensorBoardOutput(logdir),
    ]
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video_train = common.Every(config.eval_every)
    should_video_eval = common.Every(config.eval_every)
    should_expl = common.Until(config.expl_until // config.action_repeat)

    def make_env(mode):
        env = train_env if mode == 'train' else eval_env
        env = GymWrapper(env, obs_key='joints')
        if hasattr(env.act_space['action'], 'n'):
            env = common.OneHotAction(env)
        else:
            env = common.NormalizeAction(env)
        env = common.TimeLimit(env, config.time_limit)
        return env

    def per_episode(ep, mode):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
        logger.scalar(f'{mode}_return', score)
        logger.scalar(f'{mode}_length', length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
            should = {'train': should_video_train, 'eval': should_video_eval}[mode]
            # if should(step):
            #     for key in config.log_keys_video:
            #         logger.video(f'{mode}_policy_{key}', ep[key])
            replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        logger.write()

    print('Create envs.')
    num_eval_envs = min(config.envs, config.eval_eps)
    if config.envs_parallel == 'none':
        train_envs = [make_env('train') for _ in range(config.envs)]
        eval_envs = [make_env('eval') for _ in range(num_eval_envs)]
    # else:
    #     make_async_env = lambda mode: common.Async(
    #         functools.partial(make_env, mode), config.envs_parallel)
    #     train_envs = [make_async_env('train') for _ in range(config.envs)]
    #     eval_envs = [make_async_env('eval') for _ in range(eval_envs)]
    act_space = train_envs[0].act_space
    obs_space = train_envs[0].obs_space
    train_driver = common.Driver(train_envs)
    train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_step(train_replay.add_step)
    train_driver.on_reset(train_replay.add_step)
    eval_driver = common.Driver(eval_envs)
    eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
    eval_driver.on_episode(eval_replay.add_episode)

    prefill = max(0, config.prefill - train_replay.stats['total_steps'])
    if prefill:
        print(f'Prefill dataset ({prefill} steps).')
        random_agent = common.RandomAgent(act_space)
        train_driver(random_agent, steps=prefill, episodes=1)
        eval_driver(random_agent, episodes=1)
        train_driver.reset()
        eval_driver.reset()

    print('Create agent.')
    train_dataset = iter(train_replay.dataset(**config.dataset))
    report_dataset = iter(train_replay.dataset(**config.dataset))
    eval_dataset = iter(eval_replay.dataset(**config.dataset))
    agnt = agent.Agent(config, obs_space, act_space, step)
    train_agent = common.CarryOverState(agnt.train)
    train_agent(next(train_dataset))
    if pretrain_path is not None:
        print(f'Pretrained agent loaded from {pretrain_path}')
        agnt.load(pretrain_path)
    elif (logdir / 'variables.pkl').exists():
        agnt.load(logdir / 'variables.pkl')
    else:
        print('Pretrain agent.')
        for _ in range(config.pretrain):
            train_agent(next(train_dataset))
        agnt.save(logdir / 'variables_pretrained.pkl')
        if model_pretraining:
            print('Done pretraining!')
            return
    
    train_policy = lambda *args: agnt.policy(
        *args, mode='explore' if should_expl(step) else 'train')
    eval_policy = lambda *args: agnt.policy(*args, mode='eval')

    def train_step(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                mets = train_agent(next(train_dataset))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(next(report_dataset)), prefix='train')
            logger.write(fps=True)
    train_driver.on_step(train_step)

    while step < config.steps:
        logger.write()
        print('Start evaluation.')
        logger.add(agnt.report(next(eval_dataset)), prefix='eval')
        eval_driver(eval_policy, episodes=config.eval_eps)
        print('Start training.')
        train_driver(train_policy, steps=config.eval_every)
        agnt.save(logdir / 'variables.pkl')
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

    
def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/antmaze/online_planner.yaml')
    parser.add_argument('--dreamer-config', type=str, default='dreamerv2/dreamerv2/configs.yaml')
    parser.add_argument('--exp-id', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="none")
    parser.add_argument('--model_pretraining', action='store_true', default=False)
    
    args, unknown = parser.parse_known_args()
    cli_args = parse_oc_args(unknown)
    
    print(f'Loading experiment config from {args.config}')

    # load configs  
    cfg = oc.load(args.config)
    cfg = oc.merge(cfg, cli_args)
    agent_cfg = cfg.planner
    world_model_cfg = cfg.world_model
    # make envs.
    train_seed = cfg.experiment.seed
    test_seed = 2**31 - 1 - cfg.experiment.seed


    device = f'cuda:{cfg.experiment.gpu}' if cfg.experiment.gpu >= 0 else 'cpu'

    # make ground environment
    train_env = make_ground_env(
                                    test=True,
                                    args=agent_cfg.env,
                                    gamma=agent_cfg.env.gamma,
                                    train_seed=train_seed,
                                    device=device,
                                )

    test_env = make_ground_env(
                                    test=True,
                                    args=agent_cfg.env,
                                    gamma=agent_cfg.env.gamma,
                                    train_seed=test_seed,
                                    device=device,
                                )


    # train_env = GymWrapper(train_env, obs_key='joints', act_key='direction')
    discrete = False
    configs = yaml.safe_load((
        pathlib.Path('/users/rrodri19/abs-mdp/src/agents') / 'configs.yaml').read_text())
    config = common.Config(configs['defaults'])
    config, unknown = config.update({
        'logdir': f'exp_results/dreamer/{agent_cfg.env.envname}_{args.exp_id}',
        'log_every': 1e3,
        'train_every': 10,
        'actor_ent': 3e-3,
        'loss_scales.kl': 1.0,
        'discount': 0.9995,
        'encoder.mlp_keys': '.*',
        'encoder.cnn_keys': '$^',
        'replay.minlen': 1,
        'steps': 1e6,
        'prefill': 512 * 64,
        'pretrain': 20000,
        'model_opt.lr': 3e-4,
        'actor_opt.lr': 1e-4,
        'critic_opt.lr': 1e-4,
        'kl.free': 1.0,
        'rssm': {'hidden': 200, 'deter': 200, 'discrete': discrete},
        'reward_head': {'dist': 'binary'},
        'eval_every': 1e4,
        'time_limit': 100,
        'eval_eps': 10,
        'clip_rewards': 'identity'
    }).parse_flags(known_only=True)

    pretrain_path = pathlib.Path(args.model_path).expanduser() if args.model_path != "none" else None
    train_dreamer(train_env, test_env, config, pretrain_path=pretrain_path, model_pretraining=args.model_pretraining)

if __name__ == '__main__':
    main()