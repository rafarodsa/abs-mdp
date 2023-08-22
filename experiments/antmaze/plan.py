'''
    Plan with DDQN for Pinball
    author: Rafael Rodriguez Sanchez
'''
import argparse

import torch
import numpy as np
from src.utils import printarr

from envs import EnvGoalWrapper, EnvOptionWrapper, EnvInitsetWrapper, AbstractEnvWrapper

from experiments.antmaze.utils.make_antmaze_with_options import make_antmaze_options, make_antmaze


from omegaconf import OmegaConf as oc
from experiments import parse_oc_args
from experiments.run_ddqn import run_abstract_ddqn

import pfrl
from pfrl.q_functions import DiscreteActionValueHead
import yaml




from src.models import ModuleFactory
GOALS = [
    [3., 0.],
    [5., 0.],
    [8., 0.],
    [8., 2.5],
    [8., 5.],
    [8., 7.5],
    [5., 7.5],
    [3., 7.5],
    [0., 7.5]
]

GOAL = GOALS[0]
STEP_SIZE = 1/25
GOAL_TOL = 0.5
GOAL_REWARD = 1.
INITSET_THRESH = 0.5
ENV_CONFIG_FILE = 'envs/pinball/configs/pinball_simple_single.cfg'
GAMMA = 0.998



def goal_fn(goal, goal_tol):
    def __goal(s):
        return np.linalg.norm(s[:2] - goal[:2]) <= goal_tol
    return __goal


def make_abstract_env(test=False, test_seed=127, train_seed=255, args=None, reward_scale=0., gamma=0.99, device='cpu', use_ground_init=False):
    raise NotImplemented 

def make_ground_env(test=False, test_seed=0, train_seed=1, args=None, reward_scale=0., gamma=GAMMA, device='cpu'):
    options, initset = make_antmaze_options(args.envname, device=device)
    goal = GOALS[args.goal]
    print(f'Goal: {goal}')
    discounted = not test
    env = make_antmaze(args.envname, test_seed if test else train_seed, options, initset, device=device)
    # env.set_goal(np.array(goal))
    env = EnvGoalWrapper(env, goal_fn(goal, GOAL_TOL), goal_reward=GOAL_REWARD, discounted=discounted, gamma=gamma)
    return env




def run():

    ## get config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/pb_obstacles/fullstate/configs/ddqn.yaml')
    parser.add_argument('--exp-id', type=str, default=None)
    cli_args, unknown = parser.parse_known_args()
    cli_cfg = parse_oc_args(unknown)
    cfg = oc.load(cli_args.config)
    args = oc.merge(cfg, cli_cfg)

    device = f'cuda:{args.experiment.gpu}' if args.experiment.gpu >= 0 else 'cpu'
    # make envs.
    train_seed = args.experiment.seed
    test_seed = 2**31 - 1 - args.experiment.seed

    encoder = lambda s: s

    ## make envs.
    if args.env.absgroundmdp or args.experiment.finetune:
        env = make_ground_env(
                                train_seed=train_seed, 
                                args=args.env, 
                                reward_scale=args.env.reward_scale,
                                gamma=args.env.gamma,
                                device = device
                            )
    else:
        env = make_abstract_env(
                                train_seed=train_seed, 
                                args=args.env, 
                                reward_scale=args.env.reward_scale,
                                gamma=args.env.gamma,
                                device = device,
                                use_ground_init=args.env.use_ground_init
                            )
    state_dim = env.observation_space.shape[0]
    # make eval envs
    eval_envs = {}
    ground_eval_env = make_ground_env(
                            test=True,
                            test_seed=test_seed, 
                            args=args.env, 
                            reward_scale=args.env.reward_scale,
                            gamma=args.env.gamma,
                            device = device
                        )
    
    if not args.env.absgroundmdp and not args.experiment.finetune: # not training the ground agent.
        sim_eval_env = make_abstract_env(
                                    test=True,
                                    test_seed=test_seed, 
                                    args=args.env, 
                                    reward_scale=args.env.reward_scale,
                                    gamma=args.env.gamma,
                                    device = device,
                                    use_ground_init=args.env.use_ground_init
                                )
        
        ground_eval_env = AbstractEnvWrapper(ground_eval_env, lambda s: sim_eval_env.env.encoder(torch.from_numpy(s)))
        eval_envs['sim'] = sim_eval_env
    
    
    eval_envs['ground'] = ground_eval_env

    envs = {'train': env, 'eval': eval_envs}

    # finetuning
    if args.experiment.finetune:
        assert args.agent.load is not None, 'Must provide a model to finetune'
        assert args.env.absmdp is not None, 'Must provide the abstract MDP the agent planned on'
        sim_eval_env = make_abstract_env(
                            test_seed=test_seed, 
                            args=args.env, 
                            reward_scale=args.env.reward_scale,
                            gamma=args.env.gamma,
                            device = device,
                            use_ground_init=args.env.use_ground_init
                        )
        # encoder = lambda s: sim_eval_env.env.encoder(s) if isinstance(s, torch.Tensor) else sim_eval_env.env.encoder(torch.from_numpy(s).to(device))  # get encoder
        encoder = sim_eval_env.env.encoder
        state_dim = sim_eval_env.observation_space.shape[0]
        

    # prepare outdir
    outdir = f'{args.experiment_cwd}/{args.experiment_name}/planning_ddqn' if not args.experiment.finetune else f'{args.experiment_cwd}/{args.experiment_name}/planning_ddqn/finetuning'
    outdir = pfrl.experiments.prepare_output_dir(None, outdir, exp_id=cli_args.exp_id, make_backup=False)
    args.experiment.outdir = outdir
    print(f'Logging to {outdir}')

    # make model

    cfg.q_func.input_dim = state_dim
    q_func = ModuleFactory.build(cfg.q_func)
    q_func = torch.nn.Sequential(q_func, DiscreteActionValueHead())

    # save config file to reproduce
    print(f'Saving config to {outdir}/config.yaml')
    _args = oc.to_container(args, resolve=True)
    with open(f'{outdir}/config.yaml', 'w') as f:
        yaml.dump(_args, f, default_flow_style=False)

    # train
    run_abstract_ddqn(envs, q_func, encoder, args.agent, args.experiment, device=device)


if __name__ == '__main__':
    run()