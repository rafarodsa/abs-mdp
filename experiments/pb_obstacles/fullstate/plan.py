'''
    Plan with DDQN for Pinball
    author: Rafael Rodriguez Sanchez
'''
import argparse
import os

import torch
import numpy as np
from src.utils import printarr

from envs.pinball import PinballEnvContinuous, create_position_options
from envs import EnvGoalWrapper, EnvOptionWrapper, EnvInitsetWrapper, AbstractEnvWrapper

from omegaconf import OmegaConf as oc
from experiments import parse_oc_args
from experiments.run_ddqn import run_abstract_ddqn

import pfrl
from pfrl.q_functions import DiscreteActionValueHead
import yaml

from src.models import ModuleFactory
GOALS = [
    [0.6, 0.5, 0., 0.],
    [0.1, 0.1, 0., 0.],
    [0.1, 0.8, 0., 0.],
    [0.2, 0.9, 0., 0.],
    [0.9, 0.9, 0., 0.],
    [0.5, 0.1, 0., 0.],
    [0.3, 0.3, 0., 0.],
    [0.4, 0.6, 0., 0.],
    [0.9, 0.45, 0., 0.], # hard goal
    [0.9, 0.1, 0., 0.] # hard goal
]

GOAL = GOALS[0]
STEP_SIZE = 1/25
GOAL_TOL = 0.05
GOAL_REWARD = 1.
INITSET_THRESH = 0.5
ENV_CONFIG_FILE = 'envs/pinball/configs/pinball_simple_single.cfg'
GAMMA = 0.9997

def compute_nearest_pos(position, n_pos=20, min_pos=0.05, max_pos=0.95):
    # batch = position.shape[0] if len(position.shape) > 1 else 1
    step = (max_pos - min_pos) / n_pos
    pos_ = (position - min_pos) / (max_pos-min_pos) * n_pos
    min_i, max_i = np.floor(pos_), np.ceil(pos_) # closest grid position to move
    min = np.vstack([min_i, max_i])
    x, y = np.meshgrid(min[:,0], min[:, 1])
    pts = np.stack([x.reshape(-1), y.reshape(-1)], axis=1)
    distances =  np.linalg.norm(pts - pos_[np.newaxis], axis=-1)
    _min_dis = distances.argmax()
    goal = pts[_min_dis] * step + min_pos

    goal = goal.clip(min_pos, max_pos)
    return goal

def check_init_state(state, goal, step_size, tol):
    n_steps = np.abs(state[:2]-goal[:2])/step_size
    min_n = np.floor(n_steps)
    max_n = np.ceil(n_steps)
    return np.all(np.logical_or(np.abs(min_n-n_steps) <= tol, np.abs(max_n-n_steps) <= tol)) and state[1] < 0.1

def init_state_sampler(goal, step_size=1/15, tol=0.015):
    config = 'envs/pinball/configs/pinball_simple_single.cfg'
    pinball_env = PinballEnvContinuous(config)
    def __sampler():
        accepted = False
        while not accepted:
            s = pinball_env.sample_initial_positions(1)[0].astype(np.float32)
            accepted = np.linalg.norm(goal[:2]-s[:2]) >= 0.2
        return s
    return __sampler

# goal
def goal_fn(phi, goal=[0.2, 0.9, 0., 0.], goal_tol=0.01):
    goal = phi(torch.Tensor(goal)).numpy()
    # goal = compute_nearest_pos(goal)
    def _goal(state):
        distance =  np.linalg.norm(state[:2] - goal)
        return distance <= goal_tol
    return _goal


def ground_state_sampler(g_sampler, grounding_function, n_samples=1000, device='cpu'):
    
    g_samples = torch.from_numpy(g_sampler(n_samples).astype(np.float32)).to(device)
    def __sample(z):
        '''
            z : torch.Tensor (1,N)
        '''
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        b_size = z.shape[0]
        energy = grounding_function(g_samples.repeat(b_size, 1), z.repeat_interleave(n_samples, dim=0)).reshape(b_size, n_samples)
        max_energy = energy.max(-1)
        s = g_samples[max_energy.indices]
        return s
    return __sample

def gaussian_ball_goal_fn(phi, goal, goal_tol, n_samples=100, device='cpu'):
    goal = torch.Tensor(goal).unsqueeze(0)
    samples = torch.randn(n_samples, 4) * goal_tol
    encoded_samples = phi(samples + goal)
    encoded_goal = phi(goal)
    printarr(encoded_goal, encoded_samples)
    encoded_tol = torch.sqrt(((encoded_goal-encoded_samples) ** 2).sum(-1).mean())
    encoded_tol, encoded_goal = encoded_tol.to(device), encoded_goal.to(device)
    def __goal(z):
        _d = (((z-encoded_goal)/encoded_tol) ** 2).sum(-1)
        return torch.exp(-_d) > 0.3
    return __goal


def ground_action_mask(state, options, device):
    # print('Warning: Computing ground action mask')
    state = state.cpu()
    mask = torch.from_numpy(np.array([o.initiation(state[0].numpy()) for o in options]).T).float().to(device)
    return mask

def make_ground_init_from_latent(grounding, device):
    env = PinballEnvContinuous(config=ENV_CONFIG_FILE)
    options = CreateContinuousOptions(env)
    g_sampler = ground_state_sampler(env.sample_init_states, grounding, n_samples=1000, device=device)
    def __ground_initset(z):
        s = g_sampler(z)
        return ground_action_mask(s, options, device=device)
    return __ground_initset

def make_initset_from_classifier(initset_classifier, threshold, device='cpu'):
    def __initset(z):
        logits = initset_classifier(z).to(device)
        return (torch.sigmoid(logits) > threshold).float()
    return __initset



CreateContinuousOptions = lambda env: create_position_options(env, translation_distance=STEP_SIZE, check_can_execute=False)


def make_abstract_env(test=False, test_seed=127, train_seed=255, args=None, reward_scale=0., gamma=0.99, device='cpu', use_ground_init=False):
    print('================ CREATE ENVIRONMENT ================', GOAL_REWARD)
    env_sim = torch.load(args.absmdp)
    env_sim.to(device)
    env_sim._sample_state = args.sample_abstract_transition
    discounted = not test
    goal = GOALS[args.goal]
    print(f'GOAL: {goal[:2]} ± {GOAL_TOL}')
    env = EnvGoalWrapper(env_sim, 
                         goal_fn=gaussian_ball_goal_fn(env_sim.encoder, goal=goal, goal_tol=GOAL_TOL), 
                         goal_reward=GOAL_REWARD, 
                         init_state_sampler=init_state_sampler(goal), 
                         discounted=discounted, 
                         reward_scale=reward_scale, 
                         gamma=gamma)

    ### MAKE INITSET FUNCTION
    if use_ground_init:
        initset_fn = make_ground_init_from_latent(env_sim.grounding, device=device)
    else:
        initset_fn = make_initset_from_classifier(env_sim.init_classifier, device=device, threshold=INITSET_THRESH)
    env = EnvInitsetWrapper(env, initset_fn)
    env.seed(test_seed if test else train_seed)
    return env

def make_ground_env(test=False, test_seed=0, train_seed=1, args=None, reward_scale=0., gamma=0.99, use_ground_init=True, initset_fn=None, device='cpu'):
    print('================ CREATE GROUND ENVIRONMENT ================', GOAL_REWARD)
    # evaluation in real env
    discounted = not test
    goal = GOALS[args.goal]
    print(f'GOAL: {goal[:2]} ± {GOAL_TOL}')
    env = PinballEnvContinuous(config=ENV_CONFIG_FILE, gamma=gamma)
    options = CreateContinuousOptions(env)
    env = EnvOptionWrapper(options, env, discounted=discounted)
    env = EnvGoalWrapper(
                            env, 
                            goal_fn=goal_fn(lambda s: s[..., :2], goal=goal, goal_tol=GOAL_TOL), 
                            goal_reward=GOAL_REWARD, 
                            init_state_sampler=init_state_sampler(goal), 
                            discounted=discounted, 
                            reward_scale=reward_scale,
                            gamma=gamma
                        )
    
    if not use_ground_init: # use initset from abstract agent
        assert initset_fn is not None, 'Must provide initset function'
        env = EnvInitsetWrapper(env, initset_fn)
    env.seed(test_seed if test else train_seed)
    return env

def parse_oc_args(oc_args):
    assert len(oc_args)%2==0
    oc_args = ['='.join([oc_args[i].split('--')[-1], oc_args[i+1]]) for i in range(len(oc_args)) if i%2==0]
    cli_config = oc.from_cli(oc_args)
    return cli_config

def numpy_adaptor(function, device='cpu'):
    def __f(s):
        if isinstance(s, torch.Tensor):
            return function(s)
        else:
            return function(torch.from_numpy(s).to(device))
    return __f

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
    initset_fn = None
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
        initset_fn = env.initset
        sim_encoder = env.env.encoder


    state_dim = env.observation_space.shape[0]
    # make eval envs
    eval_envs = {}
    if not args.experiment.finetune:
        ground_eval_env = make_ground_env(
                                test=True,
                                test_seed=test_seed, 
                                args=args.env, 
                                reward_scale=args.env.reward_scale,
                                gamma=args.env.gamma,
                                device = device,
                                use_ground_init=args.env.use_ground_init or args.env.absgroundmdp,
                                initset_fn=numpy_adaptor(lambda s: initset_fn(sim_encoder(s)), device=device)
                            )
    
        if not args.env.absgroundmdp: # not training the ground agent.
            sim_eval_env = make_abstract_env(
                                        test=True,
                                        test_seed=test_seed, 
                                        args=args.env, 
                                        reward_scale=args.env.reward_scale,
                                        gamma=args.env.gamma,
                                        device = device,
                                        use_ground_init=args.env.use_ground_init
                                    )
            ground_eval_env = AbstractEnvWrapper(ground_eval_env, lambda s: sim_eval_env.env.encoder(torch.from_numpy(s))) # abstract agent sees abstract env (AA -> GE)
            eval_envs['sim'] = sim_eval_env

    # finetuning
    if args.experiment.finetune:
        # assert args.agent.load is not None, 'Must provide a model to finetune'
        assert args.env.absmdp is not None, 'Must provide the abstract MDP the agent planned on'

        # look for finished experiment
        if args.agent.load is None:
            loading_path = f'{args.experiment_cwd}/{args.experiment_name}/{args.experiment.outdir}/{cli_args.exp_id}'
            # search for subdir that endsi in _finish
            found = False
            for d in os.listdir(loading_path):
                if d.endswith('_finish'):
                    loading_path = f'{loading_path}/{d}'
                    found = True
                    break

            assert found or args.agent.load is not None, f'Could not find a finished experiment in {loading_path}'
            print(f'Loading from {loading_path}')
        
            args.agent.load = loading_path
        

        sim_eval_env = make_abstract_env(
                            test=True,
                            test_seed=test_seed, 
                            args=args.env, 
                            reward_scale=args.env.reward_scale,
                            gamma=args.env.gamma,
                            device = device,
                            use_ground_init=args.env.use_ground_init
                        )
        
        # encoder = lambda s: sim_eval_env.env.encoder(s) if isinstance(s, torch.Tensor) else sim_eval_env.env.encoder(torch.from_numpy(s).to(device))  # get encoder
        encoder = sim_eval_env.env.encoder
        # initset_fn = sim_eval_env.initset

        initset_fn = numpy_adaptor(lambda s: sim_eval_env.initset(encoder(s)), device=device)
        
        ground_eval_env = make_ground_env(
                                test=True,
                                test_seed=test_seed, 
                                args=args.env, 
                                reward_scale=args.env.reward_scale,
                                gamma=args.env.gamma,
                                device = device,
                                use_ground_init=args.env.use_ground_init,
                                initset_fn=initset_fn
                            )
        env = EnvInitsetWrapper(env, initset_fn) # change initset for the ground finetuning environment
        state_dim = sim_eval_env.observation_space.shape[0]


        
    eval_envs['ground'] = ground_eval_env
    envs = {'train': env, 'eval': eval_envs} 

    # prepare outdir
    outdir = f'{args.experiment_cwd}/{args.experiment_name}/{args.experiment.outdir}' if not args.experiment.finetune else f'{args.experiment_cwd}/{args.experiment_name}/{args.experiment.outdir}_finetuning'
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
    run_abstract_ddqn(envs, q_func, encoder, args.agent, args.experiment, finetuning_args=args.finetuning, device=device)


if __name__ == '__main__':
    run()