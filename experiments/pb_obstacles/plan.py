'''
    Plan with DDQN for Pinball
    author: Rafael Rodriguez Sanchez
'''
import argparse
import os
import logging 

import torch
import numpy as np
from src.utils import printarr

from envs.pinball import PinballEnvContinuous, create_position_options
from envs import EnvGoalWrapper, EnvOptionWrapper, EnvInitsetWrapper, AbstractEnvWrapper
from envs.pinball.pinball_gym import PinballPixelWrapper

from omegaconf import OmegaConf as oc
from experiments import parse_oc_args
from experiments.run_ddqn import run_abstract_ddqn

from src.absmdp.mdp import NormalizedObsWrapper
from src.agents.multiprocess_env import MultiprocessVectorEnv

import pfrl
from pfrl.q_functions import DiscreteActionValueHead
import yaml

from src.models import ModuleFactory

import optuna

from functools import partial


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

def init_state_sampler(goal, step_size=1/15, tol=0.015, from_pixels=False):
    config = 'envs/pinball/configs/pinball_simple_single.cfg'
    pinball_env = PinballEnvContinuous(config, width=50, height=50, render_mode='rgb_array')
    if from_pixels:
        env = PinballPixelWrapper(pinball_env, bw=True)
    def __sampler():
        accepted = False
        while not accepted:
            s = pinball_env.sample_initial_positions(1)[0].astype(np.float32)
            accepted = np.linalg.norm(goal[:2]-s[:2]) >= 0.2

        if from_pixels:
            s = env.reset(s).astype(np.float32)
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

def sample_points_in_circle(r, n):
    # Choose n radii between 0 and r
    s = r * np.sqrt(np.random.rand(n))
    # Choose n angles between 0 and 2*pi
    theta = 2 * np.pi * np.random.rand(n)
    # Convert polar coordinates to Cartesian coordinates
    x = s * np.cos(theta)
    y = s * np.sin(theta)
    return np.column_stack((x, y))


def test_task_reward(goal, goal_tol, phi, abstract_reward_fn, env, device, n_samples=10000):
    import matplotlib.pyplot as plt
    samples_obs = []
    samples_s = []
    for _ in range(n_samples):
        obs = env.reset()
        s = env.pinball.get_state()
        samples_obs.append(obs)
        samples_s.append(s)
    samples_obs = np.stack(samples_obs).astype(np.float32)
    samples_s = np.stack(samples_s).astype(np.float32)

    
    with torch.no_grad():
        encoded_samples = phi(torch.from_numpy(samples_obs)).to(device)

    real_goal = ((samples_s[:, :2] - np.array(goal)[:2]) ** 2).sum(-1) < GOAL_TOL **2 

    abstract_goal = np.array([abstract_reward_fn(encoded_samples[i]).cpu().numpy() for i in range(n_samples)])


    tpr = (real_goal * abstract_goal).sum() / real_goal.sum()
    tnr = ((1-real_goal) * (1-abstract_goal)).sum() / (1-real_goal).sum()
    acc = (real_goal == abstract_goal).mean()
    print(f'TPR: {tpr}. TNR: {tnr}, ACC: {acc}')

    printarr(abstract_goal, real_goal, samples_s, real_goal, abstract_goal)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(samples_s[:, 0], samples_s[:, 1], c=abstract_goal, s=1)
    plt.title('Abstract Goal')
    plt.subplot(1, 2, 2)
    plt.scatter(samples_s[:, 0], samples_s[:, 1], c=real_goal.astype(np.float32), s=1)
    plt.title('Real Goal')
    plt.savefig('task_reward.png')



def gaussian_ball_goal_fn(phi, goal, goal_tol, n_samples=1000, device='cpu', env=None):        
    goal = np.array(goal).astype(np.float32)
    samples = np.concatenate([sample_points_in_circle(goal_tol, n_samples), np.random.rand(n_samples, 2)-0.5], axis=-1) + goal[None]
    samples = samples.astype(np.float32)
    printarr(goal,  samples)
    _g = goal
    if env is not None:
        goal = env.reset(goal)
        samples = np.array([env.reset(samples[i]) for i in range(n_samples)])
    goal = torch.from_numpy(goal)
    samples = torch.from_numpy(samples)
    encoded_samples = phi(samples)
    encoded_goal = phi(goal)
    printarr(encoded_goal, encoded_samples)
    encoded_tol = torch.sqrt(((encoded_goal-encoded_samples) ** 2).sum(-1).mean())
    encoded_tol, encoded_goal = encoded_tol.to(device), encoded_goal.to(device)
    def __goal(z):
        _d = (((z-encoded_goal)/encoded_tol) ** 2).sum(-1)
        return torch.exp(-_d) > 0.22

    
    test_task_reward(_g, goal_tol, phi, __goal, env, device)
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


def make_abstract_env(test=False, test_seed=127, train_seed=255, args=None, reward_scale=0., gamma=0.99, device='cpu', use_ground_init=False, goal_fn=None, from_pixels=False):
    assert goal_fn is not None, 'Must provide a goal function to create the abstract env'

    print('================ CREATE ENVIRONMENT ================', GOAL_REWARD)
    env_sim = torch.load(args.absmdp)
    env_sim.to(device)
    env_sim._sample_state = args.sample_abstract_transition
    discounted = not test
    goal = GOALS[args.goal]
    print(f'GOAL[{args.goal}]: {goal[:2]} ± {GOAL_TOL}')
    env = EnvGoalWrapper(env_sim, 
                         goal_fn=goal_fn(env_sim.encoder, goal=goal, goal_tol=GOAL_TOL), 
                         goal_reward=GOAL_REWARD, 
                         init_state_sampler=init_state_sampler(goal, from_pixels=from_pixels), 
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

def make_ground_env(test=False, test_seed=0, train_seed=1, args=None, reward_scale=0., gamma=0.99, use_ground_init=True, initset_fn=None, device='cpu', from_pixels=False):
    print('================ CREATE GROUND ENVIRONMENT ================', GOAL_REWARD)
    # evaluation in real env
    discounted = not test
    goal = GOALS[args.goal]
    print(f'GOAL: {goal[:2]} ± {GOAL_TOL}')
    env = PinballEnvContinuous(config=ENV_CONFIG_FILE, gamma=gamma, width=50, height=50, render_mode='rgb_array')
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
    if from_pixels:
        env = PinballPixelWrapper(env, bw=True)
    env.seed(test_seed if test else train_seed)
    return env

def make_batched_ground_env(make_env, num_envs, seeds):
    vec_env = MultiprocessVectorEnv(
        [
            partial(make_env, train_seed=seeds[i])
            for i in range(num_envs)
        ]
    )
    return vec_env


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


def get_params():
    parser = argparse.ArgumentParser()

    # Main arguments
    parser.add_argument('script', type=str, choices=['plan', 'tune'], default='plan')
    parser.add_argument('--config', type=str, default='experiments/pb_obstacles/pixel/config/ddqn_ground.yaml')
    parser.add_argument('--exp-id', type=str, default=None)
    parser.add_argument('--pixels', action='store_true', default=False, help='Use pixels as input')
    parser.add_argument('--batch-env', type=int, default=1, help='Number of ground environments to run in parallel')
    # Add specific arguments for 'tune'
    parser.add_argument('--tuner-config', type=str, default='experiments/pb_obstacles/fullstate/config/tune.yaml')
    
    cli_args, unknown = parser.parse_known_args()
    cli_cfg = parse_oc_args(unknown)
    cfg = oc.load(cli_args.config) # base planner config
    planner_config = oc.merge(cfg, cli_cfg)

    return planner_config, cli_args

def run(planner_config, cli_args, tune=False, trial=None):
    if tune:
        assert trial is not None, 'Must provide a trial object'

    ## get config
    device = f'cuda:{planner_config.experiment.gpu}' if planner_config.experiment.gpu >= 0 else 'cpu'

    # make envs.
    train_seed = planner_config.experiment.seed

        
    if cli_args.batch_env > 1:
        # Set different random seeds for different subprocesses.
        # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
        # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
        process_seeds = np.arange(cli_args.batch_env).astype(np.int64) + planner_config.experiment.seed * cli_args.batch_env
        print(f'Process seeds: {process_seeds}')
        assert process_seeds.max() < 2**32
        process_seeds = [int(s) for s in process_seeds]
       
    
    test_seed = 2**31 - 1 - planner_config.experiment.seed

   
    abstract_goal_fn = gaussian_ball_goal_fn
    encoder = lambda s: s
    initset_fn = None
    ## make training envs.

    ground_env = make_ground_env(
                        train_seed=train_seed, 
                        args=planner_config.env, 
                        reward_scale=planner_config.env.reward_scale,
                        gamma=planner_config.env.gamma,
                        device = device, 
                        from_pixels=cli_args.pixels,
                    )
    if cli_args.pixels:
        abstract_goal_fn = partial(gaussian_ball_goal_fn, env=ground_env)
    
    if planner_config.env.absgroundmdp or planner_config.experiment.finetune:
        if cli_args.batch_env == 1:
            env = ground_env        
        else:
            
            env_maker = partial(make_ground_env, args=planner_config.env, reward_scale=planner_config.env.reward_scale, gamma=planner_config.env.gamma, device=device, from_pixels=cli_args.pixels)
            env = make_batched_ground_env(env_maker, cli_args.batch_env, process_seeds)

    else:
        env = make_abstract_env(
                                train_seed=train_seed, 
                                args=planner_config.env, 
                                reward_scale=planner_config.env.reward_scale,
                                gamma=planner_config.env.gamma,
                                device = device,
                                use_ground_init=planner_config.env.use_ground_init,
                                goal_fn=abstract_goal_fn,
                                from_pixels=cli_args.pixels
                            )
        initset_fn = env.initset
        sim_encoder = env.env.encoder

    # make eval envs
    eval_envs = {}
    if not planner_config.experiment.finetune:
        if cli_args.pixels:
            _initset_fn = None
        else:
            _initset_fn = numpy_adaptor(lambda s: initset_fn(sim_encoder(s)), device=device)
       
        ground_eval_env = make_ground_env(
                                test=True,
                                test_seed=test_seed, 
                                args=planner_config.env, 
                                reward_scale=planner_config.env.reward_scale,
                                gamma=planner_config.env.gamma,
                                device = device,
                                # use_ground_init=planner_config.env.use_ground_init or planner_config.env.absgroundmdp,
                                # initset_fn=_initset_fn,
                                from_pixels=cli_args.pixels
                            )
    
        if not planner_config.env.absgroundmdp: # not training the ground agent.
            sim_eval_env = make_abstract_env(
                                        test=True,
                                        test_seed=test_seed, 
                                        args=planner_config.env, 
                                        reward_scale=planner_config.env.reward_scale,
                                        gamma=planner_config.env.gamma,
                                        device = device,
                                        use_ground_init=planner_config.env.use_ground_init,
                                        goal_fn=abstract_goal_fn,
                                        from_pixels=cli_args.pixels
                                    )
            ground_eval_env = AbstractEnvWrapper(ground_eval_env, lambda s: sim_eval_env.env.encoder(torch.from_numpy(s))) # abstract agent sees abstract env (AA -> GE)
            if not tune:
                eval_envs['sim'] = sim_eval_env


    state_dim = env.observation_space.shape[0]
    # make eval envs

    # finetuning
    if planner_config.experiment.finetune:
        # assert args.agent.load is not None, 'Must provide a model to finetune'
        assert planner_config.env.absmdp is not None, 'Must provide the abstract MDP the agent planned on'

        # look for finished experiment
        if planner_config.agent.load is None:
            loading_path = f'{planner_config.experiment_cwd}/{planner_config.experiment_name}/{planner_config.experiment.outdir}/{cli_args.exp_id}'
            # search for subdir that endsi in _finish
            found = False
            for d in os.listdir(loading_path):
                if d.endswith('_finish'):
                    loading_path = f'{loading_path}/{d}'
                    found = True
                    break

            assert found or planner_config.agent.load is not None, f'Could not find a finished experiment in {loading_path}'
            print(f'Loading from {loading_path}')
        
            planner_config.agent.load = loading_path
        

        sim_eval_env = make_abstract_env(
                            test=True,
                            test_seed=test_seed, 
                            args=planner_config.env, 
                            reward_scale=planner_config.env.reward_scale,
                            gamma=planner_config.env.gamma,
                            device = device,
                            use_ground_init=planner_config.env.use_ground_init,
                            goal_fn=abstract_goal_fn,
                            from_pixels=cli_args.pixels
                        )
        
        # encoder = lambda s: sim_eval_env.env.encoder(s) if isinstance(s, torch.Tensor) else sim_eval_env.env.encoder(torch.from_numpy(s).to(device))  # get encoder
        encoder = sim_eval_env.env.encoder
        # initset_fn = sim_eval_env.initset

        initset_fn = numpy_adaptor(lambda s: sim_eval_env.initset(encoder(s)), device=device)
        
        ground_eval_env = make_ground_env(
                                test=True,
                                test_seed=test_seed, 
                                args=planner_config.env, 
                                reward_scale=planner_config.env.reward_scale,
                                gamma=planner_config.env.gamma,
                                device = device,
                                use_ground_init=planner_config.env.use_ground_init,
                                initset_fn=initset_fn,
                                from_pixels=cli_args.pixels
                            )
        env = EnvInitsetWrapper(env, initset_fn) # change initset for the ground finetuning environment
        state_dim = sim_eval_env.observation_space.shape[0]


        
    eval_envs['ground'] = ground_eval_env
    envs = {'train': env, 'eval': eval_envs} 

    # prepare outdir
    outdir = f'{planner_config.experiment_cwd}/{planner_config.experiment_name}/{planner_config.experiment.outdir}' if not planner_config.experiment.finetune else f'{planner_config.experiment_cwd}/{planner_config.experiment_name}/{planner_config.experiment.outdir}_finetuning'
    outdir = pfrl.experiments.prepare_output_dir(None, outdir, exp_id=cli_args.exp_id, make_backup=False)
    planner_config.experiment.outdir = outdir
    print(f'Logging to {outdir}')

    # make model
    if planner_config.env.absgroundmdp:
        planner_config.q_func.ground.input_dim = state_dim # needed for MLPs
        q_func = ModuleFactory.build(planner_config.q_func.ground)
        q_func = torch.nn.Sequential(q_func, DiscreteActionValueHead())
    else:
        planner_config.q_func.abstract.input_dim = state_dim
        q_func = ModuleFactory.build(planner_config.q_func.abstract)
        q_func = torch.nn.Sequential(q_func, DiscreteActionValueHead())
    
    

    # save config file to reproduce
    print(f'Saving config to {outdir}/config.yaml')
    _args = oc.to_container(planner_config, resolve=True)
    with open(f'{outdir}/config.yaml', 'w') as f:
        yaml.dump(_args, f, default_flow_style=False)

    # train

    if planner_config.agent.normalize_obs:
        def __normalizer(env):
            s = [env.reset() for _ in range(10000)]
            s = np.stack(s)
            mean, std = s.mean(0), s.std(0)
            return lambda s: (s-mean)/std
        normalizer = __normalizer(env)
    else:
        print('not normalizing')
        normalizer = lambda x: x

    print(os.getpid(), 'PID')

    scores = run_abstract_ddqn(
                                envs, 
                                q_func, 
                                encoder, 
                                planner_config.agent, 
                                planner_config.experiment, 
                                finetuning_args=planner_config.finetuning, 
                                normalizer=normalizer, 
                                device=device, 
                                tune=tune, 
                                trial=trial,
                                batched=cli_args.batch_env > 1,
                                example_env=ground_env
                            )
    return scores


### 

class OptunaTrainingStepsBudgetCallback:
    def __init__(self, training_steps_budget, logger=None):
        self.training_steps_budget = training_steps_budget
        self.logger = logger or logging.getLogger(__name__)

    def __call__(self, study, trial):
        training_steps = sum(
            trial.last_step
            for trial in study.get_trials()
            if trial.last_step is not None
        )
        self.logger.info(
            "{} / {} (sum of training steps / budget)".format(
                training_steps, self.training_steps_budget
            )
        )
        if training_steps >= self.training_steps_budget:
            study.stop()



def setup_optuna(config):
    sampler = optuna.samplers.TPESampler(seed=config.seed)

     # pruner
    if config.pruner.type == "NopPruner":
        pruner = optuna.pruners.NopPruner()
    elif config.pruner.type == "ThresholdPruner":
        pruner = optuna.pruners.ThresholdPruner(
            lower=config.pruner.lower,
            n_warmup_steps=config.pruner.n_warmup_steps,
        )
    elif config.pruner.type == "PercentilePruner":
        pruner = optuna.pruners.PercentilePruner(
            percentile=config.pruner.percentile,
            n_startup_trials=config.pruner.n_startup_trials,
            n_warmup_steps=config.pruner.n_warmup_steps,
        )
    elif config.pruner.type == "HyperbandPruner":
        pruner = optuna.pruners.HyperbandPruner(min_resource=config.pruner.eval_interval)
    
    print(f'Using pruner {config.pruner}')
    print(f'Study storage: {config.storage}')
    storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(f'{config.storage}/{config.study_name}.db')
    )
    study = optuna.create_study(
        study_name=config.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=config.direction,
        load_if_exists=True,
    )

    callbacks = [
        OptunaTrainingStepsBudgetCallback(
            training_steps_budget=config.steps_budget,
        ),
    ]

    return study, callbacks

def _get_score_from_eval_stats_history(
    eval_stats_history, agg="last", target="eval_score", evaluator_idx=0
):
    """Get a scalar score from a list of evaluation statistics dicts."""
    final_score = None
    if agg == "last":
        for stats in reversed(eval_stats_history):
            stats = stats[evaluator_idx]
            if target in stats:
                final_score = stats[target]
                break
    elif agg == "mean":
        scores = []
        for stats in eval_stats_history:
            stats = stats[evaluator_idx]
            if target in stats:
                score = stats[target]
                if score is not None:
                    scores.append(score)
        if len(scores) > 0:
            final_score = sum(scores) / len(scores)
        else:
            final_score = -1e12
    elif agg == "best":
        scores = []
        for stats in eval_stats_history:
            stats = stats[evaluator_idx]
            if target in stats:
                score = stats[target]
                if score is not None:
                    scores.append(score)
        final_score = max(scores)  # Assuming larger is better
    else:
        raise ValueError("Unknown agg method: {}".format(agg))

    if final_score is None:
        final_score = float("NaN")
    return final_score

def tune(planner_config, cli_args):

    # load optuna config
    optuna_config = oc.load(cli_args.tuner_config)
    optuna_config.study_name = f'{optuna_config.study_name}_{planner_config.experiment_name}'
    # setup optuna
    study, callbacks = setup_optuna(optuna_config)


    # define objective
    def _tuning_objective(trial):
        
        params_to_tune = oc.to_container(optuna_config.params_to_tune, resolve=False)
        params = []

        for param_name, config in params_to_tune.items():
            _type, _cfg = config[0], config[1:]

            if _type == 'int':
                low, high= _cfg[0]
                log = False if len(_cfg) == 1 or _cfg[1] != 'log_uniform' else True
                value = trial.suggest_int(param_name, low, high, log=log)
            elif _type == 'float':
                low, high= _cfg[0]
                log = False if len(_cfg) == 1 or _cfg[1] != 'log_uniform' else True
                value = trial.suggest_float(param_name, low, high, log=log)
            elif _type == 'categorical':
                value = trial.suggest_categorical(param_name, choices=_cfg[0])
            else:
                raise NotImplementedError(f'Unknown type {_type}')
            params.append(f'{param_name}={value}')

        fixed_params = oc.to_container(optuna_config.planner_params, resolve=False)
        params +=[f'{k}={v}' for k,v in fixed_params.items()]
        param_str = ' | '.join(params)
        print(f"Tuning with params: \t {param_str}")
        # update planner config with sample params
        _planner_cfg = oc.merge(planner_config, oc.from_dotlist(params))
        # run
        eval_stats_history = run(_planner_cfg, cli_args, tune=True, trial=trial)
        # get score
        target = 'eval_score'
        final_score = _get_score_from_eval_stats_history(eval_stats_history=eval_stats_history, agg=optuna_config.aggregator, target=target)

        return final_score

    study.optimize(_tuning_objective, callbacks=callbacks, n_jobs=optuna_config.n_jobs, n_trials=optuna_config.n_trials)

if __name__ == '__main__':

    config, cli_args = get_params()
    if cli_args.script == 'plan':
        run(config, cli_args=cli_args)
    elif cli_args.script == 'tune':
        tune(config, cli_args=cli_args)
    else:
        raise NotImplementedError(f'Unknown script {cli_args.script}')
    