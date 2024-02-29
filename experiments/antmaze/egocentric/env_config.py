import numpy as np

from experiments.antmaze.egocentric.maze import EgocentricMaze
from experiments.antmaze.egocentric.env_adaptor import EmbodiedEnv, GroundTruthEnvWrapper
from envs.env_options import EnvOptionWrapper
from envs.env_goal import EnvGoalWrapper


GOALS = {
    'ant_maze_xl': [[14, 14], [8, 14], [2, 14], [14, 8], [2, 8]],
    'ant_empty': [[14, 14], [8, 14], [2, 14], [8, 9], [2, 8]],
    'ant_maze_s': [[8, 4] ],
    'ant_maze_m': [[2, 5], [8, 5], [2,8], [8,8]]
}

def get_pose(env):
    physics = env._env._env._physics
    return list(map(np.array, env._walker.get_pose(physics)))
        
def make_reward_function(goal, env, tol=0.5):
    target_pos = np.array(env._arena.grid_to_world_positions([goal])[0][:2])
    print(f'Goal Position: {target_pos}')
    def reward_fn(obs):
        pos = np.array(get_pose(env)[0])[:2]
        return ((pos - target_pos) ** 2).sum() < tol ** 2
    return reward_fn

def make_egocentric_maze(name, goal, test=False, gamma=0.995, test_seed=None, train_seed=None, reward_scale=0., include_stop=False):
    
    from experiments.antmaze.egocentric.options import make_options
    # goal space.
    assert name in GOALS and len(GOALS[name]) > goal, f'Goal {goal} in maze {name} not defined!'
    goal = GOALS[name][goal]
    print(f'ENV: {name}, GOAL: {goal}')
    base_env = EgocentricMaze(name, goal, termination=True) 
    env = EmbodiedEnv(base_env, ignore_obs_keys=['walker/egocentric_camera'])
    options = list(make_options(base_env, max_exec_time=100, include_stop=include_stop).values()) # mapping name->option
    task_reward = make_reward_function(goal, base_env, tol=1.8)
    env = EnvOptionWrapper(options, env, discounted=(not test))
    env = EnvGoalWrapper(env, task_reward, discounted=False, gamma=gamma, reward_scale=reward_scale)
    return env


def make_egocentric_maze_ground_truth(name, goal, test=False, gamma=0.995, test_seed=None, train_seed=None, reward_scale=0., include_stop=False):
    
    from experiments.antmaze.egocentric.options import make_options
    # goal space.
    assert name in GOALS and len(GOALS[name]) > goal, f'Goal {goal} in maze {name} not defined!'
    goal = GOALS[name][goal]
    print(f'ENV: {name}, GOAL: {goal}')
    base_env = EgocentricMaze(name, goal, termination=True) 
    env = GroundTruthEnvWrapper(base_env, ignore_obs_keys=['walker/egocentric_camera'])
    options = list(make_options(base_env, max_exec_time=100, include_stop=include_stop).values()) # mapping name->option
    task_reward = make_reward_function(goal, base_env, tol=1.8)
    env = EnvOptionWrapper(options, env, discounted=(not test))
    env = EnvGoalWrapper(env, task_reward, discounted=False, gamma=gamma, reward_scale=reward_scale)
    return env