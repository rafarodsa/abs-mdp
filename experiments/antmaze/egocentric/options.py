
import pathlib

import numpy as np
import ruamel.yaml as yaml

from dreamerv3 import embodied
from dreamerv3.agent import Agent

from envs.antmaze.ant_school import EgocentricMazeSchool
from src.options.option import Option
from tqdm import tqdm
from dm_control.utils.transformations import quat_rotate

OPTIONS_INFO = {
    'walk': {
        'path': 'exp_results/antmaze/egocentric/walk_empty_arena_vel2_1000_xxs',
        'type': 'walk'
    },
    
    'rotateccw': {
        'path': 'exp_results/antmaze/egocentric/dreamerv3_antmaze_ego_rotate_empty_arena_vel_5_xxs_2__task_ant_rotateccw',
        'type': 'rotate',
        'rotation': -0.39
    },
    'rotatecw':
     {
        'path':'exp_results/antmaze/egocentric/dreamerv3_antmaze_ego_rotate_empty_arena_vel_5_xxs_1__task_ant_rotatecw',
        'type': 'rotate', 
        'rotation': 0.39
    }
}

CONVERSION = {
    np.floating: np.float32,
    np.signedinteger: np.int64,
    np.uint8: np.uint8,
    bool: bool,
}
def convert(value):
  value = np.asarray(value)
  if value.dtype not in CONVERSION.values():
    for src, dst in CONVERSION.items():
      if np.issubdtype(value.dtype, src):
        if value.dtype != dst:
          value = value.astype(dst)
        break
    else:
      raise TypeError(f"Object '{value}' has unsupported dtype: {value.dtype}")
  return value

def orientation(quat):
    x = quat_rotate(quat, np.array([1., 0., 0.]))
    return np.arctan2(x[1], x[0])

def angle_error(angle1, angle2):
    dif1 = np.mod(angle1-angle2, np.pi * 2)
    dif2 = np.mod(angle2-angle1, np.pi * 2)
    return np.minimum(dif1, dif2)


def get_pose(env):
    physics = env._env._env._physics
    return list(map(np.array, env._walker.get_pose(physics)))

def load_agent(env, path):
    # load config
    path = embodied.Path(path)
    ckpt =  path / 'checkpoint.ckpt'
    config = path / 'config.yaml'
    config = embodied.Config.load(pathlib.Path(str(config)))
    config = config.update({'jax.platform': 'cpu', 'jax.jit': True}) # load to cpu
    # make agent
    agent = Agent(env.obs_space, env.act_space, embodied.Counter(), config)
    ckpt = embodied.Checkpoint(ckpt)
    ckpt.agent = agent
    ckpt.load(keys=['agent'])
    return agent


def rotation_option(conf, env):
    # env = EgocentricMazeSchool(name='ant_rotatecw')
    agent = load_agent(env, path=conf['path'])
    
    initiation = lambda *args: 1.
    # termination condition
    def termination(obs, tol=0.01):
        target_angle = np.mod(orientation(get_pose(env)[-1]) + conf['rotation'], 2 * np.pi)
        def _termination(obs): # geometric distribution
            angle = np.mod(orientation(get_pose(env)[-1]), 2*np.pi)
            return angle_error(angle, target_angle) < 0.1
        return _termination

    def policy_factory(init_obs):
        # target_angle = np.mod(compute_orientation(init_obs) + 0.25, 2*np.pi) - np.pi
        def _policy(obs, state):
            # obs['target_angle'] = target_angle
            obs = {k: convert(v)[None] for k, v in obs.items()}
            # print(obs)
            action, state = agent.policy(obs, state,  mode='eval')
            return {**action, 'reset': False}, state
        return _policy        

    return initiation, policy_factory, termination

def walking_option(conf, env):

    agent = load_agent(env, path=conf['path'])
    initiation = lambda *args: 1.
    # termination condition
    def termination(obs, distance=0.5, tol=0.2):
        pos, quat = get_pose(env)
        direction = orientation(quat) 
        target_pos = pos[:2] + distance * np.array([np.cos(direction), np.sin(direction)])
        def _termination(obs): # geometric distribution
            pos, quat = get_pose(env)
            return ((pos[:2] - target_pos) ** 2).sum() < tol ** 2
        return _termination

    def policy_factory(init_obs):
        def _policy(obs, state):
            obs = {k: convert(v)[None] for k, v in obs.items()}
            action, state = agent.policy(obs, state,  mode='eval')
            return {**action, 'reset': np.array([0.])}, state
        return _policy        

    return initiation, policy_factory, termination

def walking_distance_option(conf, env):

    agent = load_agent(env, path=conf['path'])
    initiation = lambda *args: 1.
    # termination condition
    def termination(obs, prob_termination=1/1000, tol=0.01):
        def _termination(obs): # geometric distribution
            return np.random.rand() < prob_termination
        return _termination

    def policy_factory(init_obs):
        def _policy(obs, state):
            obs = {k: convert(v)[None] for k, v in obs.items()}
            action, state = agent.policy(obs, state,  mode='eval')
            return {**action, 'reset': np.array([0.])}, state
        return _policy        

    return initiation, policy_factory, termination


def make_options(env, max_exec_time=200):
    options = {}
    for option_name, conf in OPTIONS_INFO.items():
        if conf['type'] == 'rotate':
            option = rotation_option(conf, env)
        elif conf['type'] == 'walk':
            option = walking_option(conf, env)
        else:
            raise ValueError(f'Option type {conf["type"]} is not implemented!')
        options[option_name] = Option(*option, name=option_name, recurrent=True, max_executing_time=max_exec_time, check_can_execute=False)
    return options


def test_option(env, option, trials=20, max_exec_time=500, sequential=False):
    import director.embodied.envs.loconav as nav

    def execute_option(obs, env, option):
        option.execute(obs)
        pose = get_pose(env)
        data = dict(pose=[pose], obs=[obs])
        t = 0
        while option.is_executing() and t < max_exec_time:
            action = option.act(obs)
            if action is None:
                break
            obs = env.step(action)
            pose = get_pose(env)
            data['obs'].append(obs)
            data['pose'].append(pose)
            t += 1
        print(f'Option executed for {t} steps')
        return data

    # env = nav.LocoNav('ant_empty', camera=2)
    # env = EgocentricMazeSchool(name='ant_walkforward')
    # env = EgocentricMazeSchool(name='ant_rotatecw')
    action = env.act_space['action'].sample()
    data = []
    obs = env.step(dict(action=action, reset=True))
    for i in tqdm(range(trials)):
        # reset
        if not sequential:
            obs = env.step(dict(action=action, reset=True))
        data.append(execute_option(obs, env, option))
        obs = data[-1]['obs'][-1]

    return data


def make_lists_equal(lst):
    max_length = max(len(sublist) for sublist in lst)  #Find the maximum length among all sublists
    # Extend each sublist to match the maximum length
    for sublist in lst:
        sublist.extend([sublist[-1]] * (max_length - len(sublist)))
    return lst

def get_element(key, data):
        elements = []
        for trial in data:
            elements.append(list(map(lambda x: x[key], trial['obs'])))
        elements = np.array(make_lists_equal(elements))
        return elements


def test_walking_error():
    from jax.tree_util import tree_map
    import matplotlib.pyplot as plt
    options = make_options()
    # import ipdb; ipdb.set_trace()
    data = test_option(options['walk'])
    set_vel = np.array([1., 0., 0.])

    # get vels from data.
    errors = []
    vels = []
    for trial in data:
        vels.append(list(map(lambda x: x['walker/sensors_velocimeter'], trial['obs'])))

    errors = tree_map(lambda x: (x - set_vel) ** 2, vels)
    lens = np.array(list(map(len, vels)))

    def make_lists_equal(lst):
        max_length = max(len(sublist) for sublist in lst)  #Find the maximum length among all sublists
        # Extend each sublist to match the maximum length
        for sublist in lst:
            sublist.extend([sublist[-1]] * (max_length - len(sublist)))
        return lst
    errors = np.array(make_lists_equal(errors))
    # stats

    mean_errors = errors.mean(0) # time
    std_error = errors.std(0)
    print(f'Error {mean_errors[-1]} +- {std_error[-1]}')
    t = np.arange(mean_errors.shape[0])
    
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.plot(t, mean_errors[...,i])
        plt.fill_between(t, mean_errors[...,i]-std_error[...,i], mean_errors[...,i]+std_error[...,i], alpha=0.1)
    plt.savefig('vel_error.png')

    poses = []
    # plot positions
    for trial in data:
        poses.append(list(map(lambda x: x[0], trial['pose'])))
    poses = list(map(np.array, poses))
    plt.figure()
    for p in poses:
        plt.scatter(p[:, 0], p[:, 1], s=1)
    plt.savefig('position.png')

    import imageio
    imgs = []
    for trial in data:
        imgs.append(list(map(lambda x: x['image'], trial['obs'])))
    imgs = list(map(np.array, imgs))
    with imageio.get_writer('outgif.gif', mode='I') as writer:
        for img in list(np.concatenate(imgs, axis=0)):
            writer.append_data(img)
 

def test_walking_distance_error():
    from jax.tree_util import tree_map
    import matplotlib.pyplot as plt
    env = EgocentricMazeSchool(name='ant_rotatecw')


    options = make_options(env)
    # import ipdb; ipdb.set_trace()
    data = test_option(env, options['walk'], trials=5, sequential=True)
    # stats

    poses = list(map(lambda lst: list(map(lambda x: x, lst['pose'])), data)) # List[List[(Position, Quat)]]
    positions = np.array(make_lists_equal([[p[0][:2] for p in trial] for trial in poses]))
    
    target_positions = []
    direction = lambda angle: np.array([np.cos(angle), np.sin(angle)])
    for t in poses:
        pos, quat = t[0] # initial pos
        targets = pos[:2] + direction(orientation(quat))
        target_positions.append(targets)
    target_positions = np.array(target_positions)
    # target_positions = np.array(make_lists_equal(target_positions))


    errors = np.sqrt(((target_positions[:, None] -  positions) ** 2))

    mean_errors = errors.mean(0) # time
    std_error = errors.std(0)
    print(f'Error {mean_errors[-1]} +- {std_error[-1]}')
    t = np.arange(mean_errors.shape[0])
    
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.plot(t, mean_errors[...,i])
        plt.fill_between(t, mean_errors[...,i]-std_error[...,i], mean_errors[...,i]+std_error[...,i], alpha=0.1)
    plt.savefig('position_error.png')

    vels = get_element('walker/sensors_velocimeter', data)
    mean_vels = vels.mean(0)
    std_vels = vels.std(0)
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.plot(t, mean_vels[...,i])
        plt.fill_between(t, mean_vels[...,i]-std_vels[...,i], mean_vels[...,i]+std_vels[...,i], alpha=0.1)
    plt.savefig('vels.png')

    plt.figure()
    for p in list(positions):
        plt.scatter(p[:, 0], p[:, 1], s=1)
    plt.savefig('position.png')

    import imageio
    imgs = []
    for trial in data:
        imgs.append(list(map(lambda x: x['image'], trial['obs'])))
    imgs = list(map(np.array, imgs))
    with imageio.get_writer('outgif.gif', mode='I') as writer:
        for img in list(np.concatenate(imgs, axis=0)):
            writer.append_data(img)


def test_rotation_error(direction='rotateccw'):

    from jax.tree_util import tree_map
    import matplotlib.pyplot as plt

    def make_lists_equal(lst):
        max_length = max(len(sublist) for sublist in lst)  #Find the maximum length among all sublists
        # Extend each sublist to match the maximum length
        for sublist in lst:
            sublist.extend([sublist[-1]] * (max_length - len(sublist)))
        return lst
    
    env = EgocentricMazeSchool(name='ant_rotatecw')
    directions = ['rotateccw', 'rotatecw']
    options = make_options(env)

    trajs = {}
    for direction in directions:
        trajs[direction] = test_option(env, options[direction], trials=5, sequential=True)

    # plot positions
    plt.figure()
    for i, (direction, data) in enumerate(trajs.items()):
        poses = []
        target_angles = []
        for trial in data:
            poses.append(list(map(lambda x: orientation(x[1]), trial['pose'])))
        poses = np.mod(np.array(make_lists_equal(poses)), 2*np.pi)
        delta_angle = 0.39 if direction == 'rotatecw' else -0.39
        target_angles = np.mod(poses[:, 0] + delta_angle, 2*np.pi)[:, None]

        error = angle_error(poses, target_angles)
        mean_error = error.mean(0)
        std_error = error.std(0)
        t = np.arange(mean_error.shape[0])
        plt.subplot(1, len(directions), i+1)
        plt.plot(t, mean_error)
        plt.fill_between(t, mean_error-std_error, mean_error+std_error, alpha=0.1)
        plt.title(direction)
        plt.savefig('angle_error.png')


    plt.figure()
    for i, (direction, data) in enumerate(trajs.items()):
        poses = []
        # plot positions
        for trial in data:
            poses.append(list(map(lambda x: x[0], trial['pose'])))
        poses = list(map(np.array, poses))
        plt.subplot(1, len(directions), i+1)
        for p in poses:
            plt.scatter(p[:, 0], p[:, 1], s=1)
            plt.scatter(p[:, 0], p[:, 1])
        plt.title(direction)
        plt.savefig('position_rotation.png')

    plt.figure()
    for i, (direction, data) in enumerate(trajs.items()):
        poses = []
        # plot positions
        for trial in data:
            poses.append(list(map(lambda x: orientation(x[1]), trial['pose'])))
        poses = list(map(np.array, poses))

        plt.subplot(1, len(directions), i+1)
        for p in poses:
            plt.scatter(np.cos(p), np.sin(p), s=1)
        plt.title(direction)
        plt.savefig('angles.png')

    plt.figure()
    for i, (direction, data) in enumerate(trajs.items()):
        vels = get_element('walker/sensors_gyro', data)
        mean_vels = vels.mean(0)
        std_vels = vels.std(0)
        t = np.arange(mean_vels.shape[0])
        for j in range(3):
            plt.subplot(len(directions),3,i*3 + j + 1)
            plt.plot(t, mean_vels[...,j])
            plt.fill_between(t, mean_vels[...,j]-std_vels[...,j], mean_vels[...,j]+std_vels[...,j], alpha=0.1)
            plt.title(direction + f'_{j}')
        plt.savefig('angular_vels.png')


# termination conditions
# walking termination condition (geometric? time?)
# rotation termination condition (reached target angle within certain tolerance)

# initiation sets. everywhere?

# tests
# load egocentric maze
# check rotations by executing many times and checking positions, plot error over time?

# run random policy over options.

if __name__ == '__main__':
    # test_rotation_error()

    test_walking_distance_error()