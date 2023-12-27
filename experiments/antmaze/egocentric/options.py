
import pathlib

import numpy as np
import ruamel.yaml as yaml

from dreamerv3 import embodied
from dreamerv3.agent import Agent

from envs.antmaze.ant_school import EgocentricMazeSchool
from src.options.option import Option
from tqdm import tqdm

OPTIONS_INFO = {
    'walk': {
        'path': '/users/rrodri19/abs-mdp/exp_results/antmaze/egocentric/walk_no_rotation_no_jumping',
        'type': 'walk'
    },
    'rotatecw': {
        'path': '/users/rrodri19/abs-mdp/exp_results/antmaze/egocentric/dreamerv3_antmaze_ego_rotate_1__task_ant_rotatecw',
        'type': 'rotate'
    },
    'rotateccw':
     {
        'path':'/users/rrodri19/abs-mdp/exp_results/antmaze/egocentric/dreamerv3_antmaze_ego_rotate_2__task_ant_rotateccw',
        'type': 'rotate'
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

def load_agent(env, path):
    # load config
    path = embodied.Path(path)
    ckpt =  path / 'checkpoint.ckpt'
    config = path / 'config.yaml'
    config = embodied.Config.load(pathlib.Path(str(config)))
    # make agent
    agent = Agent(env.obs_space, env.act_space, embodied.Counter(), config)
    ckpt = embodied.Checkpoint(ckpt)
    ckpt.agent = agent
    ckpt.load(keys=['agent'])
    return agent


def compute_orientation(obs):
    # compute orientation from observation vector
    pass


def rotation_option(path):
    env = EgocentricMazeSchool(name='ant_rotatecw')
    agent = load_agent(env, path=path)

    initiation = lambda *args: 1.
    # termination condition
    def termination(target_angle, tol=0.01):
        # def _termination(obs):
        #     # compute orientation
        #     angle = compute_orientation(obs)
        #     return float((angle-target_angle) ** 2 < tol ** 2)
        def _termination(obs): # geometric distribution
            return np.random.rand() < 1/75
        return _termination

    def policy_factory(init_obs):
        # target_angle = np.mod(compute_orientation(init_obs) + 0.25, 2*np.pi) - np.pi
        def _policy(obs, state):
            # obs['target_angle'] = target_angle
            obs = {k: convert(v)[None] for k, v in obs.items()}
            action, state = agent.policy(obs, state,  mode='eval')
            return {**action, 'reset': False}, state
        return _policy        

    return initiation, policy_factory, termination

def walking_option(path):
    env = EgocentricMazeSchool(name='ant_walkforward')
    agent = load_agent(env, path=path)

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


def make_options():
    options = {}
    for option_name, conf in OPTIONS_INFO.items():
        if conf['type'] == 'rotate':
            option = rotation_option(conf['path'])
        elif conf['type'] == 'walk':
            option = walking_option(conf['path'])
        else:
            raise ValueError(f'Option type {conf["type"]} is not implemented!')
        options[option_name] = Option(*option, name=option_name, recurrent=True)
    return options




def test_option(option, trials=20):
    import director.embodied.envs.loconav as nav
    def get_pose(env):
        physics = env._env._env._physics
        return list(map(np.array, env._walker.get_pose(physics)))

    def execute_option(obs, env, option):
        option.execute(obs)
        pose = get_pose(env)
        data = dict(pose=[pose], obs=[obs])
        while option.is_executing():
            action = option.act(obs)
            if action is None:
                break
            obs = env.step(action)
            pose = get_pose(env)
            data['obs'].append(obs)
            data['pose'].append(pose)
        return data

    # env = nav.LocoNav('ant_empty', camera=2)
    # env = EgocentricMazeSchool(name='ant_walkforward')
    env = EgocentricMazeSchool(name='ant_rotatecw')
    action = env.act_space['action'].sample()
    data = []
    obs = env.step(dict(action=action, reset=True))
    for i in tqdm(range(trials)):
        # reset
        obs = env.step(dict(action=action, reset=True))
        data.append(execute_option(obs, env, option))
        obs = data[-1]['obs'][-1]

    return data


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
 

def test_rotation_error():
    from jax.tree_util import tree_map
    import matplotlib.pyplot as plt
    from dm_control.utils.transformations import quat_rotate
    options = make_options()
    # import ipdb; ipdb.set_trace()
    data = test_option(options['rotatecw'])

    def get_element(key, data):
        elements = []
        for trial in data:
            elements.append(list(map(lambda x: x[key], trial['obs'])))
        elements = np.array(make_lists_equal(elements))
        return elements
    
    def make_lists_equal(lst):
        max_length = max(len(sublist) for sublist in lst)  #Find the maximum length among all sublists
        # Extend each sublist to match the maximum length
        for sublist in lst:
            sublist.extend([sublist[-1]] * (max_length - len(sublist)))
        return lst

    def orientation(quat):
         x = quat_rotate(quat, np.array([1., 0., 0.]))
         return np.arctan2(x[1], x[0])

    # plot positions
    poses = []
    target_angles = []
    for trial in data:
        poses.append(list(map(lambda x: orientation(x[1]), trial['pose'])))
        target_angles.append(list(map(lambda x: x['target_angle'][0], trial['obs'])))
    poses = np.array(make_lists_equal(poses))
    target_angles = np.array(make_lists_equal(target_angles))
    import ipdb; ipdb.set_trace()
    error = np.sqrt((poses-target_angles) ** 2)
    mean_error = np.mod(error, 2*np.pi).mean(0)
    std_error = error.std(0)
    t = np.arange(mean_error.shape[0])
    plt.figure()
    plt.plot(t, mean_error)
    plt.fill_between(t, mean_error-std_error, mean_error+std_error, alpha=0.1)
    plt.savefig('angle_error.png')

    plt.figure()
    plt.plot(t, poses[10])
    plt.savefig('rotation.png')


    poses = []
    # plot positions
    for trial in data:
        poses.append(list(map(lambda x: x[0], trial['pose'])))
    poses = list(map(np.array, poses))
    plt.figure()
    for p in poses:
        plt.scatter(p[:, 0], p[:, 1], s=1)
        plt.scatter(p[:, 0], p[:, 1])
    plt.savefig('position.png')

    plt.figure()
    # angular_vels = get_element('walker/sensors_gyro', data)[10]
    # for i in range(3):
    #     plt.subplot(1,3,i+1)
    #     plt.plot(t, angular_vels[:, i])
    # plt.savefig('angular_vel.png')

    import imageio
    imgs = []
    for trial in data:
        imgs.append(list(map(lambda x: x['image'], trial['obs'])))
    imgs = list(map(np.array, imgs))
    with imageio.get_writer('outgif.gif', mode='I') as writer:
        for img in list(np.concatenate(imgs, axis=0)):
            writer.append_data(img)


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
    test_walking_error()