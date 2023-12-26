
import numpy as np
from dm_control import composer
import dm_control.rl
from dm_control.locomotion.arenas import Floor
from dm_control.utils import rewards
from dm_control.composer.variation import MJCFVariator
from dm_control.composer.variation import rotations

import director.embodied
from director.embodied.envs.loconav import LocoNav
import director.embodied.envs.dmc as dmc
import os
import functools
from dm_control.mujoco.wrapper.mjbindings import mjlib
from scipy.spatial.transform import Rotation as rot


class Walk(composer.Task):
    def __init__(self, walker, desired_vel=1., desired_distance=1, freq=50):
        self._walker = walker
        self._desired_displacement = desired_distance
        self._arena = Floor()
        self._arena.add_free_entity(self._walker)
        self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))
        self._prev_vel = 0.
        self._total_displacement = 0.
        self._dt = 1./freq
        self._target_velocity = desired_vel
        self._task_observables = {}
        for observable in (self._walker.observables.proprioception +
                       self._walker.observables.kinematic_sensors +
                       self._walker.observables.dynamic_sensors):
            observable.enabled = True   
        self._walker.observables.egocentric_camera.enabled = True
        self._mjcf_variator = MJCFVariator()
    
    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def get_reward(self, physics):
        vel = self._walker.observables.sensors_velocimeter(physics)
        forward_velocity = vel[0]
        displacement = 0.
        
        forward_vel_penalty = -0.1 * (self._target_velocity - forward_velocity) ** 2

        tilt_penalty = -0.05 * abs(self._walker.observables.gyro_backward_roll(physics))
        rotation_penalty = -0.05 * abs(self._walker.observables.gyro_anticlockwise_spin(physics))
        lateral_vel_penalty = -0.05 * (abs(vel[1]) + abs(vel[2]))

        displacement = -0.1 * (displacement - self._desired_displacement) ** 2
        self._prev_vel = forward_velocity

        return  0. * displacement + tilt_penalty + lateral_vel_penalty + forward_vel_penalty + rotation_penalty

    def initialize_episode(self, physics, random_state):
        # import ipdb; ipdb.set_trace()
        self._walker.initialize_episode(physics, random_state)
        vel_dir =  random_state.uniform(0., 1., (2,))
        vel_dir = vel_dir / np.linalg.norm(vel_dir)
        physics.named.data.qvel[:2] = random_state.uniform(0, 2) * vel_dir

        # import ipdb; ipdb.set_trace()
        n_joints = 4 ## TODO this is only for ant
        offset = 7
        for i in range(n_joints):
            leg_angle = np.array([0., random_state.uniform(0., 0.3490)])
            physics.named.data.qpos[i*2+offset:i*2+offset+2] = leg_angle
        
        self._prev_vel = 0.
        self._total_displacement = 0.

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

class Rotate(composer.Task):
    def __init__(self, walker, desired_angle=10, freq=50):
        self._walker = walker
        self._arena = Floor()
        self._arena.add_free_entity(self._walker)
        self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))
        self._prev_vel = 0.
        self._total_displacement = 0.
        self._dt = 1./freq
        self._target_velocity = 0.
        self._desired_angle = desired_angle
        self._task_observables = {}
        for observable in (self._walker.observables.proprioception +
                       self._walker.observables.kinematic_sensors +
                       self._walker.observables.dynamic_sensors):
            observable.enabled = True   
        self._walker.observables.egocentric_camera.enabled = True
        self._mjcf_variator = MJCFVariator()
        self._init_angle = 0.
        self._episode_started = False
    
    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def get_reward(self, physics):
        R = self._walker.observables.orientation(physics)
        R = R.reshape(3,3)
        yaw = np.arctan2(R[1, 0], R[0, 0])

        set_point = -0.1 * (yaw - (np.mod(self._init_angle + self._desired_angle, 2*np.pi)-np.pi)) ** 2
        vel = self._walker.observables.sensors_velocimeter(physics)
        vel_penalty = -0.01 * (vel ** 2).sum()
        rot_direction = -0.1 * (np.sign(self._walker.observables.gyro_anticlockwise_spin(physics)) - np.sign(self._desired_angle)) ** 2

        return float(vel_penalty + set_point + rot_direction)

    def initialize_episode(self, physics, random_state):
        # import ipdb; ipdb.set_trace()
        self._walker.initialize_episode(physics, random_state)
        # vel_dir =  random_state.uniform(0., 1., (2,))
        # vel_dir = vel_dir / np.linalg.norm(vel_dir)
        # physics.named.data.qvel[:2] = random_state.uniform(0, 2) * vel_dir
        n_joints = 4 ## TODO this is only for ant
        offset = 7

        u3 = random_state.uniform(-np.pi, np.pi)
        quat = np.array([np.cos(u3/2),
                         0.,
                         0.,
                         np.sin(u3/2)])

        physics.named.data.qpos[3:7] = quat


        for i in range(n_joints):
            leg_angle = np.array([0., random_state.uniform(0., 0.3490)])
            physics.named.data.qpos[i*2+offset:i*2+offset+2] = leg_angle
    
        self._prev_vel = 0.
        self._total_displacement = 0.
        self._episode_started = False
        self._init_angle = u3

    def get_metrics(self):
        target_angle = np.mod(self._init_angle + self._desired_angle, 2*np.pi)-np.pi
        return dict(target_angle=target_angle)

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)


class EgocentricMazeSchool(LocoNav):
    def __init__(
      self, name, task='walkforward', repeat=1, size=(64, 64), camera=3, again=False,
      termination=False, weaker=1.0):
        if name.endswith('hz'):
            name, freq = name.rsplit('_', 1)
            freq = int(freq.strip('hz'))
        else:
            freq = 50
        if freq != 50:
            print(f'Using non-standard control frequency {freq} Hz.')
        if 'MUJOCO_GL' not in os.environ:
            os.environ['MUJOCO_GL'] = 'osmesa'
        from dm_control import composer
        from dm_control.locomotion.props import target_sphere
        from dm_control.locomotion.tasks import random_goal_maze
        walker, task = name.split('_', 1)
        if camera == -1:
            camera = self.DEFAULT_CAMERAS.get(walker, 0)
        self._walker = self._make_walker(walker)
        self._task = task
        if task == 'walkforward':
            task = Walk(walker=self._walker, freq=freq)
        elif task == 'rotatecw':
            task = Rotate(walker=self._walker, desired_angle=0.25, freq=freq)
        elif task == 'rotateccw':
            task = Rotate(walker=self._walker, desired_angle=-0.25, freq=freq)
        else:
            raise ValueError(f'{task} not implemented')
        self.task = task
        env = composer.Environment(
            time_limit=60, task=task, random_state=None,
            strip_singleton_obs_buffer_dim=True)
        self._env = dmc.DMC(env, repeat, size, camera)
        self._visited = None
        self._weaker = weaker

    def step(self, action):
        obs = super().step(action)
        obs.update(**self.get_ant_metrics())
        return obs
    
    def get_ant_metrics(self):
        physics = self._env._env._physics
        vel = np.array(self._walker.observables.sensors_velocimeter(physics))
        R = np.array(self._walker.observables.orientation(physics))
        R = R.reshape(3,3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        yaw_vel = np.array(self._walker.observables.gyro_anticlockwise_spin(physics))
        obs = dict(log_vel=vel, log_yaw=yaw, log_yaw_vel=yaw_vel)
        obs.update(**self.task.get_metrics())
        # from jax.tree_util import tree_map
        # print(tree_map(lambda x: type(x), obs))
        return obs
    
    
    @property
    def obs_space(self):
        if self._task == 'walkforward':
            return super().obs_space
        return {
            **super().obs_space,
            'target_angle': director.embodied.Space(np.float32, low=-np.pi, high=np.pi),
        }


if __name__=='__main__':
    from dm_control.locomotion.walkers.ant import Ant
    ant = Ant()
    task = Rotate(walker=ant)
    env = composer.Environment(task)
    env.reset()
    env.step(env.action_spec().maximum)
    import ipdb; ipdb.set_trace()
