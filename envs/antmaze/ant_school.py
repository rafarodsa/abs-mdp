
import numpy as np
from dm_control import composer
import dm_control.rl
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.arenas import Floor
from dm_control.utils import rewards
from dm_control.composer.variation import MJCFVariator


class Walk(composer.Task):
    def __init__(self, walker, desired_distance):
        self._walker = walker
        self._desired_displacement = desired_distance
        self._arena = Floor()
        self._arena.add_free_entity(self._walker)
        self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))
        self._prev_vel = 0.
        self._total_displacement = 0.
        self._dt = 1.
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
        displacement = (forward_velocity + self._prev_vel) * self._dt

        tilt_penalty = -0.01 * abs(self._walker.observables.gyro_backward_roll(physics))
        lateral_vel_penalty = -0.01 * abs(vel[1])

        displacement = -0.1 * (displacement - self._desired_displacement) ** 2
        self._prev_vel = forward_velocity

        return  displacement + tilt_penalty + lateral_vel_penalty

    def initialize_episode(self, physics, random_state):
        self._walker.initialize_episode(physics, random_state)
        self._prev_vel = 0.
        self._total_displacement = 0.

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)


if __name__=='__main__':
    from dm_control.locomotion.walkers.ant import Ant
    ant = Ant()
    task = Walk(walker=ant, desired_distance=0.1)
    env = composer.Environment(task)
    env.reset()
    env.step(env.action_spec().maximum)
    import ipdb; ipdb.set_trace()


    
    

