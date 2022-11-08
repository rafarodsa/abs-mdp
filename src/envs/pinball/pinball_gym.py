"""
    OpenAI Gym Adapter for the Pinball Domain
    author: Rafael Rodriguez-Sanchez
    date: October 2022 
    email: rrs@brown.edu
"""
import gym
from gym import spaces
from gym.error import DependencyNotInstalled

import numpy as np
from .pinball import PinballModel, PinballView, BallModel

from collections import UserDict
from matplotlib.path import Path


class GoalPinballModel(PinballModel):
    def set_target_pos(self, target_pos):
        self.target_pos = target_pos

    def set_initial_pos(self, start_pos):
        self.ball = BallModel(start_position=start_pos, radius=0.01)



class PinballEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, config, start_pos=None, target_pos=None, width=500, height=500, render_mode=None):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=np.zeros(4), high=np.ones(4))
        self.gamma = 1
        self.configuration = config
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.screen = None  
        self.clock = None

        self.pinball = GoalPinballModel(self.configuration)
        self.state = self.pinball.get_state()
        self._obstacles = [Path(obstacle.points) for obstacle in self.pinball.obstacles]
        if start_pos:
            self.pinball.set_initial_pos(start_pos)
        if target_pos:
            self.pinball.set_target_pos(target_pos)


    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid, action_space {self.action_space}"

        reward = self.pinball.take_action(action)
        next_state = self.pinball.get_state()
        done = self.pinball.episode_ended()

        if self.render_mode == "human":
            self.render()

        return np.array(next_state), reward, done, False, {}


    def get_obstacles(self):
        """
            return list of Pinball Obstacles
        """
        return self.pinball.obstacles

    
    def sample_initial_positions(self, N):
        """
            Samples initial positions uniformly with velocity 0.
        """
        all_points = []
        total_points = 0
        vels = np.zeros((N, 2))
        while total_points < N:
            points = self._get_points_outside_obstacles(np.random.uniform(size=(N, 2)))
            all_points.append(points)
            total_points += points.shape[0]
        return np.hstack([np.vstack(all_points)[:N], vels])

    def _get_points_outside_obstacles(self, points):
        points_mask = np.ones(points.shape[0], dtype=np.bool8)
        for obstacle in self._obstacles:
            points_mask = np.logical_and(np.logical_not(obstacle.contains_points(points, radius=0.02)), points_mask)

        return points[points_mask]

    def reset(self, state=None):
        self.pinball = PinballModel(self.configuration) 
        return np.array(self.pinball.get_state())

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )
        
        if self.screen is None:
            if self.render_mode == "human":
                pygame.init()
                pygame.display.set_caption('Pinball Domain')
                self.screen = pygame.display.set_mode([self.width, self.height])
                self.environment_view = PinballView(self.screen, self.pinball)      
            else:
                self.screen = pygame.Surface((self.width, self.height))
                self.environment_view = PinballView(self.screen, self.pinball)  

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self.environment_view.blit()
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            self.environment_view.blit()
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

class DummyDict(UserDict):
    def __init__(self, default_value_f):
        self.default_value_f = default_value_f
        self.data = []
    
    def __missing__(self, key):
        return self.default_value_f(key)

class PinballModelContinuous(GoalPinballModel):
    MAX_SPEED = 1
    
    def __init__(self, configuration):
        super().__init__(configuration)
        self.action_effects = DummyDict(self.__action_effects)

    def __action_effects(self, action):
        _action = np.clip(action, -self.MAX_SPEED, self.MAX_SPEED)
        return _action[0], _action[1]


class PinballEnvContinuous(PinballEnv):
    MAX_SPEED = 1
    def __init__(self, config, start_pos=None, target_pos=None, width=500, height=500, render_mode=None):
        super().__init__(config, width, height, render_mode)
        self.action_space = spaces.Box(low=-self.MAX_SPEED, high=self.MAX_SPEED, shape=(2,), dtype=np.float64)
        self.pinball = PinballModelContinuous(config)
    
    def reset(self, state=None):
        self.pinball = PinballModelContinuous(self.configuration)
        if state is not None:
            self.pinball.set_initial_pos(state[:2]) 
        return np.array(self.pinball.get_state())

    def step(self, action):
        action = tuple(action)
        return super().step(action)

if __name__=="__main__":
    from matplotlib import pyplot as plt

    pinball = PinballEnv("/Users/rrs/Desktop/abs-mdp/src/envs/pinball/configs/pinball_no_obstacles.cfg", render_mode="human")
    for _ in range(100):
        a = np.random.randint(5)
        pinball.step(a)
        pinball.render()