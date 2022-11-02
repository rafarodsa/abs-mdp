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
from .pinball import PinballModel, PinballView

from collections import UserDict

class PinballEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, config, width=500, height=500, render_mode=None):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=np.zeros(4), high=np.ones(4))
        self.gamma = 1
        self.configuration = config
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.screen = None  
        self.clock = None

        self.pinball = PinballModel(self.configuration)
        self.state = self.pinball.get_state()
        

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid, action_space {self.action_space}"

        action = tuple(action)
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


    def reset(self):
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

class PinballModelContinuous(PinballModel):
    MAX_SPEED = 1
    
    def __init__(self, configuration):
        super().__init__(configuration)
        self.action_effects = DummyDict(self.__action_effects)

    def __action_effects(self, action):
        _action = np.clip(action, -self.MAX_SPEED, self.MAX_SPEED)
        return _action[0], _action[1]


class PinballEnvContinuous(PinballEnv):
    MAX_SPEED = 1
    def __init__(self, config, width=500, height=500, render_mode=None):
        super().__init__(config, width, height, render_mode)
        self.action_space = spaces.Box(low=-self.MAX_SPEED, high=self.MAX_SPEED, shape=(2,), dtype=np.float64)
        self.pinball = PinballModelContinuous(config)
    
    def reset(self):
        self.pinball = PinballModelContinuous(self.configuration)
        return np.array(self.pinball.get_state())



if __name__=="__main__":
    from matplotlib import pyplot as plt

    pinball = PinballEnv("/Users/rrs/Desktop/abs-mdp/src/envs/pinball/configs/pinball_no_obstacles.cfg", render_mode="human")
    for _ in range(100):
        a = np.random.randint(5)
        pinball.step(a)
        pinball.render()