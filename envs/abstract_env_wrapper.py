from typing import Any
import gym 
from src.utils.printarr import printarr


class AbstractEnvWrapper(gym.Env):
    def __init__(self, env, encoder):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.encoder = encoder

    def step(self, action):
        ret = self.env.step(action)
        if len(ret) == 4:
            next_s, r, done, info = ret
        else:
            next_s, r, done, truncated, info = ret

        next_z = self.encoder(next_s)
        info['next_s'] = next_s
        return next_z, r, done, info
    
    def reset(self, state=None):
        s = self.env.reset(state)
        return self.encoder(s)
    
    def __getattr__(self, name):
        return getattr(self.env, name)