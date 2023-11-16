
import numpy as np
import gym
from gym.spaces import Box, Dict
from dm_env import specs

def _spec_to_box(spec, dtype=np.float32):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        else:
            raise ValueError("Unrecognized type")

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return Box(low, high, dtype=dtype)

class EgocentricDMCEnv(gym.Env):
    def __init__(self, env):
        super.__init__(self)
        self._env = env
        self.action_space = _spec_to_box(self._env.action_spec())
        self.observation_space = ??

    def _get_obs(self, timestep):
        pass
        # observation {image, proprioception} 

    def step(self, action):
        timestep = self._env.step(action)
        obs = self._get_obs(timestep)
        return obs, timestep.reward, obs.last(), {'discount': timestep.discount}

    def reset(self):
        obs = self._env.reset()
        return self._get_obs(obs)

    def render(self):
        self._env.physics.render()

