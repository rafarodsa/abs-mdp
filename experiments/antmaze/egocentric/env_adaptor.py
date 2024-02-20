import gym


class EmbodiedEnv(gym.Env):

    mapping = dict(done='is_terminal', reward='reward', is_last='is_last')
    infos = ('log_global_pos', 'log_global_orientation', 'log_topview')

    def __init__(self, env, gamma=0.995, action_key='action', ignore_obs_keys=[]):
        self.env = env
        self.action_space = self._convert_space(env.act_space[action_key])
        self.action_key = action_key
        self.ignore_obs_keys = ignore_obs_keys
        self.observation_space = gym.spaces.Dict({k:self._convert_space(v) for k,v in env.obs_space.items() if k != 'reward' and not k in self.ignore_obs_keys})
        self.gamma = gamma
        self.past_info = None
    
    def step(self, action):
        if isinstance(action, dict):
            assert 'reset' in action.keys() and self.action_key in action.keys()
        else:
            action = {self.action_key: action, 'reset': False}
        obs = self.env.step(action) 
        done = obs[self.mapping['done']] or obs['is_last']
        reward = obs[self.mapping['reward']]
        

        next_info = {f'{k}': v  for k, v in obs.items() if k != 'reward' and k in  self.infos  and not k in self.ignore_obs_keys}
        info = {}
        info.update({f'current/{k}': v  for k, v in self.past_info.items()})
        info.update({f'next/{k}': v  for k, v in next_info.items()})
        self.past_info = next_info
        
        obs = {k: v  for k, v in obs.items() if k != 'reward' and k not in self.infos  and not k in self.ignore_obs_keys}
        return obs, reward, done, False, info

    def reset(self, state=None):
        obs = self.env.step({self.action_key: None, 'reset': True}) 
        obs = {k: v  for k, v in obs.items() if k != 'reward' and not k in self.ignore_obs_keys  and k not in self.infos}
        self.past_info = {k: v  for k, v in obs.items() if k != 'reward' and k in  self.infos  and not k in self.ignore_obs_keys}
        return obs
    
    def render(self, mode='human'):
        return self.env.render()

    def _convert_space(self, space):
        if space._discrete and len(space._shape) <= 1:
            return gym.spaces.Discrete(space._high)
        return gym.spaces.Box(space._low, space._high, space._shape)

class GroundTruthEnvWrapper(EmbodiedEnv):

    def step(self, action):
        if isinstance(action, dict):
            assert 'reset' in action.keys() and self.action_key in action.keys()
        else:
            action = {self.action_key: action, 'reset': False}
        obs = self.env.step(action) 
        done = obs[self.mapping['done']] or obs['is_last']
        reward = obs[self.mapping['reward']]
        

        next_info = {f'{k}': v  for k, v in obs.items() if k != 'reward' and k in  self.infos  and not k in self.ignore_obs_keys}
        info = {}
        info.update({f'current/{k}': v  for k, v in self.past_info.items()})
        info.update({f'next/{k}': v  for k, v in next_info.items()})
        self.past_info = next_info
        
        obs = {k: v  for k, v in obs.items() if k != 'reward' and not k in self.ignore_obs_keys}
        return obs, reward, done, False, info

    def reset(self, state=None):
        obs = self.env.step({self.action_key: None, 'reset': True}) 
        obs = {k: v  for k, v in obs.items() if k != 'reward' and not k in self.ignore_obs_keys}
        self.past_info = {k: v  for k, v in obs.items() if k != 'reward' and k in  self.infos  and not k in self.ignore_obs_keys}
        return obs