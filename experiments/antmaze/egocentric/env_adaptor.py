import gym



class EmbodiedEnv(gym.Env):

    mapping = dict(done='is_terminal', reward='reward', is_first='is_first', is_last='is_last')

    def __init__(self, env, action_key='action'):
        self.env = env
        self.action_space = env.act_space[action_key]
        self.action_key = action_key
        self.observation_space = {k:v for k,v in env.obs_space.items() if k not in self.mapping.values()}
    
    def step(self, action):
        obs = self.env.step({self.action_key: action, 'reset': False})
        done = obs[self.mapping['done']]
        reward = obs[self.mapping['reward']]
        obs = {k: v  for k, v in obs.items() if k not in self.mapping.values()}
        return obs, reward, done, False, {}

    def reset(self, state=None):
        obs = self.env.step({self.action_key: None, 'reset': True})
        obs = {k: v  for k, v in obs.items() if k not in self.mapping.values()}
        return obs
    
    def render(self, mode='human'):
        return self.env.render()
    