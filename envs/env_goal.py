import gym 
from src.utils.printarr import printarr

class EnvGoalWrapper(gym.Env):
    def __init__(self, env, goal_fn, goal_reward=1, gamma=0.99, reward_scale=0., init_state_sampler=None, discounted=True):
        self.env = env
        self.goal_fn = goal_fn
        self.goal_reward = goal_reward
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.init_state_sampler = init_state_sampler if init_state_sampler is not None else lambda: None
        self.discounted = discounted
    
    def step(self, action):
        ret = self.env.step(action)
        if len(ret) == 4:
            next_s, r, done, info = ret
        else:
            next_s, r, done, truncated, info = ret
        r = self.reward_scale * r
        tau = 1 if 'tau' not in info else info['tau']
        if self.goal_fn(next_s):
            r = r + self.goal_reward * (self.gamma ** (tau-1)) if self.discounted else r + self.goal_reward
            done = True
            print('======================HERE! GOAL======================')
        return next_s, r, done, info

    def reset(self, state=None):
        # print('===================resetting')
        s = self.init_state_sampler() if state is None else state
        return self.env.reset(s)
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class EnvGoalOptionsWrapper(EnvGoalWrapper):
    def step(self, action):
        next_s, r, done, truncated, info = self.env.step(action)
        if self.goal_fn(next_s):
            r += self.goal_reward * self.env.gamma ** info['execution_length']
            done = True
        return next_s, r, done, truncated, info


if __name__=='__main__':
    from envs.pinball.pinball_gym import PinballEnvContinuous
    from envs.pinball.controllers_pinball import create_position_options
    from envs.env_options import EnvOptionWrapper
    import numpy as np

    env = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg', render_mode='human')
    options = create_position_options(env)

    def goal_fn(state, goal=[0.5, 0.5]):
        return np.linalg.norm(state[:2]-goal) < 1/20 + 0.01

    env = EnvGoalOptionsWrapper(EnvOptionWrapper(options, env), goal_fn=goal_fn)
    s = env.reset()
    for i in range(100):
        # random action
        action = env.action_space.sample()
        next_s, r, done, truncated, info = env.step(action)
        print(s, next_s)
        env.render()
        


