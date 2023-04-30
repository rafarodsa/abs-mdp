import gym 


class EnvGoalWrapper(gym.Env):
    def __init__(self, env, goal_fn, goal_reward=1000):
        self.env = env
        self.goal_fn = goal_fn
        self.goal_reward = goal_reward
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    
    def step(self, action):
        next_s, r, done, truncated, info = self.env.step(action)
        if self.goal_fn(next_s):
            r = self.goal_reward
            done = True
        return next_s, r, done, info

    def reset(self):
        return self.env.reset()
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


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
        


