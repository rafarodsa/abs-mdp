import gym 
import numpy as np


class EnvOptionWrapper(gym.Wrapper):
    def __init__(self, options, env, discounted=False):
        self.env = env
        self.options = options
        self.action_space = gym.spaces.Discrete(len(options))
        self.observation_space = env.observation_space
        self._last_initset = None
        self.discounted = discounted
        # self.state = None
    
    def step(self, action):
        option = self.options[action]
        next_s, r, done, truncated, info = self._execute_option(option)
        next_initset = self.initset(next_s)
        done = done or next_initset.sum() == 0
        info['initset_next_s'] = next_initset
        info['initset_s'] = self.last_initset
        info['success'] = not truncated
        self._last_initset = next_initset
        self.state = next_s
        return next_s.astype(np.float32), r, done, truncated, info
    
    @property
    def last_initset(self):
        return self._last_initset

    def initset(self, state):
        initiation = np.array([int(o.initiation(state)) for o in self.options])
        return initiation

    def _execute_option(self, option):
        s = np.array(self.state)
        execute = option.execute(s)
        execute = True
        done = False
        t = 0
        r = 0
        info = {}
        if execute: 
            next_s = s
            while option.is_executing() and not done and t < option.max_executing_time:
                action = option.act(next_s)
                if action is None:
                    break
                next_s, _r, done, truncated, info = self.env.step(action)
                r = r + self.env.gamma ** t * _r  if self.discounted else r + _r# accumulate discounted reward
                t += 1
            if t >= option.max_executing_time and not done:
                truncated = True
            else:
                truncated = False
        else:
            next_s = self.env.state
            truncated = False

        info['tau'] = t  # add execution length to info dict
        return next_s, r, done, truncated, info

    def reset(self, state=None):
        s = self.env.reset(state)
        initset_s = self.initset(s)
        while initset_s.sum() == 0:
            s = self.env.reset().astype(np.float32)
            initset_s = self.initset(s)
        self._last_initset = initset_s
        self.state = s
        return s
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

class EnvInitsetWrapper(gym.Env):
    def __init__(self, env, initset_fn):
        self.env = env
        self.initset_fn = initset_fn
        self.last_initset = None

    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def initset(self, obs):
        return self.initset_fn(obs)
    
    def step(self, action):
        ret  = self.env.step(action)
        if len(ret) == 4:
            next_obs, r, done, info = ret
        elif len(ret) == 5:
            next_obs, r, done, truncated, info = ret
        initset = self.initset(next_obs)
        no_action_avail = initset.sum() == 0
        done = no_action_avail or done
        


        info['initset_next_s'] = initset
        info['initset_s'] = self.last_initset
        self.last_initset = initset
        return next_obs, r, done, info
    
    def reset(self, state=None):
        obs = self.env.reset(state)
        initset_next_s = self.initset(obs)
        while initset_next_s.sum() == 0:
            obs = self.env.reset()
            initset_next_s = self.initset(obs)
        self.last_initset = initset_next_s
        return obs
    
        
if __name__=='__main__':
    from envs.pinball.pinball_gym import PinballEnvContinuous
    from envs.pinball.controllers_pinball import create_position_options, PinballGridOptions

    env = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg', render_mode='human')
    options = PinballGridOptions(env, tol=1/20/5)
    env = EnvOptionWrapper(options, env)
    for j in range(5):
        s = env.reset()
        next_s = s
        for i in range(50):
            # random action
            initiation = np.array([int(o.initiation(next_s)) for o in options])
            actions_avail = np.nonzero(initiation == 1)[0]
            action = np.random.choice(actions_avail)
            next_s, r, done, truncated, info = env.step(action)
            print(s, next_s, r, action, info, initiation)
            env.render()
        


