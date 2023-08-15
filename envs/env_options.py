
import gym 
import numpy as np

class EnvOptionWrapper(gym.Env):
    def __init__(self, options, env):
        self.env = env
        self.options = options
        self.action_space = gym.spaces.Discrete(len(options))
        self.observation_space = env.observation_space
    
    def step(self, action):
        option = self.options[action]
        next_s, r, done, truncated, info = self._execute_option(option)
        next_initset = self.action_mask(next_s)
        done = done or next_initset.sum() == 0
        info['next_initset'] = next_initset
        return next_s.astype(np.float32), r, done, truncated, info
    

    def action_mask(self, state):
        initiation = np.array([int(o.initiation(state)) for o in self.options])
        return initiation

    def _execute_option(self, option):
        s = np.array(self.env.state)
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
                r += self.env.gamma ** t * _r # accumulate discounted reward
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
        return self.env.reset(state).astype(np.float32)
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

        
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
        


