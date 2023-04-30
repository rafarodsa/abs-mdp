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
from envs.pinball.pinball import PinballModel, PinballView, BallModel

from collections import UserDict, deque
from matplotlib.path import Path

import pygame

class GoalPinballModel(PinballModel):
    def set_target_pos(self, target_pos):
        self.target_pos = target_pos

    def set_initial_pos(self, start_pos):
        self._ball_rad = self.ball.radius
        self.ball = BallModel(start_position=start_pos, radius=self._ball_rad)
    
    def set_initial_state(self, state):
        self.ball.position[0], self.ball.position[1] = state[:2]
        self.ball.xdot, self.ball.ydot = state[2:]

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
        # self.state = self.pinball.get_state()
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
            _p = np.random.uniform(low=0.02, high=0.98, size=(N, 2))
            points = self._get_points_outside_obstacles(_p)
            all_points.append(points)
            total_points += points.shape[0]
        pos = np.vstack(all_points)[:N]
        return np.hstack([pos, vels])

    def sample_init_states(self, N):
        """
            Samples states uniformly 
        """
        all_points = []
        total_points = 0
        vels = np.random.uniform(size=(N, 2))
        while total_points < N:
            
            points = self._get_points_outside_obstacles(_p)
            all_points.append(points)
            total_points += points.shape[0]
        return np.hstack([np.vstack(all_points)[:N], vels])

    def _get_points_outside_obstacles(self, points):
        points_mask = np.ones(points.shape[0], dtype=np.bool8)
        for obstacle in self._obstacles:
            in_obstacle = obstacle.contains_points(points, radius=0.)
            points_mask = np.logical_and(np.logical_not(in_obstacle), points_mask)
        return points[points_mask]
    
    def is_valid_state(self, state):
        for obstacle in self._obstacles:
            if obstacle.contains_points([state[:2]], radius=0.):
                return False
        return True

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
                np.array(pygame.surfarray.pixels3d(self.screen))/255, axes=(1, 0, 2)
            )
    @property
    def state(self):
        return np.array(self.pinball.get_state())
    

class DummyDict(UserDict):
    def __init__(self, default_value_f):
        self.default_value_f = default_value_f
        self.data = []
    
    def __missing__(self, key):
        return self.default_value_f(key)

class PinballModelContinuous(GoalPinballModel):
    MAX_ACC = 1
    STEP_COST = -1
    ACC_COST = -5
    
    def __init__(self, configuration):
        super().__init__(configuration)
        self.action_effects = DummyDict(self._action_effects)

    def _action_effects(self, action):
        _action = np.clip(action, -self.MAX_ACC, self.MAX_ACC)
        return _action[0], _action[1]        

    def reward(self, action):
        return self.ACC_COST * np.linalg.norm(action) # acc_cost per each acceleration unit.

class PinballEnvContinuous(PinballEnv):
    MAX_SPEED = 1
    GOAL_REWARD = 10000
    def __init__(self, config, start_pos=None, target_pos=None, width=500, height=500, render_mode=None):
        super().__init__(config, start_pos, target_pos, width, height, render_mode)
        self.action_space = spaces.Box(low=-self.MAX_SPEED, high=self.MAX_SPEED, shape=(2,), dtype=np.float32)
        self.pinball = PinballModelContinuous(config)
    
    def reset(self, state=None):
        if state is None:
            state = self.sample_initial_positions(1)[0]
       
        if not self.is_valid_state(state):
            print(f"Invalid state [{state[0]}, {state[1]}]")
       
        self.pinball.set_initial_state(state)
        return state

    def step(self, action):
        action = tuple(action)
        next_s, _, done, _, _ = super().step(action)
        reward = self.pinball.reward(action) if not done else self.GOAL_REWARD
        return next_s, reward, done, False, {}

class PinballPixelWrapper(gym.Env):
    def __init__(self, environment, n_frames=1):
        self.frames = deque(maxlen=n_frames)
        self.env = environment
        self.n_frames = n_frames

    def step(self, *args, **kwargs):
        ret = self.env.step(*args, **kwargs)
        frame = self.env.render()
        if len(self.frames) < self.n_frames:
            self._init_queue(frame)
        self.frames.append(frame)

        return np.array(self.frames), *ret[1:-1], {"next_state": ret[0]}

    def _init_queue(self, frame):
        for i in range(self.n_frames):
            self.frames.append(np.zeros_like(frame))

    def reset(self, *args, **kwargs):
        self.frames.clear()
        self.env.reset(*args, **kwargs)
        frame = self.env.render()
        if len(self.frames) < self.n_frames:
            self._init_queue(frame)
        self.frames.append(frame)
        return np.array(self.frames)

    def sample_initial_positions(self, N):
        return self.env.sample_initial_positions(N)
    
    @property
    def action_space(self):
        return self.env.action_space

    def get_obstacles(self):
        return self.env.get_obstacles()



class PinballDistractors(PinballEnvContinuous):
    def __init__(self, config, start_pos=None, target_pos=None, width=500, height=500, render_mode=None, distractors=None):
        super().__init__(config, start_pos, target_pos, width, height, render_mode)
        self.distractors = distractors

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
                self.environment_view = PinballViewDistractor(self.screen, self.pinball, self.distractors)      
            else:
                self.screen = pygame.Surface((self.width, self.height))
                self.environment_view = PinballViewDistractor(self.screen, self.pinball, self.distractors)  

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
                np.array(pygame.surfarray.pixels3d(self.screen))/255, axes=(1, 0, 2)
            )


class PinballViewDistractor(PinballView):
    def __init__(self,screen, model, distractors):
        super().__init__(screen, model)
        self.distractors = distractors

    def random_background(self):
        if self.distractors is not None:
            self.background_surface = pygame.Surface(self.screen.get_size())
            d = np.random.choice(len(self.distractors))
            img = (self.distractors[d] * 255).astype(np.uint8)
            surf = pygame.surfarray.make_surface(img)
            center = np.array(self.screen.get_size()) / 2 - np.array(img.shape)[:2]/ 2
            self.background_surface.blit(surf, center)
            for obs in self.model.obstacles:
                pygame.draw.polygon(self.background_surface, self.DARK_GRAY, list(map(self._to_pixels, obs.points)), 0)

            pygame.draw.circle(
                self.background_surface, self.TARGET_COLOR, self._to_pixels(self.model.target_pos), int(self.model.target_rad*self.screen.get_width()))
        return self.background_surface
        
    def blit(self):
        self.background_surface = self.random_background()
        super().blit()


if __name__=="__main__":
    from matplotlib import pyplot as plt
    import torchvision, torch
    
    # load MNIST from torchvision
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    dataiter = [d.expand(3,-1,-1,-1).squeeze().permute(2,1,0).numpy() for d, l in iter(trainloader)]

    dataiter =None
    size = 50

    pinball = PinballDistractors(
                                    config="/Users/rrs/Desktop/abs-mdp/envs/pinball/configs/pinball_simple_single.cfg", 
                                    width=size, 
                                    height=size, 
                                    render_mode="rgb_array",
                                    distractors=dataiter
                                )
    pixel_pinball = PinballPixelWrapper(pinball, n_frames=5)
    pinball.reset()
    imgs = []
    for _  in range(10):
        for i in range(20):
            a = [0.9, 0.9]
            next_s, _, _, _, _ = pinball.step(a)
            imgs.append(pinball.render())
        plt.imshow(np.array(imgs).mean(0))
        plt.show()
        imgs = []
        pinball.reset()
        # _s = pinball.reset(next_s)
        
    
