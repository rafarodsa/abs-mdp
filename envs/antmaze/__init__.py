
import gym
from .antmaze_wrapper import D4RLAntMazeWrapper
import pfrl
import d4rl
import numpy as np

def make_env(name, seed=None, start=np.array([-10, -10]), goal=np.array([-20, -20]), horizon=1000):

  if "antmaze" in name.lower():
    env = gym.make(name)

  env = D4RLAntMazeWrapper(env,
              start_state=start,
              goal_state=goal,
              use_dense_reward=False
            )
  # env = pfrl.wrappers.CastObservationToFloat32(env)
  # env = pfrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=horizon)
  if seed is not None:
    env.env.seed(seed)
  return env
