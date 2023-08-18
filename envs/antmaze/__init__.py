
import gym
from antmaze_wrapper import D4RLAntMazeWrapper
import pfrl

def make_env(name, start, goal, seed, horizon=1000):

  if "antmaze" in name.lower():
    env = gym.make(name)

  env = D4RLAntMazeWrapper(env,
              start_state=start,
              goal_state=goal,
              use_dense_reward=False
            )
  env = pfrl.wrappers.CastObservationToFloat32(env)
  env = pfrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=horizon)
  env.env.seed(seed)
  return env
