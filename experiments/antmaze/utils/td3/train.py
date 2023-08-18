import gym
import json
import pfrl
import time
import torch
import pickle
import argparse
import numpy as np
import d4rl

from hrl.utils import create_log_dir
from hrl.agent.td3.utils import save
from hrl.agent.td3.TD3AgentClass import TD3
from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
# from hrl.wrappers.environments.ant_maze_env import AntMazeEnv
from hrl.agent.td3.utils import make_chunked_value_function_plot 


def make_env(name, start, goal, dense_reward, seed, horizon=1000):
  print(gym.envs.registry.all())

  if "reacher" not in name.lower():
    env = gym.make(name)
    print('gym environment made')
  else:
    gym_mujoco_kwargs = {
      'maze_id': 'Reacher',
      'n_bins': 0,
      'observe_blocks': False,
      'put_spin_near_agent': False,
      'top_down_view': False,
      'manual_collision': True,
      'maze_size_scaling': 3,
      'color_str': ""
    }
    # env = AntMazeEnv(**gym_mujoco_kwargs)

  goal_reward = 0. if dense_reward else 1.
  print(env)
  env = D4RLAntMazeWrapper(env,
              start_state=start,
              goal_state=goal,
              use_dense_reward=dense_reward
            )
  print('D4RL environment made')
              # goal_reward=goal_reward,
              # step_reward=0.)

  env = pfrl.wrappers.CastObservationToFloat32(env)
  env = pfrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=horizon)
  env.env.seed(seed)
  return env

def sample_goal(env, s0, min_distance=2.) -> np.ndarray:
  """Output the deltax and deltay."""
  accepted = False
  s0 = s0[:2]
  # s = np.array([8., 0.])
  while not accepted:
    s = env.sample_random_state()
    # s = np.array([0., 0.])
    accepted = np.linalg.norm(s0-s) >= min_distance
  return s 


def sample_goal_ball(env, s0, min_distance=1e-3, radius=0.1) -> np.ndarray:
  '''
      Sample goal in a ball of radius.
  '''
  accepted = False
  s0 = s0[:2]
  # s = np.array([8., 0.])
  print(f'S0: {s0}')
  while not accepted:
    s = env.sample_random_state()
    # s = np.array([0., 0.])
    distance = np.linalg.norm(s0-s)
    accepted = distance >= min_distance and distance <= radius
  return s 


def run_episode(agent, env, state, current_episode, num_steps=1000):
  done = False

  # goal = sample_goal(env, state)
  goal = sample_goal_ball(env, state, min_distance=1, radius=1.5)
  env.set_goal(goal)
  print(f'Episode: {current_episode} s0: {env.init_state[:2]}, Goal: {goal}, Distance {np.linalg.norm(goal-env.init_state[:2])}')
  
  total_reward = 0.
  trajectory = []

  for i in range(num_steps):
    
    augmented_state = np.concatenate((state, goal), axis=0)
    action = agent.act(augmented_state)
    
    next_state, reward, done, info = env.step(action)
    augmented_next_state = np.concatenate((next_state, goal), axis=0)
    transition = (augmented_state, action, reward, augmented_next_state, done)
    
    agent.step(*transition)

    trajectory.append(transition)
    state = next_state
    total_reward += reward

    if done:
      break

  print(f'Episode: {current_episode} Reward: {total_reward} Steps: {i} Final State {state[:2]} Distance to Goal {np.linalg.norm(goal-state[:2])}')
  return trajectory, state, done, total_reward, i


def train(agent, env, num_episodes, num_steps_per_episode, log_file, args):
  _log_steps = []
  _log_rewards = []
  for current_episode in range(num_episodes):
    s0 = env.reset()

    trajectory, state, reached, episode_reward, episode_length = run_episode(
      agent, env, s0, current_episode, num_steps=num_steps_per_episode)
    
    if not reached:
      hindsight_goal = state[:2]
      her(agent, env, trajectory, hindsight_goal, args.use_dense_rewards)

    # log 
    _log_steps.append(episode_length)
    _log_rewards.append(episode_reward)

    with open(log_file, "wb+") as f:
      episode_metrics = {
              "step": _log_steps, 
              "reward": _log_rewards,
      }
      pickle.dump(episode_metrics, f)
          
    if args.save_replay_buffer and current_episode % 100 == 0:
      agent.replay_buffer.save(_buffer_log_file)

    if args.save_agent and current_episode % 500 == 0:
      save(agent, f"saved_models/{args.experiment_name}/{args.seed}/td3_episode_{current_episode}")


def her(agent, env, trajectory, new_goal, use_dense_rewards=False):
  assert new_goal.shape == (2, )
  for state, action, _, next_state, _ in trajectory:
    augmented_state = state.copy()
    augmented_state[-2:] = new_goal
    augmented_next_state = next_state.copy()
    augmented_next_state[-2:] = new_goal
    reward, done = env.sparse_gc_reward_func(next_state, new_goal) if not use_dense_rewards else env.dense_gc_reward_func(next_state, new_goal)
    agent.step(
      augmented_state, action, reward, augmented_next_state, done
    )
    if done: 
      break
  

if __name__ == "__main__":
  print('Training script TD3...')
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int)
  parser.add_argument("--gpu_id", type=int)
  parser.add_argument("--experiment_name", type=str)
  parser.add_argument("--environment_name", type=str)
  parser.add_argument("--num_training_episodes", type=int, default=1000)
  parser.add_argument("--use_random_starts", action="store_true", default=False)
  parser.add_argument("--plot_value_function", action="store_true", default=False)
  parser.add_argument("--use_dense_rewards", action="store_true", default=False)
  parser.add_argument("--save_replay_buffer", action="store_true", default=False)
  parser.add_argument("--save_agent", action="store_true", default=False)
  parser.add_argument("--lr", type=float, default=3e-4)
  args = parser.parse_args()

  print('load args')

  create_log_dir("logs")
  create_log_dir(f"logs/{args.experiment_name}")
  create_log_dir(f"logs/{args.experiment_name}/{args.seed}")
  
  create_log_dir("plots")
  create_log_dir(f"plots/{args.experiment_name}")
  create_log_dir(f"plots/{args.experiment_name}/{args.seed}")

  create_log_dir("saved_modes")
  create_log_dir(f"saved_models/{args.experiment_name}")
  create_log_dir(f"saved_models/{args.experiment_name}/{args.seed}")

  with open(f"logs/{args.experiment_name}/{args.seed}/hyperparameters.txt", "w+") as _args_file:
    json.dump(args.__dict__, _args_file, indent=2)

  _log_file = f"logs/{args.experiment_name}/{args.seed}/td3_log.pkl"
  _buffer_log_file = f"logs/{args.experiment_name}/{args.seed}/td3_replay_buffer.pkl"

  print('making env...')
  env = make_env(args.environment_name,
           start=np.array([8., 0.]),
           goal=np.array([0., 0.]),
           seed=args.seed,
           dense_reward=args.use_dense_rewards)
  print('environment made')
  print(f'{env}')
  # pfrl.utils.set_random_seed(args.seed)

  obs_size = env.observation_space.shape[0]
  goal_size = 2
  action_size = env.action_space.shape[0]
  print('initiating agent')
  agent = TD3(obs_size + goal_size,
        action_size,
        max_action=1.,
        use_output_normalization=False,
        device=torch.device(
          f"cuda:{args.gpu_id}" if args.gpu_id > -1 else "cpu"
        ),
        store_extra_info=args.save_replay_buffer,
        lr_c=args.lr, lr_a=args.lr,
  )

  t0 = time.time()
  steps_per_episode = 1000
  print(t0)
  train(agent, env, args.num_training_episodes, steps_per_episode, _log_file, args)
  print(f"Finished after {(time.time() - t0) / 3600.} hrs")