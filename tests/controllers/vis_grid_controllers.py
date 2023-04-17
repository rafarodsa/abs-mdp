


import numpy as np
import matplotlib.pyplot as plt

from envs.pinball.pinball_gym import PinballEnvContinuous as Pinball
from envs.pinball.controllers_pinball import PinballGridOptions, compute_grid_goal

from scripts.utils import collect_trajectory, compute_initiation_masks
from functools import reduce
from src.utils.printarr import printarr
from tqdm import tqdm

import random
import pygame
N = 20
tol = 1/N/5
directions = [np.array([0, -1]), np.array([0, 1]), np.array([-1, 0]), np.array([1, 0])]


def test_grid_controllers(config_path):
    env = Pinball(config_path)

    grid_options = PinballGridOptions(env, n_pos=N, tol=tol)
    num_traj = 100
    trajectories = [collect_trajectory(env, grid_options, max_exec_time=100, horizon=100) for _ in tqdm(range(num_traj))]
    trajectory = reduce(lambda x, y: x + y, trajectories, [])
    states = [transition.obs for transition in trajectory]
    states = np.array(states)
    pos, vels = states[:, :2], states[:, 2:]

    # subplot pos, vels
    
    plt.subplot(2, 1, 1)
    plt.scatter(pos[:, 0], pos[:, 1], s=5)
    plt.title('Positions')
    plt.subplot(2, 1, 2)
    plt.scatter(vels[:, 0], vels[:, 1], s=5)
    plt.title('Velocities')
    plt.figure()

    # plot trajectories
    trajectories = [traj for traj in trajectories if len(traj) > 0]
    for traj in trajectories:
        states = [transition.obs for transition in traj]
        actions = [transition.action for transition in traj]
        executed = [transition.executed for transition in traj]
        goals = np.array([compute_grid_goal(s[:2], directions[a], n_pos=N, tol=tol) for s, a, e in zip(states, actions, executed) if e == 1])
        
        executed = np.array(executed)
        states = np.array(states)[executed==1]
        actions = np.array(actions)[executed==1, None]
       
        printarr(states, actions, executed)
        if executed.sum() >= 1:
            print(np.concatenate([states[:, :2], actions, goals], axis=-1))
            pos, vels = states[:, :2], states[:, 2:]
            # set limits between 0 1 
            # plt.xlim(0, 1)
            # plt.ylim(0, 1)
            plt.scatter(goals[:-1, 0], goals[:-1, 1], c='r')
            plt.scatter(states[0:1, 0], states[0:1, 1], c='g', marker='*')
            plt.scatter(states[-1:, 0], states[-1:, 1], c='b', marker='*')
            plt.plot(pos[:, 0], pos[:, 1], c='k')
    plt.grid()
    plt.show()


def visualize_random_policy(config_path):
    env = Pinball(config_path, render_mode='human')
    n_steps = 50
    # s =[0.08345238, 0.85161418, 0., 0.]
    # s = env.reset(np.array(s))
    s = env.reset()
    next_s = s
 
    grid_options = PinballGridOptions(env, n_pos=20, tol=1/20/5)
    for n in tqdm(range(n_steps)):
        init = compute_initiation_masks(s, grid_options)
        choices = np.where(init == 1)[0]
    
        a = random.choice(choices)
        print(f'Action: {a}, Initiation: {init}, Choices: {choices}')
        option = grid_options[a]
        can_execute = option.execute(s)
        env.render()
        ts = int(1/60*1000)
        pygame.time.wait(ts)
        max_exec_time = 100
        t = 0
        if can_execute:
            while option.is_executing() and t < max_exec_time:
                action = option.act(next_s)
                if action is None:
                    break # option terminated
                next_s, r, done, _, info = env.step(action)
                # env.render()
                # pygame.time.wait(ts)
              
                t += 1
        s = next_s
        if t >= max_exec_time:
            print('Option timed out')
            print(f'Action {a}, Initial State {s}, goal {compute_grid_goal(s[:2], directions[a], n_pos=N, tol=tol)}')


        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='envs/pinball/configs/pinball_simple_single.cfg')
    args = parser.parse_args()
    
    # print(compute_grid_goal(np.array([0.22889519, 0.09499182]), np.array([0, 1]), n_pos=N, tol=tol))
    # print(compute_grid_goal(np.array([0.90277734, 0.57896749]), np.array([0,1]), n_pos=N, tol=tol))
    # visualize_random_policy(args.config)
    test_grid_controllers(args.config)
    