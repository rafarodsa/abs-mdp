from src.envs.pinball.pinball_gym import PinballEnvContinuous, PinballEnv
from src.envs.pinball.controllers_pinball import initiation_set, position_controller_discrete, termination_velocity, position_controller_continuous

import numpy as np
import matplotlib.pyplot as plt

def test_ball_collision():
    env = PinballEnv("/Users/rrs/Desktop/abs-mdp/src/envs/pinball/configs/pinball_simple_single.cfg")
    tests = [((0.01, 0.99, 0, -0.1), True), # doesnt hit anything
             ((0.01, 0.99, 0, 0.1), False), # hit wall
             ((0.25, 0.4, 0.1, 0), False), # hit obstacle exactly in the vertex
             ((0.7, 0.03, 0.1, 0), False) # hit edge inside
            ]

    for test in tests:
        print(test)
        _init_f = initiation_set(env, np.array(test[0][2:]))
        assert _init_f(np.array(test[0][:2])) == test[1]

def test_run_pid_controller(displacement, kp_vel, ki_vel, kp_pos, kd_pos, horizon=100):
    env = PinballEnv("/Users/rrs/Desktop/abs-mdp/src/envs/pinball/configs/pinball_no_obstacles.cfg")
    s = env.reset()
    goal = np.concatenate([s[:2], np.zeros(2)], axis=0) + displacement
    controller = position_controller_discrete(goal, kp_vel, ki_vel, kp_pos, kd_pos)
    trajectory = [s]
    for _ in range(horizon):
        s = env.step(controller(s))[0]
        trajectory.append(s)
    
    error = goal - s
    pos_error = np.sqrt(np.linalg.norm(error[:2]))
    vel_error = np.sqrt(np.linalg.norm(error[2:]))
    return trajectory, pos_error, vel_error, goal

def test_run_pid_controller_continuous(displacement, kp_vel, ki_vel, kp_pos, kd_pos, horizon=100):
    env = PinballEnvContinuous("/Users/rrs/Desktop/abs-mdp/src/envs/pinball/configs/pinball_no_obstacles.cfg")
    s = env.sample_initial_positions(1)
    s = env.reset(s[0])
    goal = np.concatenate([s[:2], np.zeros(2)], axis=0) + displacement
    controller = position_controller_continuous(goal, kp_vel, ki_vel, kp_pos, kd_pos)
    trajectory = [s]
    for _ in range(horizon):
        s = env.step(controller(s))[0]
        trajectory.append(s)
    
    error = goal - s
    pos_error = np.sqrt(np.linalg.norm(error[:2]))
    vel_error = np.sqrt(np.linalg.norm(error[2:]))
    return trajectory, pos_error, vel_error, goal

def plot_trajectory(trajectory, goal):
    time = np.arange(len(trajectory))
    states = np.array(trajectory)

    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(time, states[:, i])
        plt.plot(time, goal[i] * np.ones_like(time)) 
    
    plt.show()

def plot_trajectory_with_termination(trajectory, goal, terminated):
    time = np.arange(len(trajectory))
    states = np.array(trajectory)
    # terminated_time = list(map(lambda t: int(np.random.random(1) < t), terminated))[1:].index(1)
    terminated_time = len(time)

    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(time[:terminated_time], states[:terminated_time, i])
        plt.plot(time[:terminated_time], goal[i] * np.ones_like(time[:terminated_time])) 
        plt.plot(time, terminated)
    
    plt.show()

def tune_pid():
    result = test_run_pid_controller(np.array([0.1, 0.1, 0, 0]), 10, 0.1, 100, 0.)
    plot_trajectory(result[0], result[-1])

def tune_pid_continuous():
    result = test_run_pid_controller_continuous(np.array([0.1, 0.01, 0, 0]), 5, 0.01, 100, 0.)
    plot_trajectory(result[0], result[-1])


def test_termination():
    
    result = test_run_pid_controller(np.array([0.1, 0.1, 0, 0]), 10, 0.1, 100, 0.)
    
    std_vel = 0.1
    std_pos = 0.001
    termination_f = termination_velocity(std_dev=std_vel)
    terminated = list(map(termination_f, result[0]))

    plot_trajectory_with_termination(result[0], result[-1], terminated)

def test_termination_continuous():
    
    result = test_run_pid_controller_continuous(np.array([0.1, 0.1, 0, 0]), 5, 0.01, 100, 0.)
    
    std_vel = 0.01
    termination_f = termination_velocity(std_dev=std_vel)
    terminated = list(map(termination_f, result[0]))

    plot_trajectory_with_termination(result[0], result[-1], terminated)



if __name__ == "__main__":
    test_ball_collision()
    test_termination_continuous()
    # tune_pid_continuous()    



