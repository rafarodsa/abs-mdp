"""
    Options for Pinball Environments
    author: Rafael Rodriguez-Sanchez
    date: October 2022 
    email: rrs@brown.edu

"""
import numpy as np
from itertools import tee
from scipy.special import erf, softmax

### Control Position

def initiation_set(pinball_environment, distance):
    def __initiation(state):
        if ball_collides(pinball_environment, state[:2], state[:2] + distance):
            return False
        return True
    return __initiation


def ball_collides(pinball_env, initial_position, final_position):
    """ 
        Detects collision at position
    """
    for obstacle in pinball_env.get_obstacles():
        if _intersect_obstable(obstacle, initial_position, final_position):
            return True
    return False
        

def _intersect_obstable(obstacle, initial_position, final_position):
    points = obstacle.points
    
    a, b = tee(np.vstack([np.array(points), points[0]]))
    next(b, None)
    
    for edge in zip(a, b):
        alpha, beta = _intersect(edge, initial_position, final_position)
        if alpha <= 1 and alpha >= 0 and beta >= 0 and beta <= 1:
            return True
    return False

def _intersect(edge, initial_position, final_position):
    displacement = np.array(final_position) - np.array(initial_position)
    edge_segment = np.array(edge[1])-np.array(edge[0])
    b = edge[0] - initial_position
    alpha = (edge_segment[1] * b[0]-b[1])/(displacement[0]*edge_segment[1] - displacement[1])
    beta = (alpha * displacement[1] - b[1])/edge_segment[1]

    return alpha, beta


def termination(goal_position, std_dev=0.001):
    """
        Factory for termination probability.
    """
    def __termination(state):
        if isinstance(std_dev, float):
            z = np.linalg.norm(state-goal_position)/std_dev
        else:
            z = np.linalg.norm((state-goal_position)/std_dev)
        tail_prob = 1-erf(z/np.sqrt(2))
        return tail_prob
    return __termination


class PID:
    def __init__(self, kp, ki=0., kd=0.):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.
        self.integral_error = 0.
    
    def __call__(self, error):
        self.integral_error += error
        derivative_error = self.prev_error - error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral_error + self.kd * derivative_error


ACC_X = 0
ACC_Y = 1
DEC_X = 2
DEC_Y = 3
ACC_NONE = 4

def position_controller(goal, kp_vel, ki_vel, kp_pos, kd_pos):
    """
        Factory for controller to achive goal position
    """
    controller_pos = PID(kp=kp_pos, ki= 1, kd=kd_pos)
    controller_vel = PID(kp=kp_vel, ki=ki_vel)
    actions = [[DEC_X, ACC_X], [DEC_Y, ACC_Y]]
    def __controller(state): 
        error = goal - state
        u = controller_pos(error[:2]) + controller_vel(error[2:]) # velocity control signal
        if np.linalg.norm(u) < 1/5: 
            return ACC_NONE # do nothing
        else:
            # probs = softmax(np.abs(u))
            # sample = np.random.choice([0,1], p=probs) # select axis to act
            sample = np.argmax(np.abs(u)) # select axis
            return actions[sample][int(u[sample]>=0)]
            
    return __controller