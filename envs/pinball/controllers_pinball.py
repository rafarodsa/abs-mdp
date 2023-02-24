"""
    Options for Pinball Environments
    author: Rafael Rodriguez-Sanchez
    date: October 2022 
    email: rrs@brown.edu

"""
import numpy as np
from itertools import tee
from scipy.special import erf, softmax
from src.options.option import Option
from functools import partial
from gym import spaces
### Control Position

### Initiation Sets and Terminations
def initiation_set(pinball_environment, distance):
    def __initiation(state):
        return not ball_collides(pinball_environment, state[:2], state[:2] + distance)
    return __initiation


def ball_collides(pinball_env, initial_position, final_position):
    """ 
        Detects collision at position
    """


    for obstacle in pinball_env.get_obstacles():
        if _intersect_obstable(obstacle, initial_position, final_position):
            # print(f'Obstacle intersected: {initial_position}->{final_position}')
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

    A = np.stack([displacement, edge_segment], axis=1)
    try:
        coeff = np.linalg.solve(A, b)
        alpha, beta = coeff[0], coeff[1]
    except np.linalg.LinAlgError:
        alpha, beta = np.float('inf'), np.float('inf')
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

def termination_velocity(std_dev=0.001):
    """
        Factory for termination probability when velocity is close to zero
    """
    def __termination(state):
        if isinstance(std_dev, float):
            z = np.linalg.norm(state[..., 2:])/std_dev
        else:
            z = np.linalg.norm((state[..., 2:])/std_dev)
        tail_prob = 1-erf(z/np.sqrt(2))
        return tail_prob
    return __termination

### Controllers.

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

def position_controller_discrete(goal, kp_vel, ki_vel, kp_pos, kd_pos):
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

def position_controller_continuous(goal, kp_vel, ki_vel, kp_pos, kd_pos):
    """
        Factory for controller to achive goal position
    """
    controller_pos = PID(kp=kp_pos, ki= 1, kd=kd_pos)
    controller_vel = PID(kp=kp_vel, ki=ki_vel)
    MAX_SPEED = 0.99
    def __controller(state): 
        error = goal - state
        u = controller_pos(error[:2]) + controller_vel(error[2:]) # velocity control signal
        return u.clip(-MAX_SPEED, MAX_SPEED)
            
    return __controller

def position_controller_factory(init_state, distance, continuous=True):
    return position_controller_continuous(init_state + distance, 5, 0.01, 100, 0.) if continuous else position_controller_discrete(init_state+distance, 10, 0.1, 100, 0.)

def create_position_controllers(env, translation_distance=1/10):
    position_options = []
    controller_factory = position_controller_factory
    std_dev_vel = 0.01
    for y in [-1., 1.]:
        o = Option(initiation_set(env, np.array([0., y*translation_distance])),
               partial(controller_factory, distance=np.array([0., y*translation_distance, 0., 0.]), continuous=isinstance(env.action_space, spaces.Box)),
               termination_velocity(std_dev=std_dev_vel),
               name=f"{'+' if y > 0 else '-'}Y"
        )
        position_options.append(o)
    for x in [-1., 1.]:
        o = Option(initiation_set(env, np.array([x*translation_distance, 0.])),
               partial(controller_factory, distance=np.array([x*translation_distance, 0., 0., 0.]), continuous=isinstance(env.action_space, spaces.Box)),
               termination_velocity(std_dev=std_dev_vel),
               name=f"{'+' if x > 0 else '-'}X"
        )
        position_options.append(o)
    
    
    return position_options
    

