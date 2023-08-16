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

from src.utils.printarr import printarr
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
    # add batch dim
    if len(initial_position.shape) == 1:
        initial_position = initial_position[None]
    if len(final_position.shape) == 1:
        final_position = final_position[None]
    batch = initial_position.shape[0]

    ball_radius = pinball_env.pinball.ball.radius
    collisions = []
    for obstacle in pinball_env.vertices:
        displacement = final_position - initial_position # [N, 2]
        d = np.linalg.norm(displacement, axis=1, keepdims=True) # + 1e-12
        # inside_expansion = pinball_env._inside_polygon(obstacle, final_position + displacement * ball_radius / d) 
        close_to_vertex = np.any(distances_path_to_vertices(obstacle, initial_position - displacement * ball_radius / d, final_position + displacement * ball_radius / d)[0] <= ball_radius * 1.1, axis=0) 
        final_pos_close_to_edge = distances_path_to_vertices(final_position + displacement * ball_radius / d, obstacle[:-1], obstacle[1:])[0] <= ball_radius * 1.1
        final_pos_close_to_edge = np.any(final_pos_close_to_edge, axis=1)
        intersect = _intersect_obstable(obstacle, initial_position, final_position, ball_radius=ball_radius)

        collision = np.logical_or(np.logical_or(intersect, final_pos_close_to_edge), close_to_vertex)
        collisions.append(collision)
        if batch == 1:
            if np.any(collision):
                return True
        
    collisions = np.stack(collisions, axis=0)
    return np.any(collisions, axis=0) if batch > 1 else False

def _intersect_obstable(obstacle, initial_position, final_position, ball_radius=0.02, eps=1e-1):
    edges = np.stack([obstacle[:-1], obstacle[1:]], axis=-1)
    alpha, beta = _intersect_batch(edges, initial_position, final_position, ball_rad=ball_radius)
    _intersections = np.logical_and(np.logical_and(alpha <= 1+eps, alpha >= -eps), np.logical_and(beta >= -eps, beta <= 1+eps)) # [M, N]
    theres_collision = np.any(_intersections, axis=0) #.reshape(-1, 3) # N
    return theres_collision

def _intersect_batch(edges, initial_positions, final_positions, ball_rad=0.02):
    '''
        edges: batch of edges (M, 2, 2)
        initial_positions: batch of initial positions (N, 2)
        final_positions: batch of final positions (N, 2)
    '''
    N, M = initial_positions.shape[0], edges.shape[0]
    displacement = final_positions - initial_positions # [N, 2]
    d = np.linalg.norm(displacement, axis=1, keepdims=True) # + 1e-12
    displacement = displacement / d
    # displacement = ball_rad * displacement/d[:, None] + displacement # [N, 2]
    # final_positions += displacement / d[:, None] * ball_rad
    
    # printarr(displacement)
    normal_d = np.zeros_like(displacement)
    normal_d[:, 0] = -displacement[:, 1]
    normal_d[:, 1] = displacement[:, 0]
    # normal_d = normal_d/d

    # overshoot = 0.
    # final_positions = np.concatenate([final_positions + displacement * ball_rad * overshoot, final_positions + normal_d * ball_rad + displacement * ball_rad*overshoot, final_positions - normal_d * ball_rad + displacement * ball_rad * overshoot], axis=0)
    # initial_positions = np.concatenate([initial_positions, initial_positions + normal_d * ball_rad, initial_positions - normal_d * ball_rad], axis=0)
    displacement = final_positions + displacement * ball_rad  - initial_positions # [N, 2]
    N, M = initial_positions.shape[0], edges.shape[0]

    edge_segment = edges[..., 1] - edges[..., 0] # [M, 2]
    # edge_vector = edge_segment / np.linalg.norm(edge_segment, axis=-1, keepdims=True)
    # edges[..., 1] += edge_vector * ball_rad
    # edges[..., 0] -=  edge_vector * ball_rad   
    # edge_segment = edges[..., 1] - edges[..., 0] # [M, 2]

    # assert np.allclose(new_edge_segment, edge_segment)

    b = edges[..., 0][:, None] - initial_positions[None] # [M, N, 2]
    
    # repeat displacement and edge_segment to match batch size
    displacement = np.tile(displacement[np.newaxis], reps=(M, 1, 1)) + 1e-8 # [M, N, 2]
    edge_segment = np.repeat(edge_segment[:, np.newaxis], N, axis=1) + 1e-8 # [M, N, 2]
    A = np.stack([-edge_segment, displacement], axis=-1)
    try:
        coeff = np.linalg.solve(A, b)
        alpha, beta = coeff[:, :, 0], coeff[:, :, 1] # [M, N]
        # _i = initial_positions[None].repeat(M, axis=0)
        # intersects_0 = _i + beta[..., None] * displacement
        # _e = edges[..., 0][:, None].repeat(N, axis=1)
        # intersects_1 = _e + alpha[..., None] * edge_segment

        # assert np.allclose(intersects_0, intersects_1)  
       
    except np.linalg.LinAlgError as e:
        DET_EPS = 1e-8
        det = np.linalg.det(A)
        non_sing_matrices = np.abs(det) > DET_EPS
        coeff = np.linalg.solve(A[non_sing_matrices], b[non_sing_matrices])

        alpha = np.zeros((M, N)) + np.inf 
        alpha[non_sing_matrices] = coeff[..., 0]
        beta = np.zeros((M, N)) + np.inf 
        beta[non_sing_matrices] = coeff[..., 1]

    
    return alpha, beta


def distances_path_to_vertices(vertices, initial_position, final_position, eps=0.1):
    n = final_position - initial_position # (M, 2)
    segment_length = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / segment_length

    i_to_vtx = vertices[:, None] - initial_position[None]  # (Vertices, lines, 2)
    
    coeff = np.einsum('ijd, ijd -> ij', i_to_vtx, n[None])[..., None] # (Vertices, lines, 1)
    projection = initial_position[None] + coeff  * n[None] # (Vertices, lines, 2)
    # print(projection[0])
    # distance = np.linalg.norm(i_to_vtx - projection, axis=-1) # (Vertices, lines)
    normal = projection - vertices[:, None]
    # print(normal[0])
    distance = np.linalg.norm(normal, axis=-1)
    # printarr(i_to_vtx, projection, distance, coeff, segment_length, n, normal)
    coeff = coeff[..., 0] / segment_length[None,..., 0]
    in_segment = np.logical_and(coeff <= 1+eps, coeff >= -eps)
    distance[~in_segment] = np.inf
    return distance, in_segment


def initiation_set_batch(pinball_env, distance):
    _edges = []
    for obstacle in pinball_env.get_obstacles():
        points = obstacle.points
        a, b = tee(np.vstack([np.array(points), points[0]]))
        next(b, None)
        e = list(zip(a, b))
        edges = np.array(e)
        _edges.append(edges)
    edges = np.concatenate(_edges, axis=0)

    def __initiation(state):
        if len(state.shape) == 1: # not batched
            state = state[np.newaxis]
        return np.logical_not(ball_collides_batch(edges, state[:, :2], state[:, :2] + distance))
    return __initiation

def ball_collides_batch(edges, initial_positions, final_positions, ball_radius=0.02):
    '''
        initial_positions: batch of initial positions (N, 2)
        final_positions: batch of final positions (N, 2)
    '''
    alpha, beta = _intersect_batch(edges, initial_positions, final_positions, ball_radius)
    
    # intersections = (alpha <= 1) * (alpha >= 0) * (beta >= 0) * (beta <= 1)  # [M, N]
    _intersections = np.logical_and(np.logical_and(alpha <= 1, alpha >= 0), np.logical_and(beta >= 0, beta <= 1)) # [M, N]
    theres_collision = np.any(_intersections, axis=0) # N
    return theres_collision

def termination_position(init_state, distance, std_dev=0.001):
    """
        Factory for termination probability.
    """
    goal_position = (init_state + distance)
    def __termination(state):
        distance = (state-goal_position)[:2]
        if isinstance(std_dev, float):
            z = np.linalg.norm(distance)/std_dev
        else:
            z = np.linalg.norm(distance/std_dev)
        tail_prob = 1-erf(z/np.sqrt(2))
        return tail_prob
    return __termination

def termination_velocity(init_state, distance, std_dev=0.001):
    """
        Factory for termination probability when velocity is close to zero
    """
    goal_state = init_state + distance
    goal_state[2:] = 0. # zero velocity
    def __termination(state):
        error = goal_state - state
        if isinstance(std_dev, float):
            z = np.linalg.norm(error)/std_dev
        else:
            z = np.linalg.norm(error/std_dev)
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
    controller_pos = PID(kp=kp_pos, ki=1, kd=kd_pos)
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


def position_controller_continuous(goal, kp_vel, ki_vel, kp_pos, kd_pos, ki_pos=1):
    """
        Factory for controller to achive goal position
    """
    controller_pos = PID(kp=kp_pos, ki=ki_pos, kd=kd_pos)
    controller_vel = PID(kp=kp_vel, ki=ki_vel)
    MAX_SPEED = 0.99
    def __controller(state): 
        error = goal - state
        u = controller_pos(error[:2]) + controller_vel(error[2:]) # velocity control signal
        return u.clip(-MAX_SPEED, MAX_SPEED)
            
    return __controller

def position_controller_factory(init_state, distance, continuous=True):
    PID_PARAMS = {
         "kp_vel": 8.,
         "ki_vel": 0.,
         "kp_pos": 50, 
         "kd_pos": 0.,
         "ki_pos": 10
    }
    # return position_controller_continuous(init_state + distance, kp_vel=8, ki_vel=0., kp_pos=20, kd_pos=0., ki_pos=10) if continuous else position_controller_discrete(init_state+distance, kp_vel=10, ki_vel=0.1, kp_pos=50, kd_pos=0., ki_pos=1)
    return position_controller_continuous(init_state + distance, **PID_PARAMS) if continuous else position_controller_discrete(init_state+distance, kp_vel=10, ki_vel=0.1, kp_pos=50, kd_pos=0., ki_pos=1)

def create_position_controllers_v0(env, translation_distance=1/10):
    '''
        Creates options for moving the agent in the four directions by a fixed distance.
        Velocity is zero.
    '''
    position_options = []
    controller_factory = position_controller_factory
    std_dev_vel = 0.01
    for y in [-1., 1.]:
        distance = np.array([0., y*translation_distance, 0., 0.])
        o = Option(initiation_set(env, distance[:2]),
               partial(controller_factory, distance=distance, continuous=isinstance(env.action_space, spaces.Box)),
               partial(termination_velocity, distance=distance, std_dev=std_dev_vel),
               name=f"{'+' if y > 0 else '-'}Y"
        )
        position_options.append(o)
    for x in [-1., 1.]:
        distance = np.array([x*translation_distance, 0., 0., 0.])
        o = Option(initiation_set(env, distance[:2]),
               partial(controller_factory, distance=distance, continuous=isinstance(env.action_space, spaces.Box)),
               partial(termination_velocity, distance=distance, std_dev=std_dev_vel),
               name=f"{'+' if x > 0 else '-'}X"
        )
        position_options.append(o)
    
    
    return position_options

def create_position_options(env, translation_distance=1/15, std_dev=0.01, check_can_execute=True):
    '''
        Creates options for moving the agent in the four directions by a fixed distance.
        Velocity is zero.
    '''
    position_options = []
    controller_factory = position_controller_factory
    std_dev_vel = translation_distance/5/3  #0.01
    for y in [-1., 1.]:
        distance = np.array([0., y*translation_distance, 0., 0.])
        o = Option(initiation_set(env, distance[:2]),
               partial(controller_factory, distance=distance, continuous=isinstance(env.action_space, spaces.Box)),
               partial(termination_position, distance=distance, std_dev=std_dev_vel),
               name=f"{'+' if y > 0 else '-'}Y", 
               check_can_execute=check_can_execute
        )
        position_options.append(o)
    for x in [-1., 1.]:
        distance = np.array([x*translation_distance, 0., 0., 0.])
        o = Option(initiation_set(env, distance[:2]),
               partial(controller_factory, distance=distance, continuous=isinstance(env.action_space, spaces.Box)),
               partial(termination_position, distance=distance, std_dev=std_dev_vel),
               name=f"{'+' if x > 0 else '-'}X",
               check_can_execute=check_can_execute
        )
        position_options.append(o)
    
    
    return position_options

def compute_grid_goal(position, direction, n_pos, tol, min_pos=0.05, max_pos=0.95):
        batch = position.shape[0] if len(position.shape) > 1 else 1
        step = (max_pos - min_pos) / n_pos
        pos_ = (position - min_pos) / (max_pos-min_pos) * np.abs(direction) * n_pos
        min_i, max_i = np.floor(pos_), np.ceil(pos_) # closest grid position to move
        goal = min_i * (direction < 0) + max_i * (direction > 0)
        distances = np.abs(goal * step + min_pos - position) * np.abs(direction)
        goal = (goal * step + min_pos) * (distances > tol) +  ((goal+direction) * step + min_pos) * (distances <= tol)
        goal = goal * (direction != 0) + position * (direction == 0)
        
        # print(f'Direction {direction}, Position {position} -> Goal {goal}')
        other_dir = 1-np.abs(direction)
        # print(other_dir)
        goal_correction = (position-min_pos)/ (max_pos-min_pos) * other_dir * n_pos
        # print(goal_correction, np.floor(goal_correction), np.ceil(goal_correction))
        min_p, max_p = step * np.floor(goal_correction) + min_pos, step * np.ceil(goal_correction) + min_pos # closest grid position to move
        # print(min_p, max_p)
        distance_min, distance_max = (np.abs(min_p - position) * other_dir).sum(-1), (np.abs(max_p - position) * other_dir).sum(-1)
        if batch == 1:
            goal_correction = min_p if distance_min < distance_max else max_p
            if np.abs(goal_correction - position).sum(-1) > tol:
                goal = goal * (direction != 0) + goal_correction * (direction == 0)
        else:
            goal_correction = np.where((distance_min < distance_max)[:, np.newaxis], min_p, max_p)
            goal = np.where(np.abs(goal_correction - position).sum(-1, keepdims=True) > tol, goal * (direction != 0) + goal_correction * (direction == 0), goal)

        goal = goal.clip(min_pos, max_pos)
        # print(f'Direction {direction}, Position {position} -> Goal {goal}')
        return goal

def PinballGridOptions(env, n_pos=20, tol=1/20/5):
    '''
        Options to move agent in the four coordinate dimensions to fixed positions in 
        the space.
        params:
            n_pos: number of positions per dimension
            tol: tolerance to be considered at the position 
    '''
    position_options = []    


    def initiation_set(env, direction):
        _edges = []
        for obstacle in env.get_obstacles():
            points = obstacle.points
            a, b = tee(np.vstack([np.array(points), points[0]]))
            next(b, None)
            e = list(zip(a, b))
            edges = np.array(e)
            _edges.append(edges)
        edges = np.concatenate(_edges, axis=0)

        def __initiation(state):
            batch = 1 if len(state.shape) == 1 else state.shape[0]
            if batch == 1:
                state = state.reshape(1, -1)
            goal = compute_grid_goal(state[:, :2], direction, n_pos, tol)
            
            # if batch > 1:
            #     goals = [compute_grid_goal(state[i, :2], direction, n_pos, tol) for i in range(batch)]
            #     goal_test = np.vstack(goals)
            #     # print(goal_test, goal)
            #     assert np.allclose(goal, goal_test)

            executable = 1-ball_collides_batch(edges, state[:, :2], goal)
            if batch == 1:
                executable = executable[0]
            # print(f'State {state} -> Goal {goal} is executable: {executable}')
            return executable
        return __initiation

    def termination(init_state, direction, std_dev=tol/2):
        goal = compute_grid_goal(init_state[:2], direction, n_pos, tol)
        def __termination(state):
            
            distance = (state[:2]-goal)
            if isinstance(std_dev, float):
                z = np.linalg.norm(distance/std_dev)
            else:
                z = np.linalg.norm(distance/std_dev)
            tail_prob = 1-erf(z/np.sqrt(2))
            return tail_prob
        return __termination

    def grid_pos_controller(state, direction, n_pos, tol, continuous=True):
        position = state[:2]
        goal_pos = compute_grid_goal(position, direction, n_pos, tol)
        goal = np.concatenate([goal_pos, state[2:]])
        return position_controller_continuous(goal, 5, 0.01, 150, 0., 10) if continuous else position_controller_discrete(goal, 10, 0.1, 100, 1.)

    for y in [-1., 1.]:
        direction = np.array([0, y])
        o = Option(initiation_set(env, direction),
               partial(grid_pos_controller, direction=direction, n_pos=n_pos, tol=tol, continuous=isinstance(env.action_space, spaces.Box)),
               partial(termination, direction=direction, std_dev=tol/3),
               name=f"{'+' if y > 0 else '-'}Y"
        )
        position_options.append(o)
    for x in [-1., 1.]:
        direction = np.array([x, 0])
        o = Option(initiation_set(env, direction),
               partial(grid_pos_controller, direction=direction, n_pos=n_pos, tol=tol, continuous=isinstance(env.action_space, spaces.Box)),
               partial(termination, direction=direction, std_dev=tol/3),
               name=f"{'+' if x > 0 else '-'}X"
        )
        position_options.append(o)
    return position_options

