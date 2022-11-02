from src.options.option import Option
from visgrid.gridworld import GridWorld
from visgrid.gridworld.skills import GoToGridPosition
from fourroom import FourRoom
import numpy as np
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--rows', type=int, default=10)
parser.add_argument('--cols', type=int, default=10)
args = parser.parse_args()

# define environment
env = FourRoom(rows=args.rows, cols=args.cols)

# define options for FourRooms
col_min, cols_mid, cols_max = 0, args.cols//2, args.cols
row_min, row_mid_left, row_mid_right, row_max = 0, args.rows//2, args.rows//2 + 1, args.rows

# initiation sets

def room_initiation(row_min, row_max, col_min, col_max):
    def _initation(state):
        row, col = tuple(state)
        return row >= row_min and row <= row_max and col >= col_min and col <= col_max
    return _initation
room_1 = room_initiation(0, row_mid_left-1, 0, cols_mid-1)
room_2 = room_initiation(row_mid_left, row_max-1, 0, cols_mid-1)
room_3 = room_initiation(row_mid_right, row_max - 1, cols_mid, cols_max -1)
room_4 = room_initiation(0, row_mid_right-1, cols_mid, cols_max-1)

# room mid positions
room_1_mid = (row_mid_left//2, cols_mid//2)
room_2_mid = ((row_mid_left + row_max)//2, cols_mid//2)
room_3_mid = ((row_mid_left + row_max)//2, (cols_max + cols_mid)//2)
room_4_mid = (row_mid_right//2, (cols_max + cols_mid)//2)

# termination conditions
beta = -np.log(0.7) / (args.cols//4 + args.rows//4)
room_1_term_prob = lambda state: np.exp(-beta*np.sum(np.abs(np.array(room_1_mid) - state))) * room_1(state)
room_2_term_prob = lambda state: np.exp(-beta*np.sum(np.abs(np.array(room_2_mid) - state))) * room_2(state)
room_3_term_prob = lambda state: np.exp(-beta*np.sum(np.abs(np.array(room_3_mid) - state))) * room_3(state)
room_4_term_prob = lambda state: np.exp(-beta*np.sum(np.abs(np.array(room_4_mid) - state))) * room_4(state)

# Policy function 
slipping_prob = 0.1
def policy(target, prob_success=slipping_prob):
    def _policy(state):
        search_return, _ = GoToGridPosition(env, state, target)
        return search_return[1] if np.random.binomial(1,prob_success) == 1 else np.random.choice(env.actions)
    return _policy


rooms = [room_1, room_2, room_3, room_4]
rooms_mid = [room_1_mid, room_2_mid, room_3_mid, room_4_mid]
rooms_termination = [room_1_term_prob, room_2_term_prob, room_3_term_prob, room_4_term_prob]
options = []
for room in range(4):
    options.append(Option(rooms[room], policy(rooms_mid[(room+1)%4]), rooms_termination[(room+1)%4], f"room{room+1}->room{(room+1)%4+1}"))
    options.append(Option(rooms[room], policy(rooms_mid[(room-1)%4]), rooms_termination[(room-1)%4], f"room{room+1}->room{(room-1)%4+1}"))


# generate data

def execute_policy(state, option):
    s = state
    executed = option.is_executable(s)
    done = not option.is_executable(s)
    reward = []
    while not done:
        action, terminated = option.act(s)
        if not terminated:
            next_s, r, done = env.step(action)
            s = next_s
            reward.append(r)
        done = terminated
    return (state, option, reward, s, executed)

n_samples = 2000
samples = {}
for option in options: 
    _samples = []
    _executed = []
    for i in range(n_samples):
        env.reset_agent()
        sample = execute_policy(env.get_state(), option)
        _samples.append((sample[0], sample[-2]))
        _executed.append(sample[-1])
    samples[str(option)] = (_samples, _executed)

# plot data

ax = env.plot()
markers = ["v", "^", "<", ">", "1", "2", "3","4", "8"]

for i, option in enumerate(options):
    data = samples[str(options[i])]
    points = np.array(data[0])
    executed = np.array(data[1])
    points_init = points[:, 0, :] + np.random.normal(0.5, 0.1, 1)
    points_end = points[executed, 1, :] + np.random.normal(0.5, 0.1, 1)
    # ax.scatter(points_init[:, 1], points_init[:,0], marker=markers[i])
    ax.scatter(points_end[:, 1], points_end[:, 0], marker=markers[i])

plt.show()


    

        



