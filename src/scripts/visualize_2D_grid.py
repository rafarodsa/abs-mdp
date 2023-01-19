import matplotlib.pyplot as plt
import numpy as np
import torch 

from src.abstract_mdp.abs_mdp import AbstractMDP


def predict_next_states(mdp, states, actions, executed):
    next_s = []
    next_z = []
    for action in actions:
        actions_ = action * torch.ones(states.shape[0])
        next_z_ = mdp.transition(mdp.encoder(states), actions_.long(), executed)
        next_s.append(mdp.ground(next_z_))
        next_z.append(next_z_)
    return next_s, next_z

def states_to_plot(states, n_grids=10):
    return torch.round(n_grids * states)

def test_grounding(mdp, states):
    z = mdp.encoder(states)
    s = mdp.ground(z)
    return s, z


# Arguments
path_to_model = './pinball_no_obstacle.mdp'
n_samples = 10000

# Load
mdp = AbstractMDP.load(path_to_model)

# plot initial state distribution

states = torch.from_numpy(mdp.get_initial_states(n_samples))
executed = torch.ones(n_samples) # TODO check that this flag is correct in dataset
# plot effect of each option

actions = mdp.get_actions()
s = states_to_plot(states)
next_states, next_z = predict_next_states(mdp, s/10, actions, executed)




# plot in subplots the effect of each action

reconstructed_s, z = test_grounding(mdp, states)
grounded_init_states = states_to_plot(reconstructed_s)

plt.figure()
plt.scatter(z[:, 0], z[:, 1], marker='*')
for action in range(4):
    plt.scatter(next_z[action][:,0], next_z[action][:,1], marker='>')


plt.figure()
for action in range(mdp.n_options):
    ax = plt.subplot(2,2, action+1)
    # next_s = states_to_plot(next_states[action])
    next_s = next_states[action]
    idx = (s[:, 0] % 2 == 0) * (s[:, 1] % 2 == 0) 
    ax.set_title(f"Action {action}")
    plt.scatter(next_s[idx, 0]*10, next_s[idx, 1]*10, marker='+')
    plt.scatter(s[idx, 0], s[idx, 1], marker='x')
    # plt.scatter(grounded_init_states[idx, 0], grounded_init_states[idx, 1])


plt.show()
