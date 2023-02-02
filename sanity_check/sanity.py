from base64 import encode
from importlib.util import module_from_spec
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# Constants

X_MIN, X_MAX, Y_MIN, Y_MAX = 0, 10, 0, 10
obs_dim = 10
state_dim = 2

# Centroids
centroids = []
for x in (X_MAX//4, X_MAX * 3//4):
    for y in (Y_MAX//4, Y_MAX * 3//4):
        centroids.append(np.array((x, y)))

rooms = []
for x in (X_MAX//2, X_MAX):
    for y in (Y_MAX//2, Y_MAX):
        rooms.append((x-X_MAX//2, y-Y_MAX//2, x, y))

def initiation_set(room_limits):
    def _init(state):
        low_limit = room_limits[0:2]
        high_limit = room_limits[2:]
        return not (state[0] > low_limit[0] and state[0] < high_limit[0] and state[1] > low_limit[1] and state[1] < high_limit[1])
    return _init


# Synthetic option definition

effect_dists_params = []
initiation_sets = []
std_dev = 1
for i, room in enumerate(rooms):
    initiation_sets.append(initiation_set(room))
    effect_dists_params.append((centroids[i], std_dev))

# Generate starting states

N = 1000

x = np.random.uniform(0, X_MAX, N)
y = np.random.uniform(0, Y_MAX, N)

s = np.array((x,y)).T

masks = []
for i, init_set in enumerate(initiation_sets):
    mask = np.zeros((N,1))
    for j in range(N):
        mask[j] = float(init_set(s[j]))
    masks.append(mask)

masks = np.array(masks)

# generate next states
s_prime = []
for i, effect_dist in enumerate(effect_dists_params):
    s_prime.append(np.random.multivariate_normal(effect_dist[0], effect_dist[1] * np.eye(2), size=N))


s_prime = np.array(s_prime) * masks  + s[np.newaxis] * (1-masks)
masks = masks.reshape((-1,))

# Random affine transformation

T = np.random.rand(obs_dim, state_dim)
x = np.einsum('ij, kj->ki', T, s)
x_prime = np.einsum('ij, lkj->lki', T, s_prime)

data = []
for i in range(len(effect_dists_params)):
    data.append(np.array((x, x_prime[i])))

# Plot to Check
# for r in range(1,3):
#     ax = plt.axes()    
#     ax.scatter(s[:, 0], s[:, 1])
#     ax.scatter(s_prime[r,:,0], s_prime[r,:,1])
#     plt.show()


# Models
n_actions = 4
hidden_size = 64
latent_dim = 2
encoder = nn.Sequential(
                                nn.Linear(obs_dim + n_actions, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, latent_dim + 1)
                        )

transition_model = nn.Sequential(
                                    nn.Linear(latent_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, latent_dim + 1)
                                )

grounding_model = nn.Sequential(
                                    nn.Linear(latent_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, obs_dim + 1)
                                )



# Training


### Training hyperparameters
n_epochs = 50
minibatch_size = 32
n_samples = 1  # samples to approximate expectation
learning_rate = 1e-4
beta = 1.  # hyperparameter to control information bottleneck


# Cast data to float torch tensor and add action one-hot encoding
in_data = []
out_data = []
for i in range(n_actions):
    data[i] = torch.from_numpy(data[i]).float()
    actions = nn.functional.one_hot(torch.ones(data[i].size(1)).long() * i, n_actions)
    data[i] = (torch.cat((data[i][0], actions), dim=1), data[i][1]) 
    in_data.append(data[i][0])
    out_data.append(data[i][1])

in_data = torch.cat(in_data, dim=0)[masks == 1]
out_data = torch.cat(out_data, dim=0)[masks == 1]


# shuffle dataset

rand_order = torch.randperm(in_data.size(0))
in_data = torch.index_select(in_data, 0, rand_order)
out_data = torch.index_select(out_data, 0, rand_order)

def minibatches(in_data, out_data, batch_size):
    N = in_data.size(0)
    n_batches = N//batch_size + int(N%batch_size != 0)
    for i in range(n_batches):
        yield (in_data[i*batch_size:(i+1)*batch_size], out_data[i*batch_size:(i+1)*batch_size])
    

models = (encoder, transition_model, grounding_model)
models_params = []
for model in models:
    for param in model.parameters():
        models_params.append(param)
optimizer = torch.optim.Adam(models_params, lr=learning_rate)


for _ in range(n_epochs):
    for _in, _out in minibatches(in_data, out_data, minibatch_size):

        ## Forward pass
        z = encoder(_in)  # encode
        
        # Predict next abstract state
        noise = torch.normal(0, 1, (_in.size(0), n_samples, latent_dim))
        z_ = torch.exp(z.unsqueeze(1)[:, :, -2:-1]) * noise + z.unsqueeze(1)[:,:, :-1]   
        z_prime = transition_model(z_)
        
        epsilon = torch.normal(0, 1, (_in.size(0), n_samples, latent_dim))
        z_prime_samples = torch.exp(z_prime[:, :, -2:-1]) * epsilon + z_prime[:,:, :-1]  

        # predict next ground state
        s_prime = grounding_model(z_prime_samples)
        # epsilon = torch.normal(0, 1, (_in.size(0), n_samples, obs_dim))
        # s_prime_samples = torch.exp(s_prime[:, :, -2:-1]) * epsilon + s_prime[:,:, :-1]

        s_prime = s_prime.squeeze(1)
        encoding_loss = 0.5 * (z[:, 0:latent_dim] * z[:, 0:latent_dim]).sum(dim=-1) + latent_dim * (torch.exp(z[:, -1]) - z[:, -2:-1])
        prediction_loss = -0.5 * (_out - s_prime[:, 0:obs_dim]).pow(2).sum(dim=-1) - obs_dim * s_prime[:, -2:-1]

        loss = -(prediction_loss - beta * encoding_loss).mean()
        
        ### zero grads 
        optimizer.zero_grad()
        ### backward pass
        loss.backward()
        print(loss)
        ### update
        optimizer.step()


## TODO: save model
## TODO: save data
## TODO: move evaluation to another script



# Evaluate
with torch.no_grad():
    for i in range(n_actions):
        z = encoder(data[i][0])
        z_prime = transition_model(z[:, 0:-1])
        s_prime = grounding_model(z[:, 0:-1])
        plt.scatter(z[:, 0], z[:, 1], label=f"encoded action {i}")
        plt.scatter(z_prime[:, 0], z_prime[:, 1], label=f"prediction action {i}")
        # plt.scatter(s_prime[:, 0], s_prime[:, 1], label=f"action {i}")
    plt.legend()
    plt.show()






