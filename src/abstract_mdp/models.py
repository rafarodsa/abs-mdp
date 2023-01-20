'''
    Model Factories
    author: Rafael Rodriguez-Sanchez
    date: November 2022
'''
import torch
import torch.nn as nn
from functools import partial


##  Pixel Models

class ConvResidual(nn.Module):
    def __init__(self, in_width, in_height, in_channels, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding='same')
        self.norm_1 = nn.LayerNorm([out_channels, in_width, in_height])
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding='same')
        self.norm_2 = nn.LayerNorm([out_channels, in_width, in_height])
        self.relu = nn.ReLU()

    def forward(self, x):
        conv_1 = self.relu(self.norm_1(self.conv_1(x)))
        conv_2 = self.relu(self.norm_2(self.conv_2(conv_1)))
        return conv_2 + x


def conv_out_dim(in_dim, kernel_size, stride, padding):
    return int((in_dim - kernel_size + 2 * padding) / stride + 1)


def encoder_conv_continuous(in_width, in_height, hidden_dim=128, latent_dim=2):
    encoder_feats = nn.Sequential(
                                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding='valid'),
                                nn.LayerNorm([16, conv_out_dim(in_width, kernel_size=3, padding=0, stride=2), conv_out_dim(in_height, kernel_size=3, padding=0, stride=2)]),
                                nn.ReLU(),
                                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding='valid'),
                                nn.LayerNorm([32, conv_out_dim(in_width, kernel_size=3, padding=0, stride=2), conv_out_dim(in_height, kernel_size=3, padding=0, stride=2)]),
                                ConvResidual(conv_out_dim(in_width, kernel_size=3, padding=0, stride=2), conv_out_dim(in_height, kernel_size=3, padding=0, stride=2), 32, 32),    
                                nn.Flatten(),
                                nn.Linear(32 * conv_out_dim(in_width, kernel_size=3, padding=0, stride=2) * conv_out_dim(in_height, kernel_size=3, padding=0, stride=2), hidden_dim),
                                nn.ReLU()
                    )

    encoder_mean = nn.Linear(32 * 3 * 3, latent_dim)

    encoder_log_var = nn.Linear(32 * 3 * 3, latent_dim)

    return GaussianDensity((encoder_feats, encoder_mean, encoder_log_var))

##  Single Vector Models

def encoder_fc(obs_dim, hidden_size, latent_dim):
    encoder_feats = nn.Sequential(
                                nn.Linear(obs_dim, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU()
                        )

    encoder_mean = nn.Linear(hidden_size, latent_dim)

   
    encoder_log_var = nn.Linear(hidden_size, latent_dim)

    return GaussianDensity((encoder_feats, encoder_mean, encoder_log_var))


class GaussianDensity(nn.Module):

    def __init__(self, models):
        super().__init__()
        self.feats, self.mean, self.log_var = models[0], models[1], models[2]
        self.double()

    def forward(self, input):
        feats = self.feats(input)
        mean, log_var = self.mean(feats), self.log_var(feats)
        return mean, log_var

    def sample(self, input, n_samples=1, epsilon=1e-3):
        mean, log_var = self.forward(input)
        std = torch.exp(log_var / 2)
        std = std + torch.ones_like(std) * epsilon
        std_normal = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        q = torch.distributions.Normal(mean, std)
        q_no_grad = torch.distributions.Normal(mean.detach(), std.detach())
        z = q.rsample(torch.zeros(n_samples).size())
        return z, q, std_normal, q_no_grad

#### Grounding/Decoder Models

def constant_var(dim, epsilon, x):
    return torch.ones(dim) * epsilon

def decoder_fc(obs_dim, hidden_size, latent_dim):
    grounding_feats = nn.Sequential(
                                    nn.Linear(latent_dim, hidden_size),
                                    nn.ReLU(),
                                    # nn.Linear(hidden_size, hidden_size),
                                    # nn.ReLU()
                    )

    grounding_mean = nn.Linear(hidden_size, obs_dim)
    # grounding_log_var = nn.Linear(hidden_size, obs_dim) # TODO fix variance going to zero.
    epsilon = 1e-5
    grounding_log_var =  partial(constant_var, obs_dim, epsilon) #nn.Linear(hidden_size, latent_dim)
    return GaussianDensity((grounding_feats, grounding_mean, grounding_log_var))



### Transition Models

def transition_fc_deterministic(latent_dim, n_actions, hidden_size):

    transition_model = nn.Sequential(
                                    nn.Linear(latent_dim + n_actions + 1, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, latent_dim)
                                )

    return transition_model

def transition_fc_gaussian(latent_dim, n_actions, hidden_size):
    transition_feats = nn.Sequential(
                                    nn.Linear(latent_dim + n_actions, hidden_size),
                                    nn.ReLU()
                                )
    transition_mean = nn.Linear(hidden_size, latent_dim)
    transition_log_var = nn.Linear(hidden_size, latent_dim)
    return GaussianDensity((transition_feats, transition_mean, transition_log_var))





### Reward models

def reward_fc(latent_dim, hidden_size, n_action):
    reward_model = nn.Sequential(
                                    nn.Linear(latent_dim * 2 + n_action, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1)
                    )

    return reward_model



### Initiation Classifiers

def initiation_classifier(latent_dim, hidden_size, n_actions):

    initiation_classifier = nn.Sequential(
                                    nn.Linear(latent_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, n_actions),
                                    nn.Sigmoid()
    )
    return initiation_classifier