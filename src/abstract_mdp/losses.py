"""
    Loss functions for learning an abstract MDP
    author: Rafael Rodriguez-Sanchez
    email: rrs@brown.edu
    date: November 2022
"""
import torch


###
# TODO: Add hyperparameter to sum losses
# TODO: Generalize observation dimension to n-dimensional observation
# TODO: Compute KL using torch distributions
# TODO: VAEs with Discrete encoding distributions.

class abstraction_encoding_loss(torch.nn.Module):
    def __init__(self, encoding_dim, observation_dim, beta, sigma_epsilon=1e-3):
        self.beta = beta
        self.epsilon = sigma_epsilon
        self.encoding_dim = encoding_dim
        self.observation_dim = observation_dim

    def forward(self, target_s_prime, predicted_s_prime, encoded_s):
        prediction_loss = self._prediction_loss(target_s_prime, predicted_s_prime)
        encoding_loss = self._encoding_loss(encoded_s)
        return -((1-self.beta) * prediction_loss - self.beta * encoding_loss)

    def _encoding_loss(self, encoded_s):
        return 0.5 * encoded_s[..., 0:self.encoding_dim].pow(2).sum(dim=-1) + self.encoding_dim * (torch.exp(encoded_s[..., -1]) - encoded_s[..., -1])

    def _prediction_loss(self, target_s_prime, predicted_s_prime):
        obs_dim = self.observation_dim
        return -0.5 * (target_s_prime - predicted_s_prime[..., 0:obs_dim]).pow(2).sum(dim=-1, keepdim=True)/(torch.exp(2*predicted_s_prime[..., obs_dim:]) + self.epsilon) - obs_dim * predicted_s_prime[..., obs_dim:]

    def _consistency_loss(self, transition_params, encoded_s_prime): # mean-seeking
        epsilon=self.epsilon
        mean_t = transition_params[..., :latent_dim]
        log_sigma_t = transition_params[..., latent_dim:]
        mean_z = encoded_s_prime[..., :latent_dim].unsqueeze(1)
        log_sigma_z = encoded_s_prime[..., latent_dim:].unsqueeze(1)
        entropy = self.encoding_dim * 2 * log_sigma_t

        _loss = 0.5 * (2 * self.encoding_dim * log_sigma_t  + ((mean_z.pow(2) + torch.exp(2*log_sigma_z))/(torch.exp(2*log_sigma_t) + epsilon)).sum(dim=-1, keepdim=True) - 2*torch.einsum('...i, ...j->...', mean_t, mean_z).unsqueeze(-1)/(torch.exp(2*log_sigma_t)+epsilon) + torch.einsum('...i, ...j->...', mean_t, mean_t).unsqueeze(-1)/(torch.exp(2*log_sigma_t)+epsilon))
        return (_loss - entropy).mean(-2)


# def loss(target, predicted_s_prime, predicted_z):
    
#     target = target.unsqueeze(1).unsqueeze(1)
#     encoding_loss = 0.5 * predicted_z[..., 0:latent_dim].pow(2).sum(dim=-1) + latent_dim * (torch.exp(predicted_z[..., -1]) - predicted_z[..., -1])
#     prediction_loss = -0.5 * (target - predicted_s_prime[..., 0:obs_dim]).pow(2).sum(dim=-1, keepdim=True)/(torch.exp(2*predicted_s_prime[..., obs_dim:]) + epsilon) - obs_dim * predicted_s_prime[..., obs_dim:]
#     _loss = -((1-beta)*prediction_loss - beta * encoding_loss)
#     return _loss


# def contrained_transition(transition_params, encoded_z_prime): # mean-seeking
#     epsilon=1e-5
#     mean_t = transition_params[..., :latent_dim]
#     log_sigma_t = transition_params[..., latent_dim:]
#     mean_z = encoded_z_prime[..., :latent_dim].unsqueeze(1)
#     log_sigma_z = encoded_z_prime[..., latent_dim:].unsqueeze(1)
#     entropy = latent_dim * 2 * log_sigma_t

#     _loss = 0.5 * (2 * latent_dim * log_sigma_t  + ((mean_z.pow(2) + torch.exp(2*log_sigma_z))/(torch.exp(2*log_sigma_t) + epsilon)).sum(dim=-1, keepdim=True) - 2*torch.einsum('...i, ...j->...', mean_t, mean_z).unsqueeze(-1)/(torch.exp(2*log_sigma_t)+epsilon) + torch.einsum('...i, ...j->...', mean_t, mean_t).unsqueeze(-1)/(torch.exp(2*log_sigma_t)+epsilon))
#     return (_loss - entropy).mean(-2)