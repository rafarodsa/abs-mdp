'''
    Visualize 2D latent space
'''


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import torch 

import argparse

from src.absmdp.trainer import AbstractMDPTrainer
from src.absmdp.datasets import PinballDataset

from omegaconf import OmegaConf as oc


from sklearn.decomposition import PCA

from collections import namedtuple

def predict_next_states(mdp, states, actions, executed):
    next_s = []
    next_z = []
    for action in actions:
        actions_ = action * torch.ones(states.shape[0])
        z = mdp.encoder(states)
        next_z_ = mdp.transition(z, actions_.long(), executed)
        next_s.append(mdp.ground(next_z_))
        next_z.append(next_z_)
    return next_s, next_z

def states_to_plot(states, n_grids=10):
    return torch.round(n_grids * states)

def test_grounding(mdp, states):
    z = mdp.encoder(states)
    s = mdp.ground(z)
    return s, z

def many_gaussian_values(means, vars):
    batch = means.shape[0]
    stds = np.sqrt(vars)
    gaussians = []
    for b in range(batch):
        mean, std = means[b][:2], stds[b][:2]
        min_v, max_v = mean-2*std, mean + 2 *std
        x, y = np.linspace(min_v[0],max_v[0], 200), np.linspace(min_v[1],max_v[1], 200)
        xv, yv = np.meshgrid(x, y)
        pts = np.dstack((xv,yv))
        rv = multivariate_normal(mean, np.diag(vars[b]))
        z = rv.pdf(pts)/batch
        gaussians.append(np.stack((xv,yv,z), axis=0))

    return gaussians
        

    


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-ckpt', type=str)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--config', type=str, default='experiments/pb_obstacles/fullstate/config/config.yaml')
    parser.add_argument('--save-path', type=str)
    args = parser.parse_args()

    # Load config
    cfg = oc.load(args.config)

    # Load
    model = AbstractMDPTrainer.load_from_checkpoint(args.from_ckpt, cfg=cfg)
    

    data = PinballDataset(cfg.data)
    data.setup()
    transform = np.linalg.inv(data.linear_transform)
    batches = list(data.test_dataloader())
    batch = batches[0]

    _obs, _action, _next_obs = [], [], []
    for b in batches:
        obs, action, next_obs = b.obs, b.action, b.next_obs
        _obs.append(obs)
        _action.append(action)
        _next_obs.append(next_obs)
    obs, action, next_obs = torch.cat(_obs, dim=0), torch.cat(_action, dim=0), torch.cat(_next_obs, dim=0)


    Batch = namedtuple('batch', ['obs', 'action', 'next_obs'])
    batch = Batch(obs, action, next_obs)

    with torch.no_grad():
        z_q = model.encoder.distribution(batch.obs)
        z = z_q.mean
        next_z_q = model.encoder.distribution(batch.next_obs)
        next_z = next_z_q.mean

        transition_in = torch.cat((z, batch.action), dim=-1)
        predicted_z, q_z, _ = model.transition.sample_n_dist(transition_in, 1)
        predicted_z = q_z.mean
        print(predicted_z.shape)
        predicted_next_s_q = model.decoder.distribution(predicted_z)
        predicted_next_s = predicted_next_s_q.mean
        avg_std = predicted_next_s_q.var.sqrt().mean(-1).mean()
        print(f'Avg total state encoding deviation {z_q.var.sqrt().mean(-1).mean()} per dim')
        print(f'Avg total grounding deviation {avg_std} per dim')
        print(f'Avg total next state encoding deviation {next_z_q.var.sqrt().mean(-1).mean()} per dim')
        print(f'Avg prediction MSE {((next_z - predicted_z) ** 2).sum(-1).sqrt().mean()}')

        encoded_transition = next_z - z
        predicted_transition = predicted_z - z
        

    # Plot encoder space
    _action = batch.action.argmax(-1)
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1], s=5, marker='o', color='b')

    for a in range(4):
        next_z_a = predicted_z[_action==a]
        plt.scatter(next_z_a[:, 0], next_z_a[:, 1], s=5, marker='^', label=f'action {a}')    
    plt.savefig(f'{args.save_path}/z-space.png')
    # Plot initial states


    
    acts = [1, 3]
    plt.figure()
    plt.subplot(1, 2, 1)

    for a in acts:
        plt.quiver(z[_action==a, 0], z[_action==a, 1], encoded_transition[_action==a, 0], encoded_transition[_action==a, 1], angles='xy', scale_units='xy', scale=1)
        plt.scatter(z[_action == a, 0], z[_action == a, 1], s=5, label=f'action {a}', color='b')
        plt.scatter(next_z[_action == a, 0], next_z[_action == a, 1], s=5, label=f'action {a}', color='r')

    plt.subplot(1, 2, 2)
    for a in acts:
        plt.quiver(z[_action==a, 0], z[_action==a, 1], predicted_transition[_action==a, 0], predicted_transition[_action==a, 1], angles='xy', scale_units='xy', scale=1)
        plt.scatter(z[_action == a, 0], z[_action == a, 1], s=5, label=f'action {a}', color='b')
        plt.scatter(predicted_z[_action == a, 0], predicted_z[_action == a, 1], s=5, label=f'action {a}', color='r')

    plt.savefig(f'{args.save_path}/latent_space.png')


    # # plot reconstruction of next state.
    # plt.figure()
    # gaussians = many_gaussian_values(next_z_q.mean[_action==a], next_z_q.var[_action==a])
    
    # for g in gaussians:
    #     plt.contour(g[0], g[1], g[2])
    # error = predicted_z - next_z
    # plt.quiver(next_z[_action==a, 0], next_z[_action==a, 1], error[_action==a, 0], error[_action==a, 1], angles='xy', scale_units='xy', scale=1)
    # plt.scatter(predicted_z[_action == a, 0], predicted_z[_action == a, 1], s=5, label=f'action {a}', color='r')
    # plt.scatter(next_z[_action == a, 0], next_z[_action == a, 1], label=f'action {a}', color='k', marker='x')


    # plt.savefig(f'{args.save_path}/error.png')

    
    # pca = PCA(2)
    # pca = pca.fit(batch.obs)
    transform = np.eye(4)
    s = np.einsum('ji, bj->bi', transform, batch.obs.numpy())
    next_s, pred_next_s = np.einsum('ji, bj->bi',transform ,batch.next_obs.numpy()), np.einsum('ji, bj->bi', transform, predicted_next_s.numpy())
    d_real = next_s - s
    d_pred = pred_next_s - s

    plt.figure()
    plt.subplot(1,2,1)
    for a in acts:
        plt.scatter(s[_action == a, 0], s[_action == a, 1], marker='x', s=5, color='b')
        plt.scatter(next_s[_action==a, 0], next_s[_action == a, 1], marker='x', s=5, color='r')
        plt.quiver(s[_action == a, 0], s[_action == a, 1], d_real[_action == a, 0], d_real[_action == a, 1], angles='xy', scale_units='xy', scale=1)
    # plt.quiver(s[:, 0], s[:, 1], d_pred[:, 0], d_pred[:, 1])
    plt.subplot(1,2,2)
    for a in acts: 
        plt.scatter(s[_action == a, 0], s[_action == a, 1], marker='x', s=5, color='b')
        plt.scatter(pred_next_s[_action == a, 0], pred_next_s[_action == a, 1], marker='o', s=5, color='g')
        plt.quiver(s[_action == a, 0], s[_action == a, 1], d_pred[_action == a, 0], d_pred[_action == a, 1], angles='xy', scale_units='xy', scale=1)

    plt.savefig(f'{args.save_path}/pca_s.png')

