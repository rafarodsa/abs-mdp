'''
    Visualize 2D latent space
'''


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import torch 

import argparse

from src.absmdp.vae import AbstractMDPTrainer
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

    
if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-ckpt', type=str)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--config', type=str, default='experiments/vae/config/config.yaml')
    parser.add_argument('--save-path', type=str, default='.')
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
        obs, next_obs = b.obs, b.next_obs
        _obs.append(obs)
        _next_obs.append(next_obs)
    obs, next_obs = torch.cat(_obs, dim=0), torch.cat(_next_obs, dim=0)

    print(f'obs.mean {obs.mean(0)} obs.std {obs.std(0)}')
    print(f'next_obs.mean {next_obs.mean(0)} next_obs.std {next_obs.std(0)}')

    Batch = namedtuple('batch', ['obs', 'next_obs'])
    batch = Batch(obs, next_obs)

    with torch.no_grad():
        z_sample, s, q_s = model(batch.obs)
        z = q_s[0]
        next_z_sample, next_s, q_next_s = model(batch.next_obs)
        next_z = q_next_s[0]
        

    # Plot encoder space
   
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1], s=5, marker='o', color='b')
    plt.scatter(next_z[:, 0], next_z[:, 1], s=5, marker='^')    
    plt.savefig(f'{args.save_path}/z-space.png')

    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Truth')
    plt.scatter(obs[:,0], obs[:, 1], marker='x', s=5, color='b')
    plt.scatter(next_obs[:, 0], next_obs[:, 1], marker='x', s=5, color='r')

    plt.subplot(1,2,2)
    plt.title('Encoded/Decoded')
    plt.scatter(s[:, 0], s[:, 1], marker='o', s=5, color='b')
    plt.scatter(next_s[:, 0], next_s[:, 1], marker='o', s=5, color='g')
    plt.savefig(f'{args.save_path}/reconstruction.png')

    plt.figure()
    plt.scatter(obs[:,0], obs[:, 1], marker='x', s=5, color='b')
    plt.scatter(next_obs[:, 0], next_obs[:, 1], marker='o', s=5, color='b')
    plt.scatter(s[:, 0], s[:, 1], marker='x', s=5, color='g')
    plt.scatter(next_s[:, 0], next_s[:, 1], marker='o', s=5, color='g')
    plt.savefig(f'{args.save_path}/reconstruction_overlap.png')

