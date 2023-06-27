'''
    Visualize 2D latent space
'''


import matplotlib.pyplot as plt
import numpy as np
import torch 

import argparse

from src.absmdp.infomax_attn import InfomaxAbstraction
from src.absmdp.tpc_critic import InfoNCEAbstraction as TPCAbstraction
from src.absmdp.datasets import PinballDataset

from omegaconf import OmegaConf as oc


from sklearn.decomposition import PCA

from collections import namedtuple
import os

import lightning as pl

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
    # model = InfomaxAbstraction.load_from_checkpoint(args.from_ckpt, cfg=cfg)
    model = TPCAbstraction.load_from_checkpoint(args.from_ckpt, cfg=cfg)


    data = PinballDataset(cfg.data)


    data.setup()

    # trainer = pl.Trainer()
    # trainer.test(model, data)

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
        model.eval()
        z_q = model.encoder(batch.obs)
        z = z_q
        next_z_q = model.encoder(batch.next_obs)
        next_z = next_z_q

        transition_in = torch.cat((z, batch.action), dim=-1)
        predicted_z, q_z, _ = model.transition.sample_n_dist(transition_in, 1)
        predicted_z = q_z.mean
        # predicted_next_s_q = model.grounding.distribution(predicted_z)
        # predicted_next_s = predicted_next_s_q.sample()
        # decoded_next_s_q = model.grounding.distribution(next_z)


    os.makedirs(args.save_path, exist_ok=True)
    # Plot encoder space
    _action = batch.action.argmax(-1)
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1], s=5, marker='o', color='b', label='z')
    action_names = {0: '-y', 1: '+y', 2: '-x', 3: '+x'}
    for a in range(4):
        next_z_a = next_z[_action==a]
        plt.scatter(next_z_a[:, 0], next_z_a[:, 1], s=5, marker='^', label=f'next_z: action {action_names[a]}')   
    plt.legend() 
    plt.savefig(f'{args.save_path}/z-space.png')
    # Plot initial states
    
    acts = [1, 3]
    plt.figure()
    plt.subplot(1, 2, 1)

    for a in acts:
        # plt.quiver(z[_action==a, 0], z[_action==a, 1], next_z[_action==a, 0], next_z[_action==a, 1], angles='xy', scale_units='xy', scale=1)
        plt.scatter(z[_action == a, 0], z[_action == a, 1], s=5, label=f'z', color='b')
        plt.scatter(next_z[_action == a, 0], next_z[_action == a, 1], s=5, label=f'next_z: action {action_names[a]}', color='r')

    plt.subplot(1, 2, 2)
    for a in acts:
        # plt.quiver(z[_action==a, 0], z[_action==a, 1], predicted_z[_action==a, 0], predicted_z[_action==a, 1], angles='xy', scale_units='xy', scale=1)
        plt.scatter(z[_action == a, 0], z[_action == a, 1], s=5, label=f'z', color='b')
        plt.scatter(predicted_z[_action == a, 0], predicted_z[_action == a, 1], s=5, label=f'next_z: action {action_names[a]}', color='g')
    plt.legend()
    plt.savefig(f'{args.save_path}/latent_space.png')


