'''
    Visualize 2D latent space
'''


import matplotlib.pyplot as plt
import numpy as np
import torch 

import argparse

from src.absmdp.infomax_attn import InfomaxAbstraction
from src.absmdp.tpc_critic import InfoNCEAbstraction as TPCAbstraction
from src.absmdp.discrete_tpc_critic import DiscreteInfoNCEAbstraction as DiscreteTPCAbstraction
from src.absmdp.datasets import PinballDataset
from src.absmdp.tpc_critic_rssm import RSSMAbstraction
from src.absmdp.datasets_traj import PinballDatasetTrajectory

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
    parser.add_argument('--discrete', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--rssm', action='store_true')
    args = parser.parse_args()

    # Load config
    cfg = oc.load(args.config).cfg
    
    # Load
    # model = InfomaxAbstraction.load_from_checkpoint(args.from_ckpt, cfg=cfg)
    if not args.rssm:
        if not args.discrete:
            model = TPCAbstraction.load_from_checkpoint(args.from_ckpt)
        else:
            model = DiscreteTPCAbstraction.load_from_checkpoint(args.from_ckpt)
        data = PinballDataset(cfg.data)
    else:
        model = RSSMAbstraction.load_from_checkpoint(args.from_ckpt)
        data = PinballDatasetTrajectory(cfg.data)

    data.setup()

    # trainer = pl.Trainer()
    # trainer.test(model, data)

    batches = list(data.test_dataloader())
    batch = batches[0]


    device = args.device
    _obs, _action, _next_obs = [], [], []
    for b in batches:
        obs, action, next_obs = b.obs, b.action, b.next_obs
        _obs.append(obs.to(device))
        _action.append(action.to(device))
        _next_obs.append(next_obs.to(device))
    obs, action, next_obs = torch.cat(_obs, dim=0), torch.cat(_action, dim=0), torch.cat(_next_obs, dim=0)


    Batch = namedtuple('batch', ['obs', 'action', 'next_obs'])
    batch = Batch(obs, action, next_obs)
    with torch.no_grad():
        model.eval()
        model.to(device)
        z_q = model.encoder(batch.obs)
        z = z_q
        next_z_q = model.encoder(batch.next_obs)
        next_z = next_z_q

        transition_in = torch.cat((z, batch.action), dim=-1)
        q_z = model.transition.distribution(transition_in)
        predicted_z = q_z.mean + z
        action = batch.action.argmax(-1)

    if args.rssm:
        z = z.reshape(-1, z.shape[-1])
        next_z = next_z.reshape(-1, z.shape[-1])
        predicted_z = predicted_z.reshape(-1, z.shape[-1])
        _action = action.reshape(-1)

    # prepare for plotting. move to cpu
    if device != 'cpu':
        z = z.cpu()
        predicted_z = predicted_z.cpu()
        next_z = next_z.cpu()
        _action = _action.cpu()
   

    os.makedirs(args.save_path, exist_ok=True)
    # Plot encoder space
    
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1], s=5, marker='o', color='b', label='z')
    action_names = {0: '-y', 1: '+y', 2: '-x', 3: '+x'}
    for a in range(4):
        next_z_a = next_z[_action==a]
        plt.scatter(next_z_a[:, 0], next_z_a[:, 1], s=5, marker='^', label=f'next_z: action {action_names[a]}')   
    plt.legend() 
    plt.savefig(f'{args.save_path}/z-space.png')
    # Plot initial states
    
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1], s=5, marker='o', color='b', label='z')
    plt.savefig(f'{args.save_path}/latent_s.png')


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


