'''
    Visualize 2D latent space
'''


import matplotlib.pyplot as plt
import numpy as np
import torch 

import argparse

from src.absmdp.infomax_attn import InfomaxAbstraction
from src.absmdp.datasets import PinballDataset

from omegaconf import OmegaConf as oc
from collections import namedtuple
import os


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-ckpt', type=str)
    parser.add_argument('--experiment_path', type=str)
    args = parser.parse_args()

    # Load config

    config_file = f'{args.experiment_path}/phi_train/logs/infomax-pb/version_0/hparams.yaml'
    ckpt_path = f'{args.experiment_path}/phi_train/ckpts/{args.from_ckpt}'
    save_path = f'{args.experiment_path}/vis'
    cfg = oc.load(config_file).cfg
    
    # Load
    model = InfomaxAbstraction.load_from_checkpoint(ckpt_path, cfg=cfg)
    

    data = PinballDataset(cfg.data)
    data.setup()
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
        predicted_z = q_z.mean + z
        predicted_next_s_q = model.grounding.distribution(torch.cat([predicted_z, torch.zeros_like(batch.action)], dim=-1))
        predicted_next_s = predicted_next_s_q.sample()
        decoded_next_s_q = model.grounding.distribution(torch.cat([next_z, torch.zeros_like(batch.action)], dim=-1))


    os.makedirs(save_path, exist_ok=True)
    # Plot encoder space
    _action = batch.action.argmax(-1)
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1], s=5, marker='o', color='b', label='z')
    action_names = {0: '-y', 1: '+y', 2: '-x', 3: '+x'}
    for a in range(4):
        next_z_a = next_z[_action==a]
        plt.scatter(next_z_a[:, 0], next_z_a[:, 1], s=5, marker='^', label=f'next_z: action {action_names[a]}')   
    plt.legend() 
    plt.savefig(f'{save_path}/z-space.png')
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
    plt.savefig(f'{save_path}/latent_space.png')


