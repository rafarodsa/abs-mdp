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

from src.absmdp.tpc_critic_rssm import RSSMAbstraction
from src.absmdp.datasets_traj import PinballDatasetTrajectory
from src.utils import printarr
from omegaconf import OmegaConf as oc
import seaborn as sns

from sklearn import manifold

from collections import namedtuple
from scripts.utils import get_experiment_info, prepare_outdir
from tqdm import tqdm
import os


from sklearn.feature_selection import mutual_info_regression

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--rssm', action='store_true')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--title', type=str, default='')

    args = parser.parse_args()

    config_file, ckpt_path = get_experiment_info(args.experiment)

    print(f'Loading config from {config_file}')
    print(f'Loading ckpt from {ckpt_path}')
    
    save_path = f'{args.experiment}/visualization'
    print(f'Saving at {save_path}')

    # Load config
    cfg = oc.load(config_file).cfg
    
    if len(args.dataset) > 0:
        cfg.data.data_path = args.dataset

    # Load
    # model = InfomaxAbstraction.load_from_checkpoint(args.from_ckpt, cfg=cfg)
    if not args.rssm:
        model = TPCAbstraction.load_from_checkpoint(ckpt_path, cfg=cfg)
        data = PinballDataset(cfg.data)
    else:
        model = RSSMAbstraction.load_from_checkpoint(ckpt_path)
        data = PinballDatasetTrajectory(cfg.data)

    data.setup()
    data = data.dataset

    _obs, _action, _next_obs, _info = [], [], [], []
    # for b in tqdm(range(len(data) // 4)):
    for b in range(100):
        b = data[b]
        obs, action, next_obs, length = b.obs, b.action, b.next_obs, b.length
        _obs.append(obs[:length].to(args.device))
        _action.append(action[:length].to(args.device))
        _next_obs.append(next_obs[:length].to(args.device))
        _info.append(b.info[:length])
    obs, action, next_obs = torch.cat(_obs, dim=0), torch.cat(_action, dim=0), torch.cat(_next_obs, dim=0)



    Batch = namedtuple('batch', ['obs', 'action', 'next_obs'])
    batch = Batch(obs, action, next_obs)

    with torch.no_grad():
        model.eval()
        model.to(args.device)
        z_q = model.encoder(batch.obs)
        z = z_q
        next_z_q = model.encoder(batch.next_obs)
        next_z = next_z_q

        transition_in = torch.cat((z, batch.action), dim=-1)
        q_z = model.transition.distribution(transition_in)
        predicted_z = q_z.mean + z

    _action = batch.action.argmax(-1)
    
    if args.rssm:
        z = z.reshape(-1, z.shape[-1])
        next_z = next_z.reshape(-1, z.shape[-1])
        predicted_z = predicted_z.reshape(-1, z.shape[-1])
        _action = _action.reshape(-1)

    if args.device != 'cpu':
        z = z.cpu()
        next_z = next_z.cpu()
        predicted_z = predicted_z.cpu()
        _action = _action.cpu()
       

    states = []
    for traj in _info:
        for t in traj:
            states.append(t['state'])
    states = np.stack(states, axis=0)
    energy = np.abs(states[..., 0] + states[..., 1])
    printarr(states, obs, z, energy)


    sns.set_context('talk', font_scale=2)
    sns.set_style("ticks")
    sns.color_palette("husl")



    os.makedirs(save_path, exist_ok=True)


        # Prepare an empty matrix to hold mutual information values
    mutual_info_matrix = np.empty((states.shape[1], z.shape[1]))

    # Compute mutual information for each pair of features
    for i in range(states.shape[1]):
        for j in range(z.shape[1]):
            mutual_info_matrix[i, j] = mutual_info_regression(states[:, i].reshape(-1, 1), z[:, j])[0]

    # Plot heatmap using Seaborn
    plt.figure(figsize=(9, 9))
    with sns.axes_style("white"):
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(mutual_info_matrix,
                    # xticklabels=[f"z_{i}" for i in range(z.shape[1])], 
                    xticklabels=[],
                    yticklabels=[],
                    # yticklabels=[f"s_{i}" for i in range(states.shape[1])],
                    #robust=True,
                    #cma=cmap,
                    # square=True
                    annot=False, 
                )
    plt.xlabel("Abstract Features (z)", labelpad=10)
    plt.ylabel("Ground Features (s)", labelpad=10)
    # plt.title("Mutual Information")
    plt.tight_layout()
    plt.savefig(f'{save_path}/mutual_info.png', dpi=300)
    print(f'Mutual Info heatmap saved at {save_path}/mutual_info.png')
    plt.savefig(f'{save_path}/mutual_info_transparent.png', dpi=300, transparent=True)
    
    if cfg.model.latent_dim == 2:
        plt.figure(figsize=(8,9))
        plt.scatter(z[:, 0], z[:, 1], s=5, marker='o', color='b', label='z')
        action_names = {0: '-y', 1: '+y', 2: '-x', 3: '+x'}
        for a in range(4):
            next_z_a = next_z[_action==a]
            plt.scatter(next_z_a[:, 0], next_z_a[:, 1], s=5, marker='^', label=f'next_z: action {action_names[a]}')   
        plt.legend() 
        plt.savefig(f'{save_path}/z-space.png')
        # Plot initial states
        
        acts = [1, 3]
        plt.figure(figsize=(8,9))
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

    if cfg.model.latent_dim > 2:

        # Plot encoder space
        n_neighbors = 10
        n_components = 2
        plt.figure(figsize=(8,9))
        try:
            print('MDS...')
            Y = manifold.MDS(n_components=n_components, normalized_stress='auto').fit_transform(z.double().detach().numpy())
            plt.scatter(Y[:, 0], Y[:, 1], s=3, c=energy, cmap='mako')
            plt.title(args.title)
            plt.gca().set(xticks=[], yticks=[])
            plt.tight_layout()
            plt.savefig(f'{save_path}/mds-z-space.png', dpi=300)
            plt.savefig(f'{save_path}/mds-z-space-transparent.png', dpi=300, transparent=True)
        except Exception as e:
            print('MDS failed...', e)
        print('Done...')

        plt.figure(figsize=(8,9))
        # plt.title('Ground State Space')
        plt.scatter(states[:, 0], states[:, 1], s=3, c=energy, cmap='mako')
        plt.gca().set(xticks=[], yticks=[])
        plt.tight_layout()
        plt.savefig(f'{save_path}/ground_space.png', dpi=300)
        plt.savefig(f'{save_path}/ground_space-transparent.png', dpi=300, transparent=True)

        



