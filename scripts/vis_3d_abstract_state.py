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

from omegaconf import OmegaConf as oc


from sklearn import manifold

from collections import namedtuple
import os

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
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--rssm', action='store_true')
    args = parser.parse_args()

    # Load config
    cfg = oc.load(args.config).cfg
    
    # Load
    # model = InfomaxAbstraction.load_from_checkpoint(args.from_ckpt, cfg=cfg)
    if not args.rssm:
        model = TPCAbstraction.load_from_checkpoint(args.from_ckpt, cfg=cfg)
        data = PinballDataset(cfg.data)
    else:
        model = RSSMAbstraction.load_from_checkpoint(args.from_ckpt)
        data = PinballDatasetTrajectory(cfg.data)

    data.setup()

    batches = list(data.test_dataloader())
    batch = batches[0]

    _obs, _action, _next_obs = [], [], []
    for b in batches:
        obs, action, next_obs = b.obs, b.action, b.next_obs
        _obs.append(obs.to(args.device))
        _action.append(action.to(args.device))
        _next_obs.append(next_obs.to(args.device))
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
        _action = _action.cpu()
       


    os.makedirs(args.save_path, exist_ok=True)
    # Plot encoder space
 
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(z[:, 0], z[:, 1], z[:, 2], s=5, marker='o', color='b', label='z')
    action_names = {0: '-y', 1: '+y', 2: '-x', 3: '+x'}
    for a in range(4):
        next_z_a = next_z[_action==a]
        ax.scatter3D(next_z_a[:, 0], next_z_a[:, 1], next_z_a[:, 2], s=5, marker='^', label=f'next_z: action {action_names[a]}')   
    plt.legend() 
    plt.savefig(f'{args.save_path}/3D-z-space.png')

    ### ISOMAP
    print('Isomap...')
    try:
        n_neighbors = 10
        n_components = 2
        Y = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components).fit_transform(z)
        plt.figure()
        plt.scatter(Y[:, 0], Y[:, 1], s=5)
        plt.grid()
        plt.savefig(f'{args.save_path}/isomap-z-space.png')
    except:
        print('Isomap failed..')
    
    ## LLE
    print('LLE... (modified)')
    try:
        t = manifold.LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, method='modified', eigen_solver='dense').fit(z)
        Y = t.transform(z)
        _next_z = t.transform(predicted_z)
        plt.figure()
        plt.scatter(Y[:, 0], Y[:, 1], s=5)
        plt.scatter(_next_z[:, 0], _next_z[:, 1])
        plt.grid()
        plt.savefig(f'{args.save_path}/lle-z-space.png')
    except:
        print('LLE did not work')
   
    
    ##Local Tangent space alignment LLE 
    try:
        print('LLE... (LTSA)')
        Y = manifold.LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, method='ltsa', eigen_solver='dense').fit_transform(z)
        plt.figure()
        plt.scatter(Y[:, 0], Y[:, 1], s=5)
        plt.grid()
        plt.savefig(f'{args.save_path}/ltsa-lle-z-space.png')
    except:
        print('LLE... (LSTA) failed')
    
    ## Spectral
    print('Spectral Embedding...')
    try:
        Y = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors).fit_transform(z)
        plt.figure()
        plt.scatter(Y[:, 0], Y[:, 1], s=5)
        plt.grid()
        plt.savefig(f'{args.save_path}/spectral-z-space.png')
    except:
        print('Spectral embedding did not work')
        
   
    ### TSNE embedding
    try:
        print('TSNE...')
        tsne = manifold.TSNE(n_components=n_components, init='pca')
        Z = tsne.fit_transform(z)
        plt.figure()
        plt.scatter(Z[:, 0], Z[:, 1], s=5)
        plt.grid()
        plt.savefig(f'{args.save_path}/tsne-z-space.png')
    except:
        print('TSNE failed...')

    # ## Hessian LLE 
    try:
        print('LLE... (hessian)')
        nn = n_components * (n_components+3)/2 + 1
        Y = manifold.LocallyLinearEmbedding(n_components=n_components, n_neighbors=int(nn), method='hessian', eigen_solver='dense').fit_transform(z)
        plt.figure()
        plt.scatter(Y[:, 0], Y[:, 1], s=5)
        plt.grid()
        plt.savefig(f'{args.save_path}/hess-lle-z-space.png')
    except:
        print('LLE Hessian failed')
    
    ## MDS
    try:
        print('MDS...')
        Y = manifold.MDS(n_components=n_components, normalized_stress='auto').fit_transform(z)
        plt.figure()
        plt.scatter(Y[:, 0], Y[:, 1], s=5)
        plt.grid()
        plt.savefig(f'{args.save_path}/mds-z-space.png')
    except:
        print('MDS failed...')
    
    print('Done...')
