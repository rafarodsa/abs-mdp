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


from sklearn.decomposition import PCA

from collections import namedtuple
import os

import lightning as pl

from matplotlib import cm
from src.utils.printarr import printarr

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-ckpt', type=str)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--config', type=str, default='experiments/pb_obstacles/fullstate/config/config.yaml')
    parser.add_argument('--save-path', type=str, default='.')
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
        model.to('cpu')
        data = PinballDatasetTrajectory(cfg.data)

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

    n_samples = 5
    obs = batch.obs.reshape(-1, batch.obs.shape[-1])
    goals = obs[torch.randint(0, batch.obs.shape[0], (n_samples,))]

    with torch.no_grad():

        model.eval()
        z_q = model.encoder(batch.obs)
        z_q = z_q.reshape(-1, z_q.shape[-1])
        encoded_goals = model.encoder(goals)
        energy_goals = torch.tanh(model.grounding(goals, encoded_goals))
        energy = torch.tanh(model.grounding(goals.repeat(z_q.shape[0], 1), z_q.repeat_interleave(n_samples, dim=0))).reshape(z_q.shape[0], n_samples).max(-1).values
        energy = energy.reshape(n_samples, -1)
        for i in range(n_samples):
            energy[i, energy[i] < energy_goals[i] - 0.01] = 0
        printarr(energy, z_q, goals)
        energy = energy.sum(0)
   
    # Plot heatmap
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.scatter(obs[:, 0], obs[:, 1], c=energy, cmap=cm.coolwarm, s=5)
    plt.colorbar()
    plt.scatter(goals[:, 0], goals[:, 1], c='k', s=1)
    # ax.plot_surface(z_q[:, 0], z_q[:, 1], energy.unsqueeze(-1), antialiased=False, linewidth=0)
    
    plt.savefig(os.path.join(args.save_path, 'heatmap.png'))