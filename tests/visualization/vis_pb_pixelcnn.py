'''
    Visualize 2D latent space
'''


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import torch 

import argparse

from src.absmdp.infomax import InfomaxAbstraction as AbstractMDPTrainer
from src.absmdp.datasets import PinballDataset

from omegaconf import OmegaConf as oc


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from collections import namedtuple
from src.utils.printarr import printarr
    

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-ckpt', type=str)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--config', type=str, default='experiments/pb_obstacles/fullstate/config/config.yaml')
    parser.add_argument('--save-path', type=str, default='.')
    args = parser.parse_args()

    # Load config
    cfg = oc.load(args.config)

    # Load
    model = AbstractMDPTrainer.load_from_checkpoint(args.from_ckpt, cfg=cfg)
    

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
    n_samples = 5

    with torch.no_grad():
        z = model.encoder(batch.obs)
        next_z = model.encoder(batch.next_obs)

        transition_in = torch.cat((z, batch.action), dim=-1)
        predicted_z, q_z, _ = model.transition.sample_n_dist(transition_in, 1)
        predicted_z = q_z.mean + z
        predicted_next_s_q = model.grounding.distribution(predicted_z)
        predicted_next_s = predicted_next_s_q.sample(n_samples).mean(0)
        decoded_next_s_q = model.grounding.distribution(next_z)
        decoded_next_s = decoded_next_s_q.sample(n_samples).mean(0)

        print(f'Empirical MSE {(predicted_next_s - batch.next_obs).pow(2).sum(-1).mean().sqrt()}')
        print(f'Test NLL {predicted_next_s_q.log_prob(batch.next_obs).mean()}')
        printarr(batch.obs, batch.next_obs, z, next_z, predicted_z, decoded_next_s_q.mean, predicted_next_s, decoded_next_s)
        

        encoded_transition = next_z - z
        predicted_transition = predicted_z - z
        


    latent_space = TSNE(2)
    latent_space = latent_space.fit_transform(torch.cat((z, next_z, predicted_z), dim=0))
    z, next_z, predicted_z = latent_space[:z.shape[0]], latent_space[z.shape[0]:z.shape[0]+next_z.shape[0]], latent_space[z.shape[0]+next_z.shape[0]:]

    # Plot encoder space
    _action = batch.action.argmax(-1)
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1], s=5, marker='o', color='b')
    ACTIONS = ['+Y', '-Y', '+X', '-X']
    for a in range(4):
        next_z_a = next_z[_action==a]
        plt.scatter(next_z_a[:, 0], next_z_a[:, 1], s=5, marker='^', label=f'action {ACTIONS[a]}')   
    plt.legend() 
    plt.savefig(f'{args.save_path}/z-space.png')
    
    # Plot initial states
    acts = [1, 3]
    plt.figure()
    plt.subplot(1, 2, 1)

    for a in acts:
        plt.title('Encoded transition')
        # plt.quiver(z[_action==a, 0], z[_action==a, 1], encoded_transition[_action==a, 0], encoded_transition[_action==a, 1], angles='xy', scale_units='xy', scale=1)
        plt.scatter(z[_action == a, 0], z[_action == a, 1], s=5, label=f'action {a}', color='b')
        plt.scatter(next_z[_action == a, 0], next_z[_action == a, 1], s=5, label=f'action {a}', color='r')

    plt.subplot(1, 2, 2)
    for a in acts:
        # plt.quiver(z[_action==a, 0], z[_action==a, 1], predicted_transition[_action==a, 0], predicted_transition[_action==a, 1], angles='xy', scale_units='xy', scale=1)
        plt.title('Predicted transition')
        plt.scatter(z[_action == a, 0], z[_action == a, 1], s=5, label=f'action {a}', color='b')
        plt.scatter(predicted_z[_action == a, 0], predicted_z[_action == a, 1], s=5, label=f'action {a}', color='r')

    plt.savefig(f'{args.save_path}/latent_space.png')


    
    # pca = PCA(2)
    # pca = pca.fit(batch.obs)
    # s = pca.transform(batch.obs)
    # next_s = pca.transform(batch.next_obs)
    # pred_next_s = pca.transform(predicted_next_s)
    # encoded_next_s = pca.transform(decoded_next_s.squeeze())
   
    transform = np.linalg.pinv(data.linear_transform)
    s = np.einsum('ji, bj->bi', transform, batch.obs.numpy())
    next_s, pred_next_s = np.einsum('ji, bj->bi',transform ,batch.next_obs.numpy()), np.einsum('ji, bj->bi', transform, predicted_next_s.numpy())
    encoded_next_s = np.einsum('ji, bj->bi',transform , decoded_next_s.squeeze().numpy())
    
    d_real = next_s - s
    d_pred = pred_next_s - s

    plt.figure()
    plt.subplot(1,3,1)
    for a in acts:
        plt.title('Truth')
        plt.scatter(s[_action == a, 0], s[_action == a, 1], marker='x', s=5, color='b')
        plt.scatter(next_s[_action==a, 0], next_s[_action == a, 1], marker='x', s=5, color='r')
        # plt.quiver(s[_action == a, 0], s[_action == a, 1], d_real[_action == a, 0], d_real[_action == a, 1], angles='xy', scale_units='xy', scale=1)
    # plt.quiver(s[:, 0], s[:, 1], d_pred[:, 0], d_pred[:, 1])
    plt.subplot(1,3,2)
    for a in acts: 
        plt.title('Predicted')
        plt.scatter(s[_action == a, 0], s[_action == a, 1], marker='x', s=5, color='b')
        plt.scatter(pred_next_s[_action == a, 0], pred_next_s[_action == a, 1], marker='o', s=5, color='g')
        # plt.quiver(s[_action == a, 0], s[_action == a, 1], d_pred[_action == a, 0], d_pred[_action == a, 1], angles='xy', scale_units='xy', scale=1)
    
    plt.subplot(1,3,3)
    for a in acts: 
        plt.title('Encoded/Decoded')
        plt.scatter(s[_action == a, 0], s[_action == a, 1], marker='x', s=5, color='b')
        plt.scatter(encoded_next_s[_action == a, 0], encoded_next_s[_action == a, 1], marker='o', s=5, color='g')
    plt.savefig(f'{args.save_path}/pca_s.png')

