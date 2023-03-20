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
from PIL import Image
from tqdm import tqdm

def img_preprocess(x):
    img = x.byte()
    if len(img.size()) > 3:
        # batched
        img = img.permute(0, 2, 3, 1)
    else:
        img = img.permute(1,2,0)
    return img.squeeze(-1)

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-ckpt', type=str)
    parser.add_argument('--n-samples', type=int, default=2)
    parser.add_argument('--config', type=str, default='experiments/pb_obstacles/fullstate/config/config.yaml')
    parser.add_argument('--save-path', type=str, default='.')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Load config
    cfg = oc.load(args.config)

    # Load
    model = AbstractMDPTrainer.load_from_checkpoint(args.from_ckpt, cfg=cfg)
    model.to(args.device)

    data = PinballDataset(cfg.data)
    data.setup()
    # data.to('cuda')

    batches = list(data.test_dataloader())

    # _obs, _action, _next_obs = [], [], []
    # for b in batches[0:1]:
    #     obs, action, next_obs = b.obs, b.action, b.next_obs
    #     _obs.append(obs.to(args.device))
    #     _action.append(action.to(args.device))
    #     _next_obs.append(next_obs.to(args.device))
    # obs, action, next_obs = torch.cat(_obs, dim=0), torch.cat(_action, dim=0), torch.cat(_next_obs, dim=0)


    # Batch = namedtuple('batch', ['obs', 'action', 'next_obs'])
    # batch = Batch(obs, action, next_obs)
    # n_samples = args.n_samples

    print('Test set loaded...')
    _z, _next_z, _predicted_z, _next_s_q = [], [], [], []
    actions = []
    log_probs = []
    for batch in tqdm(batches):
        with torch.no_grad():
            actions.append(batch.action)
            z = torch.tanh(model.encoder(batch.obs.to(args.device)))
            _z.append(z)
            next_obs = batch.next_obs.to(args.device)
            next_z = torch.tanh(model.encoder(next_obs))
            _next_z.append(next_z)

            transition_in = torch.cat((z, batch.action.to(args.device)), dim=-1)
            predicted_z, q_z, _ = model.transition.sample_n_dist(transition_in, 1)
            predicted_z = predicted_z.squeeze() + z
            _predicted_z.append(predicted_z)
            next_zs = torch.cat([predicted_z, next_z], dim=0)

            pred_next_s_q = model.grounding.distribution(predicted_z)
            _next_s_q.append(pred_next_s_q)

            log_probs.append(pred_next_s_q.log_prob(next_obs).sum())
            

            # printarr(batch.obs, batch.next_obs, z, next_z, predicted_z, predicted_next_s, decoded_next_s)



    # 
    # Generate Images.
    n_imgs=2
    next_s_q = _next_s_q[0]
    next_s_real = img_preprocess(batches[0].next_obs * 255)
    
    samples = model.grounding.distribution(_predicted_z[0][:n_imgs]).sample(n_samples=args.n_samples).mean(0)
    img = img_preprocess(samples).cpu().numpy()
    printarr(img, next_s_real)
    plt.figure()
    for i in range(n_imgs):
        plt.subplot(n_imgs, 2, i*n_imgs+1)
        plt.imshow(img[i])
        plt.subplot(n_imgs, 2, i*n_imgs+2)
        plt.imshow(next_s_real[i])
    plt.savefig(f'{args.save_path}/samples.png')
    ####################


    z = torch.cat(_z, dim=0)
    next_z = torch.cat(_next_z, dim=0)
    predicted_z = torch.cat(_predicted_z, dim=0)
    actions = torch.cat(actions, dim=0)
    print(f'Mean NLL: {-sum(log_probs) / z.shape[0]}')

    if args.device == 'cuda':
       z = z.cpu()
       next_z = next_z.cpu()
       predicted_z = predicted_z.cpu()


    latent_space = TSNE(2)
    latent_space = latent_space.fit_transform(torch.cat((z, next_z, predicted_z), dim=0))
    z, next_z, predicted_z = latent_space[:z.shape[0]], latent_space[z.shape[0]:z.shape[0]+next_z.shape[0]], latent_space[z.shape[0]+next_z.shape[0]:]
    printarr(z, next_z, predicted_z)
    # Plot encoder space
    _action = actions.argmax(-1).cpu()
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
        plt.scatter(z[_action == a, 0], z[_action == a, 1], s=5, label=f'action {a}', color='b')
        plt.scatter(next_z[_action == a, 0], next_z[_action == a, 1], s=5, label=f'action {a}', color='r')

    plt.subplot(1, 2, 2)
    for a in acts:
        plt.title('Predicted transition')
        plt.scatter(z[_action == a, 0], z[_action == a, 1], s=5, label=f'action {a}', color='b')
        plt.scatter(predicted_z[_action == a, 0], predicted_z[_action == a, 1], s=5, label=f'action {a}', color='r')

    plt.savefig(f'{args.save_path}/latent_space.png')

#### 


