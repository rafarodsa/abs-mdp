import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from src.absmdp.tpc_critic_rssm import RSSMAbstraction
from src.absmdp.datasets_traj import PinballDatasetTrajectory
import argparse

from sklearn import manifold

from collections import namedtuple
from scripts.utils import get_experiment_info, prepare_outdir
from src.utils import printarr
import torch
from omegaconf import OmegaConf as oc


#   Assuming obs and z are Numpy arrays, and features are along the columns
    # Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str)
parser.add_argument('--n-samples', type=int, default=1000)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--rssm', action='store_true')
parser.add_argument('--dataset', type=str, default='')

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

model = RSSMAbstraction.load_from_checkpoint(ckpt_path)
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


obs_energy = np.abs(obs[..., 0] + obs[..., 1]) # norm 1
printarr(obs_energy)

Batch = namedtuple('batch', ['obs', 'action', 'next_obs'])
batch = Batch(obs, action, next_obs)

with torch.no_grad():
    model.eval()
    z = model.encoder(batch.obs)


z = z.reshape(-1, z.shape[-1])
obs_energy = obs_energy.reshape(-1)
obs = obs.reshape(-1, obs.shape[-1])

if args.device != 'cpu':
    z = z.cpu()
       
# Prepare an empty matrix to hold mutual information values
mutual_info_matrix = np.empty((obs.shape[1], z.shape[1]))

# Compute mutual information for each pair of features
for i in range(obs.shape[1]):
    for j in range(z.shape[1]):
        mutual_info_matrix[i, j] = mutual_info_regression(obs[:, i].reshape(-1, 1), z[:, j])[0]

# Plot heatmap using Seaborn
plt.figure(figsize=(10, 8))
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(mutual_info_matrix, annot=False, 
            xticklabels=[f"z_{i}" for i in range(z.shape[1])], 
            yticklabels=[f"obs_{i}" for i in range(obs.shape[1])],
            #robust=True,
            #cma=cmap,
            #square=True
        )
plt.xlabel("Encoded Obs Features (z)")
plt.ylabel("Obs Features")
plt.title("Mutual Information between z and obs Features")
plt.savefig(f'{save_path}/mutual_info.png')
