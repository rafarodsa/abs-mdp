'''
    Build Gym Environment from learned abstractions
    author: Rafael Rodriguez-Sanchez
    date: 22 June 2023
'''

import argparse
from src.absmdp.tpc_critic import InfoNCEAbstraction as Abstraction
from src.absmdp.discrete_tpc_critic import DiscreteInfoNCEAbstraction as DiscreteTPCAbstraction
from src.absmdp.mdp import AbstractMDPCritic as AbstractMDP
from src.absmdp.discrete_mdp import DiscreteAbstractMDP
from src.absmdp.datasets import PinballDataset
from src.absmdp.datasets_traj import PinballDatasetTrajectory
from src.absmdp.tpc_critic_rssm import RSSMAbstraction, RSSMAbstractModel

from omegaconf import OmegaConf as oc
from utils import get_experiment_info
import torch
import os
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str)
parser.add_argument('--discrete', action='store_true')
parser.add_argument('--rssm', action='store_true')
parser.add_argument('--dataset', type=str, default='')
args = parser.parse_args()

config_file, phi_ckpt_path = get_experiment_info(args.experiment)
experiment_name = os.path.basename(args.experiment)
print(f'Experiment name: {experiment_name}')
outdir = f'{args.experiment}'


print(f'Loading ckpt from {phi_ckpt_path}')
print(f'Loading config from {config_file}')

cfg = oc.load(config_file).cfg
if args.rssm:
    if '+modelsuccessrate' in experiment_name:
        model = RSSMAbstractModel.load_from_checkpoint(phi_ckpt_path)
    else:
        model = RSSMAbstraction.load_from_checkpoint(phi_ckpt_path)
else:  
    model = RSSMAbstraction.load_from_checkpoint(phi_ckpt_path) 
    # model = Abstraction.load_from_checkpoint(phi_ckpt_path) if not args.discrete else DiscreteTPCAbstraction.load_from_checkpoint(phi_ckpt_path)


if len(args.dataset) > 0:
    cfg.data.data_path = args.dataset

print(f'Loading dataset {cfg.data.data_path}')
# data = PinballDataset(cfg.data) if not args.rssm else PinballDatasetTrajectory(cfg.data)
data = PinballDatasetTrajectory(cfg.data)
data.setup()


mdp = AbstractMDP.load(model, data.dataset, rssm=args.rssm) if not args.discrete else DiscreteAbstractMDP.load(model, data.dataset)
# save_path = f'{args.save_path}/mdp.pt' if args.save_path is not None else f'{args.dir}/mdp.pt'
torch.save(mdp, f'{outdir}/{"disc" if args.discrete else "cont"}_mdp.pt')

