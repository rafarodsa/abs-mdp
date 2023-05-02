'''
    Build Gym Environment from learned abstractions
    author: Rafael Rodriguez-Sanchez
    date: 1 May 2023
'''

import argparse
from src.absmdp.infomax_attn import AbstractMDPTrainer, InfomaxAbstraction
from src.absmdp.mdp import AbstractMDP
from src.absmdp.datasets import PinballDataset
from omegaconf import OmegaConf as oc

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
parser.add_argument('--phi-ckpt', type=str)
parser.add_argument('--mdp-ckpt', type=str)
parser.add_argument('--save-path', type=str, default=None)
args = parser.parse_args()

phi_ckpt_path = f'{args.phi_ckpt}'
mdp_ckpt_path = f'{args.mdp_ckpt}'

phi = InfomaxAbstraction.load_from_checkpoint(phi_ckpt_path)
model = AbstractMDPTrainer.load_from_checkpoint(mdp_ckpt_path, phi=phi, cfg=phi.cfg)
data = PinballDataset(phi.cfg.data)
data.setup()

mdp = AbstractMDP.load(model, data.dataset)
# save_path = f'{args.save_path}/mdp.pt' if args.save_path is not None else f'{args.dir}/mdp.pt'
torch.save(mdp, f'{args.dir}/mdp.pt')

