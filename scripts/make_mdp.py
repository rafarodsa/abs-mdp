'''
    Build Gym Environment from learned abstractions
    author: Rafael Rodriguez-Sanchez
    date: 22 June 2023
'''

import argparse
from src.absmdp.tpc_critic import InfoNCEAbstraction as Abstraction
from src.absmdp.mdp import AbstractMDPCritic as AbstractMDP
from src.absmdp.datasets import PinballDataset
from omegaconf import OmegaConf as oc

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='.')
parser.add_argument('--phi-ckpt', type=str)
parser.add_argument('--save-path', type=str, default=None)
args = parser.parse_args()

phi_ckpt_path = f'{args.phi_ckpt}'

model = Abstraction.load_from_checkpoint(phi_ckpt_path)
data = PinballDataset(model.cfg.data)
data.setup()

mdp = AbstractMDP.load(model, data.dataset)
# save_path = f'{args.save_path}/mdp.pt' if args.save_path is not None else f'{args.dir}/mdp.pt'
torch.save(mdp, f'{args.dir}/cont_mdp.pt')

