'''
    Build Gym Environment from learned abstractions
    author: Rafael Rodriguez-Sanchez
    date: 22 June 2023
'''

import argparse
from src.absmdp.tpc_critic import InfoNCEAbstraction as Abstraction
from src.absmdp.tpc_critic_rssm import RSSMAbstraction, RSSMAbstractModel
from src.absmdp.discrete_tpc_critic import DiscreteInfoNCEAbstraction as DiscreteTPCAbstraction
from src.absmdp.mdp import AbstractMDPCritic as AbstractMDP
from src.absmdp.discrete_mdp import DiscreteAbstractMDP
from src.absmdp.datasets import PinballDataset
from src.absmdp.datasets_traj import PinballDatasetTrajectory
from omegaconf import OmegaConf as oc

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='.')
parser.add_argument('--phi-ckpt', type=str)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--discrete', action='store_true')
parser.add_argument('--rssm', action='store_true')
parser.add_argument('--suc', action='store_true')
args = parser.parse_args()

phi_ckpt_path = f'{args.phi_ckpt}'

if not args.rssm:
    model = Abstraction.load_from_checkpoint(phi_ckpt_path) if not args.discrete else DiscreteTPCAbstraction.load_from_checkpoint(phi_ckpt_path)
else:
    if args.suc:
        model = RSSMAbstractModel.load_from_checkpoint(phi_ckpt_path)
    else:
        model = RSSMAbstraction.load_from_checkpoint(phi_ckpt_path)


model.cfg.data.non_linear_transform = False
data = PinballDataset(model.cfg.data) if not args.rssm else PinballDatasetTrajectory(model.cfg.data)
data.setup()

mdp = AbstractMDP.load(model, data.dataset, rssm=args.rssm) if not args.discrete else DiscreteAbstractMDP.load(model, data.dataset)
# save_path = f'{args.save_path}/mdp.pt' if args.save_path is not None else f'{args.dir}/mdp.pt'
torch.save(mdp, f'{args.dir}/{"disc" if args.discrete else "cont"}_mdp.pt')

