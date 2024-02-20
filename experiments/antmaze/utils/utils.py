import numpy as np
from dataclasses import dataclass
from omegaconf import OmegaConf as oc

@dataclass
class OptionExecution:
    s: np.ndarray
    next_state: np.ndarray
    goal: np.ndarray
    success: bool
    reward: np.ndarray
    steps: int

def parse_oc_args(oc_args):
    assert len(oc_args)%2==0
    oc_args = ['='.join([oc_args[i].split('--')[-1], oc_args[i+1]]) for i in range(len(oc_args)) if i%2==0]
    cli_config = oc.from_cli(oc_args)
    return cli_config

def parse_and_merge(parser):
    args, unknown = parser.parse_known_args()
    cli_args = parse_oc_args(unknown)
    cfg = oc.load(args.config)
    cfg = oc.merge(cfg, cli_args)
    return args, cfg