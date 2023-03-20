from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import OmegaConf, MISSING, DictConfig

from src.models.configs import MLPConfig, DistributionConfig

# Register a resolver with OmegaConf to handle math within .yaml files
#
# example.yaml:
# ```
# one: 1
# two: 2
# three: "${eval: ${one} + ${two}}"  # => 3
# ```
OmegaConf.register_new_resolver("eval", eval, replace=True)


@dataclass
class VAELossConfig:
    grounding_const: float = 10
    kl_const: float = 0.1
    reward_const: float = 1.
    transition_const: float = 0.1
    initset_const: float = 1.
    tau_const: float = 1.
    n_samples: int = 1
    kl_balance: float = 0.01

@dataclass
class DataConfig:
    data_path: str = MISSING
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    obs_type: str = 'full'
    n_reward_samples: int = 5
    gamma: float = 0.99
    transforms: Tuple[str] = field(default_factory=tuple)
    n_options: int = MISSING

@dataclass
class AbstractMDPConfig:
    #TODO allow to load from path into structure config
    encoder: Optional[Any]
    decoder: Optional[Any]
    transition: Optional[Any]
    reward: Optional[Any]
    init_class: Optional[Any]
    tau: Optional[Any]
    n_options: int = MISSING
    obs_dims: int = MISSING
    latent_dim: int = MISSING 

@dataclass
class TrainerConfig:
    experiment_cwd: Optional[str] = field(default='')
    lr: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    accelerator: str = "cpu"
    devices: int = 1
    save_path: str = "mdps"
    loss: VAELossConfig = MISSING
    model: AbstractMDPConfig = MISSING
    data: Any = MISSING


@dataclass
class ExperimentConfig:
    name: str = MISSING
    trainer: TrainerConfig = TrainerConfig()
    seed: int = 0
