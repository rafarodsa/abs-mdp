from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import OmegaConf as oc, MISSING


def load_module_params(path, *, _parent_):
    try:
        with open(path, 'r') as f:
            cfg = oc.structured(oc.load(f))
        cfg._set_parent(_parent_)
        return cfg
    except FileNotFoundError:
        raise ValueError(f'Could not load config file {path}')

# Register resolver to import configs within YAML files
try:
    oc.register_new_resolver('loadcfg', load_module_params)
except ValueError:
    pass

# Structured configs 

@dataclass
class ModuleConfig:
    type: Optional[str]

@dataclass
class MLPConfig(ModuleConfig):
    type: str = "mlp"
    input_dim: int = MISSING
    output_dim: int = MISSING
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"

@dataclass
class DiagGaussianConfig(ModuleConfig):
    type: str = "diag_gaussian"
    output_dim: int = MISSING
    min_var: float = 1e-6
    max_var: float = 1e6

@dataclass
class DistributionConfig:
    features: Any = MISSING
    dist: Any = MISSING

class ConfigFactory:
    factories = {
        "mlp": MLPConfig,
        "diag_gaussian": DiagGaussianConfig
    }
    @staticmethod
    def create(type: str):
        return ConfigFactory.factories[type]