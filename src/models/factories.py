'''
    Model Factories
    author: Rafael Rodriguez-Sanchez
    date: February 2023
'''

import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from typing import Any, Dict, List, Optional, Tuple
from src.models.configs import DiagGaussianConfig, MLPConfig, DistributionConfig, ModuleConfig
from src.models import MLP, DiagonalGaussian, DynamicsMLP, RewardMLP, Deterministic
from src.models import ResidualConvEncoder, DeconvBlock, PixelCNNDecoder
from src.models.gaussian import SphericalGaussian

class ModuleFactory:
    factories ={
        "mlp": MLP,
        "diag_gaussian": DiagonalGaussian,
        "spherical_gaussian": SphericalGaussian,
        "dynamics_mlp": DynamicsMLP,
        "reward_mlp": RewardMLP,
        "deterministic": Deterministic,
        "conv_residual": ResidualConvEncoder,
        "pixelcnn": PixelCNNDecoder.PixelCNNDecoderDist,
        "deconv": DeconvBlock
    }
    
    @staticmethod
    def build(cfg: ModuleConfig):
        return ModuleFactory.factories[cfg.type](cfg)
    
    @staticmethod
    def register(type: str, factory: Any):
        ModuleFactory.factories[type] = factory

def build_distribution(cfg: DistributionConfig):
    features = ModuleFactory.build(cfg.features)
    dist = ModuleFactory.build(cfg.dist)
    return dist(features)

def build_model(cfg: ModuleConfig):
    return ModuleFactory.factories[cfg.type](cfg)