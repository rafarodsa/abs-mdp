from .gaussian import DiagonalGaussian, Deterministic, DiagonalGaussianModule
from .mlp import MLP, DynamicsMLP, RewardMLP
from .pixelcnn import PixelCNNStack, GatedPixelCNNLayer, PixelCNNDecoder, DeconvBlock
from .residualconv import ConvResidualLayer, ResidualStack, ResidualConvEncoder
from .configs import DistributionConfig, ModuleConfig
from .grid_quantizer import FactoredQuantizer, FactoredQuantizerSTFactory, FactoredCategoricalModuleFactory

from .factories import ModuleFactory


ModuleFactory.register('quantizer_st', FactoredQuantizerSTFactory)
ModuleFactory.register('factored_categorical', FactoredCategoricalModuleFactory)