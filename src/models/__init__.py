from .gaussian import DiagonalGaussian, Deterministic, DiagonalGaussianModule
from .mlp import MLP, DynamicsMLP, RewardMLP, MLPCritic
from .pixelcnn import PixelCNNStack, GatedPixelCNNLayer, PixelCNNDecoder, DeconvBlock
from .residualconv import ConvResidualLayer, ResidualStack, ResidualConvEncoder, build_conv_critic
from .configs import DistributionConfig, ModuleConfig
from .grid_quantizer import FactoredQuantizer, FactoredQuantizerSTFactory, FactoredCategoricalModuleFactory

from .factories import ModuleFactory


ModuleFactory.register('quantizer_st', FactoredQuantizerSTFactory)
ModuleFactory.register('factored_categorical', FactoredCategoricalModuleFactory)
ModuleFactory.register('conv_residual_critic', build_conv_critic)
ModuleFactory.register('mlp_critic', MLPCritic)