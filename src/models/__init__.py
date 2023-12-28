from .gaussian import DiagonalGaussian, Deterministic, DiagonalGaussianModule
from .gaussian import build_gaussian_mixture
from .mlp import MLP, DynamicsMLP, RewardMLP, MLPCritic
from .pixelcnn import PixelCNNStack, GatedPixelCNNLayer, PixelCNNDecoder, DeconvBlock
from .residualconv import ConvResidualLayer, ResidualStack, ResidualConvEncoder, build_conv_critic, build_residual_critic, build_residual_encoder
from .configs import DistributionConfig, ModuleConfig
from .grid_quantizer import FactoredQuantizer, FactoredQuantizerSTFactory, FactoredCategoricalModuleFactory
from .rssm import RSSMFactory
from .multiencoder import build_multiencoder
from .factories import ModuleFactory


ModuleFactory.register('quantizer_st', FactoredQuantizerSTFactory)
ModuleFactory.register('factored_categorical', FactoredCategoricalModuleFactory)
ModuleFactory.register('conv_residual_critic', build_conv_critic)
ModuleFactory.register('mlp_critic', MLPCritic)
ModuleFactory.register('rssm', RSSMFactory)
ModuleFactory.register('cnn_critic', build_residual_critic)
ModuleFactory.register('doubling_residual', build_residual_encoder)
ModuleFactory.register('mixture_gaussian', build_gaussian_mixture)
ModuleFactory.register('multiencoder', build_multiencoder)