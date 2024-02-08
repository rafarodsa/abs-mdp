import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.mlp import MLP
import numpy as np

from src.utils.printarr import printarr

##  Pixel Encoder Models
class ConvResidualLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv_1 = nn.Conv2d(cfg.in_channels, cfg.out_channels, kernel_size=cfg.kernel_size, stride=1, padding='same')
        self.norm_1 = nn.LayerNorm([cfg.out_channels, cfg.in_width, cfg.in_height])
        self.conv_2 = nn.Conv2d(cfg.out_channels,  cfg.in_channels, kernel_size=1, stride=1, padding='same')
        self.norm_2 = nn.LayerNorm([cfg.in_channels, cfg.in_width, cfg.in_height])
        self.relu = nn.ReLU()

    def forward(self, x):
        conv_1 = self.relu(self.norm_1(self.conv_1(x)))
        conv_2 = self.relu(self.norm_2(self.conv_2(conv_1)))
        return conv_2 + x

class ResidualStack(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([ConvResidualLayer(cfg) for _ in range(cfg.n_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ResidualConvEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv_1 = nn.Conv2d(cfg.color_channels, cfg.feat_maps, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        n = 0
        cfg.residual.in_width, cfg.residual.in_height = self._compute_out_size(cfg.in_width, cfg.kernel_size, n=n), self._compute_out_size(cfg.in_height, cfg.kernel_size, n=n)
        self.residual_stack = ResidualStack(cfg.residual)
        self.max_pool_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1) # 2x2 max pooling
        hidden_dim = 256
        in_width, in_height = cfg.residual.in_width // 2 + 1, cfg.residual.in_height // 2 + 1
        in_dim = cfg.residual.in_channels * in_width * in_height
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim, cfg.out_dim)


    def forward(self, x):
        x = self.relu(self.conv_1(x))
        # x = self.relu(self.conv_2(x))

        x = self.max_pool_1(self.residual_stack(x))
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = self.linear_1(F.relu(x))
        return x
    
    def _compute_out_size(self, x, kernel_size, n=1):
        for _ in range(n):
            x = (x - kernel_size) // 2 + 1
        return x
    

class ConvCritic(nn.Module):
    def __init__(self, cnn_feats, mlp):
        super().__init__()
        self.conv_feats = cnn_feats
        self.mlp = mlp
    def forward(self, x, z):
        x = self.conv_feats(x)
        x = torch.cat([x, z], dim=1)
        return self.mlp(x)
    
def build_conv_critic(cfg):
    cnn_feats = ResidualConvEncoder(cfg.cnn)
    mlp = MLP(cfg.mlp)
    return ConvCritic(cnn_feats, mlp)


##### Doubling Depth Residual ConvNet

ACTIVATION_LAYERS = {
    'relu' : nn.ReLU,
    'silu' : nn.SiLU,
    'tanh' : nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'elu': nn.ELU,
    'mish': nn.Mish
}

def get_activation_layer(activation : str):
    if activation in ACTIVATION_LAYERS:
        return ACTIVATION_LAYERS[activation]
    else:
        raise ValueError(f'Unknown activation layer: {activation}')

class _ResidualBlock(nn.Module):
    def __init__(self, input_shape, out_channels, cnn_blocks=2, activation='silu'):
        super().__init__()
        self.input_shape = input_shape
        in_channels, width, height = input_shape
        layers = []
        self.pre_conv = nn.Sequential(
                                       nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2),
                                       nn.LayerNorm([out_channels, int((width-4) / 2 + 1), int((height-4) / 2 + 1)]),
                                       get_activation_layer(activation)()
                                    )
        for i in range(cnn_blocks): 
            layers.append(nn.LayerNorm([out_channels, int((width-4) / 2 + 1), int((height-4) / 2 + 1)]))
            layers.append(get_activation_layer(activation)())
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.pre_conv(x)
        _x = x
        for layer in self.layers:
            _x = layer(_x)
        return x + _x




class ResidualEncoder(nn.Module):
    def __init__(self, input_shape=(1,40,40), cnn_blocks=2, depth=24, min_resolution=4, cnn_activation='silu', mlp_layers = [], mlp_activation='silu'):
        super().__init__()
        self.input_shape = input_shape
        self.cnn_blocks = cnn_blocks
        self.depth = depth
        self.min_resolution = min_resolution
        self.activation = cnn_activation

        # build network.
        n_res_layers = int(np.log2(self.input_shape[-1]) - np.log2(min_resolution))
        print(f'Building Residual Encoder with {n_res_layers} layers.')
        layers = []
        for i in range(n_res_layers):
            layers.append(_ResidualBlock(input_shape, depth, cnn_blocks, cnn_activation))
            input_shape = (depth, int((input_shape[1]-4) / 2 + 1),int((input_shape[1]-4) / 2 + 1))
            depth *= 2

        self.layers = nn.ModuleList(layers)
        self.out_dim = input_shape[0] * input_shape[1] * input_shape[2]

        mlp_layers = [self.out_dim] + mlp_layers
        mlp_layers = list(zip(mlp_layers[:-1], mlp_layers[1:]))
        layers = []
        self.outdim = self.out_dim
        for (in_dim, out_dim) in mlp_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(get_activation_layer(mlp_activation)())
            self.outdim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        batch_dims, input_shape = x.shape[:-3], x.shape[-3:]
        x = x.reshape(-1, *input_shape)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)
        return x.reshape(*batch_dims, -1)
    

class ResidualCritic(nn.Module):
    def __init__(self, cnn_params, mlp_params):
        super().__init__()
        self.cnn = ResidualEncoder(**cnn_params)
        mlp_layers = [self.cnn.outdim + mlp_params['latent_dim'], ] + mlp_params['mlp_layers'] + [1,]
        layers = []
        for in_dim, out_dim in zip(mlp_layers[:-1], mlp_layers[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(get_activation_layer(mlp_params['activation'])())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, z):
        x = self.cnn(x)
        return self.mlp(torch.cat([x, z], dim=-1))

def build_residual_encoder(cfg):
    mlp_layers = cfg.mlp_layers + [cfg.outdim]
    return ResidualEncoder((cfg.color_channels, cfg.in_width, cfg.in_height), cfg.cnn_blocks, cfg.depth, cfg.min_resolution, cfg.cnn_activation, mlp_layers=mlp_layers, mlp_activation=cfg.mlp_activation)

def build_residual_critic(cfg):
    cnn = cfg.cnn
    cnn_params = {
        'input_shape': (cnn.color_channels, cnn.in_width, cnn.in_height),
        'cnn_blocks': cnn.cnn_blocks,
        'depth': cnn.depth,
        'min_resolution': cnn.min_resolution,
        'mlp_layers': cnn.mlp_layers + [cnn.outdim],
        'cnn_activation': cnn.cnn_activation,
        'mlp_activation': cnn.mlp_activation
    }

    mlp = cfg.mlp
    mlp_params = {
        'latent_dim': mlp.latent_dim,
        'mlp_layers': mlp.hidden_dims,
        'activation': mlp.activation
    }

    return ResidualCritic(cnn_params, mlp_params)