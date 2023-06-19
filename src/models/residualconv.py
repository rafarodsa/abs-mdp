import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.mlp import MLP

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
        # self.norm_1 = nn.LayerNorm([cfg.feat_maps, cfg.in_width, cfg.in_height])
        
        # self.conv_2 = nn.Conv2d(cfg.feat_maps, cfg.feat_maps, kernel_size=cfg.kernel_size, stride=2)
        # self.norm_2 = nn.LayerNorm([cfg.feat_maps, cfg.in_width, cfg.in_height])
        # self.max_pool_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        
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