import torch.nn as nn


##  Pixel Encoder Models
class ConvResidualLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv_1 = nn.Conv2d(cfg.in_channels, cfg.out_channels, kernel_size=cfg.kernel_size, stride=1, padding='same')
        self.norm_1 = nn.LayerNorm([cfg.out_channels, cfg.in_width, cfg.in_height])
        self.conv_2 = nn.Conv2d(cfg.out_channels,  cfg.in_channels, kernel_size=cfg.kernel_size, stride=1, padding='same')
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
        self.conv_1 = nn.Conv2d(cfg.color_channels, cfg.feat_maps, kernel_size=cfg.kernel_size, stride=1, padding='same')
        self.norm_1 = nn.LayerNorm([cfg.feat_maps, cfg.in_width, cfg.in_height])
        self.max_pool_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1) # 2x2 max pooling
        self.conv_2 = nn.Conv2d(cfg.feat_maps, cfg.feat_maps, kernel_size=cfg.kernel_size, stride=1, padding='same')
        self.norm_2 = nn.LayerNorm([cfg.feat_maps, cfg.in_width, cfg.in_height])
        self.max_pool_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        
        self.relu = nn.ReLU()
        cfg.residual.in_width, cfg.residual.in_height = cfg.in_width // 4 + 1, cfg.in_height // 4 + 1
        self.residual_stack = ResidualStack(cfg.residual)
        self.linear = nn.Linear(cfg.residual.in_channels * cfg.residual.in_width * cfg.residual.in_height, cfg.out_dim)

    def forward(self, x):
        
        # x = self.relu(self.norm_1(self.conv_1(x)))
        x = self.relu(self.conv_1(x))
        x = self.max_pool_1(x)
        x = self.relu(self.conv_2(x))
        x = self.max_pool_2(x)

        x = self.residual_stack(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x