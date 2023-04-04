'''
    PixelSNAIL model
    author: Rafael Rodriguez-Sanchez
    date: 28 March 2023

    Inspired by the following implementations:
        https://github.com/rosinality/vq-vae-2-pytorch/
        https://github.com/neocxi/pixelsnail-public
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import lightning as L

from src.models.quantized_logistic import QuantizedLogisticMixture
from src.utils.printarr import printarr


def down_shift(x):
    return F.pad(x, (0,0,1,0))[:,:,:-1,:]

def right_shift(x):
    return F.pad(x, (1,0))[:,:,:,:-1]

def concat_elu(x):
    return F.elu(torch.cat([x, -x], dim=1))

class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.weight_norm(self)

class DownShiftedConv2d(Conv2d):
    def forward(self, x):
        # pad H above and W on each side
        Hk, Wk = self.kernel_size
        x = F.pad(x, ((Wk-1)//2, (Wk-1)//2, Hk-1, 0))
        return super().forward(x)

class DownRightShiftedConv2d(Conv2d):
    def forward(self, x):
        # pad above and on left (ie shift input down and right)
        Hk, Wk = self.kernel_size
        x = F.pad(x, (Wk-1, 0, Hk-1, 0))
        return super().forward(x)

class GatedResidualBlock(nn.Module):
    def __init__(self, n_channels=128, kernel_size=1, conv_fn=DownShiftedConv2d, activation='concat_elu', dropout=0.5):
        super().__init__()
        if activation == 'concat_elu':
            self.activation = concat_elu
            factor = 2
        elif activation == 'elu':
            self.activation = nn.ELU()
            factor = 1
        elif activation == 'relu':
            self.activation = nn.ReLU()
            factor = 1
        
        self.causalconv = conv_fn(factor*n_channels, n_channels, kernel_size, stride=1)
        self.shortcut = Conv2d(n_channels, factor*n_channels, kernel_size=1, stride=1)
        self.conditional = Conv2d(n_channels, factor*n_channels, kernel_size=1, stride=1)
        self.pregate = conv_fn(factor*n_channels, 2*n_channels, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, shortcut=None, cond=None):
        _x = self.causalconv(self.activation(x))
        _x = self.activation(_x)
        if shortcut is not None:
            _x = _x + self.shortcut(shortcut)
        if cond is not None:
            _x = _x + self.conditional(cond)
        if self.dropout is not None:
            _x = self.dropout(_x)

        _x = self.pregate(_x)
        _x = _x[:, :_x.shape[1]//2] * torch.sigmoid(_x[:, _x.shape[1]//2:])
        return x + _x


class AttentionBlock(nn.Module):
    def __init__(self, causal_mask, n_in_channels=128, value_channels=128, key_channels=16, n_background_channels=2, dropout=0.5, n_heads=8):
        super().__init__()
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.n_heads = n_heads
        self.conv_keys_values = nn.Sequential(
                                        GatedResidualBlock(2*n_in_channels + n_background_channels, dropout=dropout),
                                        Conv2d(2*n_in_channels + n_background_channels, value_channels + key_channels * n_heads, kernel_size=1, stride=1)
                                )
        self.conv_queries = nn.Sequential(
                                        GatedResidualBlock(n_in_channels + n_background_channels, dropout=dropout),
                                        Conv2d(n_in_channels + n_background_channels, key_channels * n_heads, kernel_size=1, stride=1)
                                )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
<<<<<<< HEAD
        self.register_buffer('causal_order', causal_mask.sum(-1, keepdims=True))
=======
>>>>>>> 96ac9f2dda737c874371c7532770788dd0e83b53
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x, residual, background):

        background = background.expand(x.shape[0], -1, -1, -1)
        keys, values = self.conv_keys_values(torch.cat([x, residual, background], dim=1)).split([self.key_channels * self.n_heads, self.value_channels], dim=1)
        queries = self.conv_queries(torch.cat([x, background], dim=1)).reshape(x.shape[0], self.n_heads, self.key_channels, -1) # b x n_heads x k x (hxw)
        keys = keys.reshape(x.shape[0], self.n_heads, self.key_channels, -1) # b x n_heads x k x (hxw)
        values = values.reshape(x.shape[0], self.n_heads, self.value_channels // self.n_heads, -1) # b x n_heads x c x (hxw)

        def test_attention(keys, queries, values):
            B, K, N = keys.shape
            _, V, _ = values.shape  
            device = keys.get_device()


            logits = torch.zeros(B, N, N).to(device) - 1e10
            for i in range(N):
                for j in range(0, i): # causal
                    k = keys[:, :, i] # ith key. B x K 
                    q = queries[:, :, j] # jth queries B x K
                    
                    logit  = (k * q).sum(-1) # logit ij (B)
                    logits[:, i, j] = logit
            
            _values = torch.zeros(B, V, N).to(device)
            weights = F.softmax(logits/(K ** 0.5), dim=2) * torch.tril(torch.ones_like(logits), diagonal=-1) # B x K x Q
            
            # for i in range(N):
            #     v = (values * weights[:, i:i+1]).sum(dim=2) # B x V
            #     _values[:, :, i] = v
            
            _values = torch.matmul(weights, values.permute(0, 2, 1)).permute(0, 2, 1)


            s = weights.sum(dim=2)
            s[:, 0] = 1
            assert torch.allclose(s, torch.ones_like(s))

            causal = torch.triu(weights)
            assert torch.allclose(torch.zeros_like(causal), causal), f'{causal[0]} {causal[3]}'

            return _values, weights


        logits = torch.einsum('bhct, bhcj->bhtj', keys, queries) / (self.key_channels ** 0.5) # b x n_heads x (hxw) x (hxw)

        if self.dropout is not None:
            logits = self.dropout(logits)

<<<<<<< HEAD
        
        weights = F.softmax(logits + (1-self.causal_mask) * -1e10, dim=3)    # b x n_heads x (hxw) x (hxw)
        weights = weights * (self.causal_order != 0).unsqueeze(0)  # zero out first variable


=======
        weights = F.softmax(logits, dim=2) * self.causal_mask.unsqueeze(1) # b x n_heads x (hxw) x (hxw)
        weights = weights / (torch.sum(weights, dim=2, keepdim=True) + 1e-8) # b x (hxw) x (hxw)  # re-normalize
>>>>>>> 96ac9f2dda737c874371c7532770788dd0e83b53
        out = torch.einsum('bhtj, bhvj->bhvt', weights, values) # b x n_heads x c // n_heads x (hxw)
        
        ####### TEST ########
        # s = weights.sum(dim=2)
        # s[:, 0] = 1
        # assert torch.allclose(s, torch.ones_like(s))    

        # _values, _weights = test_attention(keys, queries, values)
        # assert torch.allclose(weights, _weights)
        # assert torch.allclose(out, _values), f'{(out-_values)[0]}'
        
        # printarr(background, x, residual, logits, self.causal_mask, queries, keys, values, out)
        #### END TEST ########
        return out.reshape(x.shape[0], -1, x.shape[2], x.shape[3]) # b x c x h x w



class SNAILBlock(nn.Module):
    def __init__(self, causal_mask, n_channels=128, kernel_size=2, key_channels=16, value_channels=128, n_residuals=4, dropout=0.5):
        super().__init__()
        self.n_residuals = n_residuals
        self.gated_residuals = nn.ModuleList([GatedResidualBlock(n_channels=n_channels, kernel_size=kernel_size, dropout=dropout, conv_fn=DownRightShiftedConv2d) for _ in range(n_residuals)])
        self.attention = AttentionBlock(causal_mask=causal_mask, n_in_channels=n_channels, key_channels=key_channels, value_channels=value_channels, dropout=dropout)
        self.activation = nn.ELU()
        self.attention_conv = Conv2d(value_channels, n_channels, kernel_size=1, stride=1)
        self.residual_conv = Conv2d(n_channels, n_channels, kernel_size=1, stride=1)
        self.out = Conv2d(n_channels, n_channels, kernel_size=1, stride=1)


    def forward(self, residual, background, cond=None):
        h = residual
        for i in range(self.n_residuals):
            _h = self.gated_residuals[i](x=h, cond=cond)
        attn = self.attention_conv(self.activation(self.attention(h, _h, background)))
        h = self.activation(self.residual_conv(self.activation(h)))
        out = self.out(self.activation(attn + h))
        return self.activation(out)
    

class PixelSNAIL(nn.Module):
<<<<<<< HEAD
    def __init__(self, input_dims=(1, 50, 50), n_channels=256, kernel_size=2, key_channels=16, value_channels=128, n_blocks=4, n_residuals=4, n_log_components=10, dropout=0.5):
=======
    def __init__(self, input_dims=(1, 50, 50), n_channels=256, kernel_size=3, key_channels=16, value_channels=128, n_blocks=4, n_residuals=4, n_log_components=10, dropout=0.5):
>>>>>>> 96ac9f2dda737c874371c7532770788dd0e83b53
        super().__init__()
        self.n_channels = n_channels
        self.n_residuals = n_residuals
        self.n_blocks = n_blocks
        self.input_dims = input_dims
        self.n_log_components = n_log_components

        I, H, W = input_dims

        # causal mask
        causal_mask = torch.tril(torch.ones((1, H * W, H * W)), diagonal=-1)
        self.register_buffer('causal_mask', causal_mask)

        # layers
        self.causal_horizontal = DownRightShiftedConv2d(I, n_channels, kernel_size=(1, 2), stride=1)
        self.causal_vertical = DownShiftedConv2d(I, n_channels, kernel_size=(2, 3), stride=1)

        self.blocks = nn.ModuleList([SNAILBlock(causal_mask=self.causal_mask, n_channels=n_channels, kernel_size=kernel_size, key_channels=key_channels, value_channels=value_channels, n_residuals=n_residuals, dropout=dropout) for _ in range(n_blocks)])
        self.out_conv = nn.Conv2d(n_channels, 3 * n_log_components, kernel_size=1, stride=1)
        self.activation = nn.ELU()

        # positional encoding
        
        background_h = ((torch.arange(H) - H / 2) / H).reshape(1, 1, H, 1).expand(1, 1, H, W)
        background_w = ((torch.arange(W) - W / 2) / W).reshape(1, 1, 1, W).expand(1, 1, H, W)
        self.register_buffer('background', torch.cat([background_h, background_w], dim=1).float())

    def forward(self, x, cond=None):
        v_x = down_shift(self.causal_vertical(x))
        h_x = right_shift(self.causal_horizontal(x))
        h = v_x + h_x  # causal input
        for i in range(self.n_blocks):
            h = self.blocks[i](h, self.background, cond)
        return self.out_conv(self.activation(h))



def build_ema_optimizer(optimizer_cls):
    class Optimizer(optimizer_cls):
        def __init__(self, *args, polyak=0.9995, **kwargs):
            if not 0.0 <= polyak <= 1.0:
                raise ValueError("Invalid polyak decay rate: {}".format(polyak))
            super().__init__(*args, **kwargs)
            self.defaults['polyak'] = polyak

        def step(self, closure=None):
            super().step(closure)

            # update exponential moving average after gradient update to parameters
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]

                    # state initialization
                    if 'ema' not in state:
                        state['ema'] = torch.zeros_like(p.data)

                    # ema update
                    state['ema'] -= (1 - self.defaults['polyak']) * (state['ema'] - p.data)
            self.swap_ema()

        def swap_ema(self):
            """ substitute exponential moving average values into parameter values """
            for group in self.param_groups:
                for p in group['params']:
                    data = p.data
                    state = self.state[p]
                    p.data = state['ema']
                    state['ema'] = data

        def __repr__(self):
            s = super().__repr__()
            return self.__class__.__mro__[1].__name__ + ' (\npolyak: {}\n'.format(self.defaults['polyak']) + s.partition('\n')[2]

    return Optimizer


class PixelSNAILTrainerMNIST(L.LightningModule):
<<<<<<< HEAD
    def __init__(self, n_channels=128, n_blocks=10, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.model = PixelSNAIL(input_dims=(1, 28, 28), n_channels=n_channels, n_blocks=n_blocks, dropout=0.1)
=======
    def __init__(self, n_channels=64, n_blocks=10, lr=0.8e-4):
        super().__init__()
        self.lr = lr
        self.model = PixelSNAIL(input_dims=(1, 28, 28), n_channels=n_channels, n_blocks=n_blocks)
>>>>>>> 96ac9f2dda737c874371c7532770788dd0e83b53
        self.embedding = nn.Embedding(10, n_channels)
        self.save_hyperparameters()
    
    def forward(self, x, cond=None):
<<<<<<< HEAD
        cond = self.embedding(cond).reshape(x.shape[0], -1, 1, 1)
        # cond=None
=======
        # cond = self.embedding(cond).reshape(x.shape[0], -1, 1, 1)
        cond=None
>>>>>>> 96ac9f2dda737c874371c7532770788dd0e83b53
        return self.model(x, cond)
    
    def _run_step(self, x, label):
        # printarr(x)
        x = x * 2 - 1 
<<<<<<< HEAD
        params = self.forward(x, label).permute(0, 2, 3, 1).contiguous().unsqueeze(1) # permute and add color channel
        n_c = self.model.n_log_components

        logistic = QuantizedLogisticMixture(loc=params[..., :n_c], log_scale=params[..., n_c:2*n_c], logit_probs=params[..., 2*n_c:])
        log_probs = logistic.log_prob(x)
        loss = -log_probs.mean()
        # out = self.forward(x, label)
        # loss = F.cross_entropy(out, ((x+1)*255/2).long().squeeze(), reduction='sum')/x.shape[0]

=======
        params = self.forward(x, label).permute(0, 2, 3, 1).contiguous()
        n_c = self.model.n_log_components
        logistic = QuantizedLogisticMixture(loc=params[..., :n_c], log_scale=params[..., n_c:2*n_c], logit_probs=params[..., 2*n_c:])
        log_probs = logistic.log_prob(x)
        loss = -log_probs.mean()
>>>>>>> 96ac9f2dda737c874371c7532770788dd0e83b53
        return loss
    
    def training_step(self, batch, batch_idx):
        x, label = batch
        loss = self._run_step(x, label)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, label = batch
        loss = self._run_step(x, label)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, label = batch
        loss = self._run_step(x, label)
        # compute bits per dim
        n_bits = loss / (np.log(2) * np.prod(x.shape[1:]))
        self.log('test_loss', n_bits)
        return n_bits

    def configure_optimizers(self):
<<<<<<< HEAD
        # optimizer = build_ema_optimizer(torch.optim.RAdam)(self.parameters(), lr=self.lr)
=======
        # optimizer = build_ema_optimizer(torch.optim.Adam)(self.parameters(), lr=self.lr)
>>>>>>> 96ac9f2dda737c874371c7532770788dd0e83b53
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr)
        return optimizer
    

<<<<<<< HEAD
def test_pixelsnail_mnist(generate=False):
=======
def test_pixelsnail_mnist():
>>>>>>> 96ac9f2dda737c874371c7532770788dd0e83b53
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader

    model = PixelSNAILTrainerMNIST()
<<<<<<< HEAD
    trainer = L.Trainer(max_epochs=20, accelerator='gpu', limit_train_batches=0.2, limit_val_batches=0.1)
=======
    trainer = L.Trainer(max_epochs=20, accelerator='gpu')
>>>>>>> 96ac9f2dda737c874371c7532770788dd0e83b53
    _transforms = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
    
    dataset = MNIST('data', train=True, download=True, transform=_transforms)
    # split
<<<<<<< HEAD
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
=======
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [55000, 5000])
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
>>>>>>> 96ac9f2dda737c874371c7532770788dd0e83b53
    # train
    trainer.fit(model, train_dataloader, val_dataloader)
    # test
    test_dataset = MNIST('data', train=False, download=True, transform=_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    trainer.test(model, test_dataloader)


<<<<<<< HEAD


if __name__=='__main__': 
    torch.set_float32_matmul_precision('medium')
=======
if __name__=='__main__': 
>>>>>>> 96ac9f2dda737c874371c7532770788dd0e83b53
    test_pixelsnail_mnist()