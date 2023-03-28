'''
    PixelSNAIL model
    author: Rafael Rodriguez-Sanchez
    date: 28 March 2023
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import lightning as L

from src.models.quantized_logistic import QuantizedLogisticMixture
from src.utils.printarr import printarr

# class Conv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         nn.utils.weight_norm(self)

def Conv2d(*args, **kwargs):
    return nn.utils.weight_norm(nn.Conv2d(*args, **kwargs))

class MaskedConv2d(nn.Module):
    """
        Implements Masked CNN for 3D inputs.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, data_channels=1, mask_type='A', type='vertical'):
        """
                data_channels: number of color channel (1 for grayscale/binary, 3 for RGB)
                mask_type: 'A' (do not include current value as input) or 'B' (include current value as input)
        """
        super().__init__()
        self.data_channels = data_channels
        self.mask_type = mask_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        x_c, y_c = (kernel_size // 2, kernel_size // 2) if isinstance(kernel_size, int) else (kernel_size[1] // 2, kernel_size[0] // 2)
        k_x, k_y = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (kernel_size[1], kernel_size[0])

        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self._mask = np.zeros(self.conv.weight.size())
       
        if type == 'vertical':
            self._mask[:, :, :y_c, :] = 1.

        if type == 'horizontal':
            self._mask[:, :, y_c, :x_c + 1] = 1.

            for o in range(self.data_channels):
                for i in range(o+1, self.data_channels):
                    self._mask[self._color_mask(i, o), y_c, x_c] = 0.
            
            if mask_type == 'A': 
                for c in range(data_channels):
                    self._mask[self._color_mask(c, c), y_c, x_c] = 0.



        self.device = self.conv.weight.get_device()
        self._mask = torch.from_numpy(self._mask).type(self.conv.weight.dtype)
        self.register_buffer('mask', self._mask)

     
        idx_x, idx_y = torch.arange(0, k_x), torch.arange(0, k_y)
        if type == 'vertical':
            assert torch.all(self.mask[:, :, idx_y < y_c] == 1) and torch.all(self.mask[:, :, idx_y >= y_c] == 0)
        elif mask_type == 'A':
            assert torch.all(self.mask[:, :, idx_y == y_c, idx_x < x_c] == 1) and torch.all(self.mask[:, :, idx_y == y_c, idx_x >= x_c] == 0) and torch.all(self.mask[:, :, idx_y != y_c] == 0)
        else:
            assert torch.all(self.mask[:, :, idx_y == y_c, idx_x <= x_c] == 1) and torch.all(self.mask[:, :, idx_y == y_c, idx_x > x_c] == 0) and torch.all(self.mask[:, :, idx_y != y_c] == 0)



    def _color_mask(self, in_c, out_c):
        """
            Indices for masking weights.
            For RGB: B is conditioned (G,B), G is conditioned on R.
        """
        a = np.arange(self.out_channels) % self.data_channels == out_c
        b = np.arange(self.in_channels) % self.data_channels == in_c

        return a[:, None] * b[None, :]


    def forward(self, x):
        # printarr(self.mask, self.conv.weight_g, self.conv.weight_v, self.conv.weight)
        self.conv.weight_v = nn.Parameter(self.conv.weight_v * self.mask)
        # self.conv.weight = nn.Parameter(self.conv.weight * self.mask)
        return self.conv(x)

    def _get_device(self):
        try:
            self.device = self.conv.weight.get_device()
            if self.device < 0:
                self.device = torch.device('cpu', 0)
        except RuntimeError:
            self.device = torch.device('cpu', 0)


class GatedResidualBlock(nn.Module):
    def __init__(self, n_channels=128, kernel_size=1, mask_type='B', dropout=0.5):
        super().__init__()
        self.activation = nn.ELU()
        self.vertical_stack = MaskedConv2d(n_channels, 2*n_channels, kernel_size, stride=1, padding='same', mask_type=mask_type, type='vertical')
        self.link = MaskedConv2d(2*n_channels, 2*n_channels, kernel_size=1, stride=1, padding=0, mask_type='B', type='horizontal')
        self.horizontal_stack = MaskedConv2d(n_channels, 2*n_channels, kernel_size, stride=1, padding='same', mask_type=mask_type, type='horizontal')
        self.conditional = Conv2d(n_channels, 2*n_channels, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None


    def forward(self, h_x, v_x=None, c=None):
        
        _h_x = self.horizontal_stack(self.activation(h_x))
        if v_x is not None:
            _v_x = self.vertical_stack(self.activation(v_x))
            h = self.link(self.activation(_v_x)) + _h_x
            v = _v_x 
            if self.dropout is not None:
                h = self.dropout(h)
                v = self.dropout(v)
            if c is not None:
                h = h + self.conditional(c)
                v = v + self.conditional(c)
            h = h[:, :h.shape[1]//2, :, :] * torch.sigmoid(h[:, h.shape[1]//2:, :, :])
            v = v[:, :v.shape[1]//2, :, :] * torch.sigmoid(v[:, v.shape[1]//2:, :, :])
        else:
            h = _h_x
            v = None
            if self.dropout is not None:
                h = self.dropout(h)
            if c is not None:
                h = h + self.conditional(c)
            h = h[:, :h.shape[1]//2, :, :] * torch.sigmoid(h[:, h.shape[1]//2:, :, :])
      
        return (v, h) if v is not None else h


class AttentionBlock(nn.Module):
    def __init__(self, causal_mask, n_in_channels=128, input_channels=1, value_channels=128, key_channels=16, n_background_channels=2, dropout=0.5):
        super().__init__()
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.conv_keys_values = nn.Sequential(
                                        GatedResidualBlock(n_in_channels + n_background_channels + input_channels, mask_type='B', dropout=dropout),
                                        Conv2d(n_in_channels + n_background_channels + input_channels, value_channels + key_channels, kernel_size=1, stride=1)
                                )
        self.conv_queries = nn.Sequential(
                                        GatedResidualBlock(1 + n_background_channels, mask_type='B', dropout=dropout),
                                        Conv2d(1 + n_background_channels, key_channels, kernel_size=1, stride=1)
                                )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x, residual, background):

        background = background.expand(x.shape[0], -1, -1, -1)
        keys, values = self.conv_keys_values(torch.cat([x, residual, background], dim=1)).split([self.key_channels, self.value_channels], dim=1)
        queries = self.conv_queries(torch.cat([x, background], dim=1)).reshape(x.shape[0], self.key_channels, -1) # b x k x (hxw)
        keys = keys.reshape(x.shape[0], self.key_channels, -1) # b x k x (hxw)
        values = values.reshape(x.shape[0], self.value_channels, -1) # b x c x (hxw)
        logits = torch.einsum('bct, bcj->btj', keys, queries) # b x (hxw) x (hxw)

        if self.dropout is not None:
            logits = self.dropout(logits)
        
        weights = F.softmax(logits, dim=2) * self.causal_mask # b x (hxw) x (hxw)
        weights = weights / (torch.sum(weights, dim=2, keepdim=True) + 1e-8) # b x (hxw) x (hxw)  # re-normalize
        out = torch.einsum('bct, btj->bcj', values, weights) # b x c x (hxw)
        # printarr(background, x, residual, logits, self.causal_mask, queries, keys, values, out)
        return out.reshape(x.shape[0], -1, x.shape[2], x.shape[3]) # b x c x h x w
    

class SNAILBlock(nn.Module):
    def __init__(self, causal_mask, n_channels=128, kernel_size=2, key_channels=16, value_channels=128, n_residuals=4, dropout=0.5):
        super().__init__()
        self.n_residuals = n_residuals
        self.gated_residuals = nn.ModuleList([GatedResidualBlock(n_channels=n_channels, kernel_size=kernel_size, dropout=dropout) for _ in range(n_residuals)])
        self.attention = AttentionBlock(causal_mask=causal_mask, n_in_channels=n_channels, key_channels=key_channels, value_channels=value_channels, dropout=dropout)
        self.activation = nn.ELU()
        self.attention_conv = Conv2d(value_channels, n_channels, kernel_size=1, stride=1)
        self.residual_conv = Conv2d(n_channels, n_channels, kernel_size=1, stride=1)
        self.out = Conv2d(n_channels, n_channels, kernel_size=1, stride=1)


    def forward(self, x, residual, background, cond=None):
        v, h = residual
        for i in range(self.n_residuals):
            v, h = self.gated_residuals[i](v_x=v, h_x=h, c=cond)
        attn = self.attention_conv(self.activation(self.attention(x, h, background)))
        h = self.activation(self.residual_conv(self.activation(h)))
        out = self.out(self.activation(attn + h))
        return v, self.activation(out)
    

class PixelSNAIL(nn.Module):
    def __init__(self, input_dims=(1, 50, 50), n_channels=256, kernel_size=3, key_channels=16, value_channels=128, n_blocks=4, n_residuals=4, n_log_components=10, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.n_residuals = n_residuals
        self.n_blocks = n_blocks
        self.input_dims = input_dims
        self.n_log_components = n_log_components

        _, H, W = input_dims

        # causal mask
        causal_mask = torch.tril(torch.ones((1, H * W, H * W)), diagonal=-1)
        self.register_buffer('causal_mask', causal_mask)

        # layers
        self.proj_conv = nn.Conv2d(1, n_channels, kernel_size=1, stride=1)
        self.causal_conv = GatedResidualBlock(n_channels, kernel_size=3, mask_type='A')
        self.blocks = nn.ModuleList([SNAILBlock(causal_mask=self.causal_mask, n_channels=n_channels, kernel_size=kernel_size, key_channels=key_channels, value_channels=value_channels, n_residuals=n_residuals, dropout=dropout) for _ in range(n_blocks)])
        self.out_conv = nn.Conv2d(n_channels, 3 * n_log_components, kernel_size=1, stride=1)
        self.activation = nn.ELU()

        # positional encoding
        
        background_h = ((torch.arange(H) - H / 2) / H).reshape(1, 1, H, 1).expand(1, 1, H, W)
        background_w = ((torch.arange(W) - W / 2) / W).reshape(1, 1, 1, W).expand(1, 1, H, W)
        self.register_buffer('background', torch.cat([background_h, background_w], dim=1).float())

    def forward(self, x, cond=None):
        _x = self.proj_conv(x)
        h = self.causal_conv(_x, _x)
        for i in range(self.n_blocks):
            h = self.blocks[i](x, h, self.background, cond)
        _, h = h
        return self.out_conv(self.activation(h))


class PixelSNAILTrainerMNIST(L.LightningModule):
    def __init__(self, n_channels=128, n_blocks=2, lr=3e-5):
        super().__init__()
        self.lr = lr
        self.model = PixelSNAIL(input_dims=(1, 28, 28), n_channels=n_channels, n_blocks=n_blocks)
        self.embedding = nn.Embedding(10, 128)
        self.save_hyperparameters()
    
    def forward(self, x, cond=None):
        cond = self.embedding(cond).reshape(x.shape[0], -1, 1, 1)
        return self.model(x, cond)
    
    def _run_step(self, x, label):
        # printarr(x)
        x = x * 2 - 1 
        params = self.forward(x, label).permute(0, 2, 3, 1).contiguous()
        n_c = self.model.n_log_components
        logistic = QuantizedLogisticMixture(loc=params[..., :n_c], log_scale=params[..., n_c:2*n_c], logit_probs=params[..., 2*n_c:])
        log_probs = logistic.log_prob(x)
        loss = -log_probs.mean()
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

def test_pixelsnail_mnist():
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader

    model = PixelSNAILTrainerMNIST()
    trainer = L.Trainer(max_epochs=1, accelerator='gpu')
    _transforms = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
    
    dataset = MNIST('data', train=True, download=True, transform=_transforms)
    # split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [55000, 5000])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    # train
    trainer.fit(model, train_dataloader, val_dataloader)
    # test
    test_dataset = MNIST('data', train=False, download=True, transform=_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    trainer.test(test_dataloaders=test_dataloader)


if __name__=='__main__': 
    test_pixelsnail_mnist()