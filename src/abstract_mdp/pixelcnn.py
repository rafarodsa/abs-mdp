"""
    PixelCNN Decoder implementation.
    
    author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
    date: January 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms


import matplotlib.pyplot as plt


class MaskedConv2d(nn.Module):
    """
        Implements Masked CNN for 3D inputs.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, data_channels=3, mask_type='A'):
        """
                data_channels: number of color channel (1 for grayscale/binary, 3 for RGB)
                mask_type: 'A' (do not include current value as input) or 'B' (include current value as input)
        """
        super().__init__()
        self.data_channels = data_channels
        self.mask_type = mask_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        x_c, y_c = kernel_size // 2, kernel_size // 2 if isinstance(kernel_size, int) else kernel_size[1] // 2, kernel_size[0] // 2


        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask = torch.ones_like(self.conv.weight)
        
        for o in range(self.data_channels):
            for i in range(o+1, self.data_channels):
                self.mask[self._color_mask(i, o), y_c, x_c] = 0
        
        if mask_type == 'A':
            for c in range(data_channels):
                self.mask[self._color_mask(c, c), y_c, x_c] = 0

        

    def _color_mask(self, in_c, out_c):
        """
            Indices for masking weights.
            For RGB: B is conditioned (G,B), G is conditioned on R.
        """
        a = torch.arange(self.out_channels) % self.data_channels == out_c
        b = torch.arange(self.in_channels) % self.data_channels == in_c

        return a.unsqueeze(1) * b.unsqueeze(0)


    def forward(self, x):
        self.conv.weight = nn.Parameter(self.conv.weight * self.mask)
        return self.conv(x)


class GatedPixelCNNLayer(nn.Module):
    """
        PixelCNN layer.
    """
    def __init__(self, in_channels, out_channels=32, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        vertical_kernel = (kernel_size // 2, kernel_size)
        horizontal_kernel = (1, kernel_size // 2)
        self.vertical_stack = nn.Conv2d(in_channels, 2 * out_channels, vertical_kernel, stride=1, padding='same')
        self.horizontal_stack = nn.Conv2d(in_channels, 2 * out_channels, horizontal_kernel, stride=1, padding='same')
        self.link = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1, stride=1, padding='same')
        self.skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding='same')
        self.residual = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding='same')
        self.conditional = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding='same')

    def forward(self, vertical, horizontal, conditional=None):
        _cond = self.conditional(conditional) if conditional is not None else 0.
        _vertical = self.vertical_stack(vertical)
        _horizontal = self.horizontal_stack(horizontal)

        _vertical = self.__translate_and_crop(_vertical, 0, -self.kernel_size // 2) 
        _horizontal = self.__translate_and_crop(_horizontal, -self.kernel_size // 2, 0)
        
        _link = self.link(_vertical)
        _horizontal = _horizontal + _link

        _vertical = torch.sigmoid(_vertical[:, :self.out_channels, :, :] + _cond) * torch.tanh(_vertical[:, self.out_channels:, :, :] + _cond)
        _horizontal = torch.sigmoid(_horizontal[:, :self.out_channels, :, :] + _cond) * torch.tanh(_horizontal[:, self.out_channels:, :, :] + _cond)
        
        _skip = self.skip(_horizontal)
        _residual = self.residual(_horizontal)
        _horizontal = horizontal + _residual

        return _vertical, _horizontal, _skip

    def __translate_and_crop(self, input, x, y):
        """
            Translates the input tensor by x and y. Pads with zeros and crop to the original size.
        """
        _, _, h, w = input.shape

        # pad the input
        pad_x = abs(x)
        pad_y = abs(y)
        
        if x < 0:
            pad_left = pad_x
            pad_right = 0
            x_i, x_f = 0, w
        else:
            pad_left = 0
            pad_right = pad_x
            x_i, x_f = pad_x, w + pad_x
        if y < 0:
            pad_top = pad_y
            pad_bottom = 0
            y_i, y_f = 0, h
        else:
            pad_top = 0
            pad_bottom = pad_y
            y_i, y_f = pad_y, h + pad_y
        
        padded_input = nn.functional.pad(input, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        cropped_input = padded_input[:, :, y_i:y_f, x_i:x_f]
        return cropped_input
            

class PixelCNNStack(nn.Module):
    def __init__(self, out_channels, kernel_size=3, n_layers=10):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.layers = nn.ModuleList([GatedPixelCNNLayer(out_channels, out_channels, kernel_size) for _ in range(n_layers)])
    
    def forward(self, vertical, horizontal, conditional=None):
        skip = 0.
        for layer in self.layers:
            vertical, horizontal, _skip = layer(vertical, horizontal, conditional)
            skip += _skip
        return vertical, horizontal, skip


class PixelCNNDecoderBinary(pl.LightningModule):
    def __init__(self, n_classes, in_channels, out_channels, kernel_size=3, n_layers=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_classes, out_channels)
        self.causal_block = GatedPixelCNNLayer(in_channels, out_channels, kernel_size)
        self.stack = PixelCNNStack(out_channels, kernel_size, n_layers-1)
        self.output = nn.Conv2d(out_channels, 2, kernel_size=1, stride=1, padding='same')

    def forward(self, x, label):
        conditional = self.embed(label).reshape(x.shape[0],-1, 1, 1)
        vertical, horizontal, _ = self.causal_block(x, x, None)
        vertical, horizontal, skip = self.stack(vertical, horizontal, conditional)
        return self.output(F.relu(skip))

    def training_step(self, batch, batch_idx):
        x, label = batch
        out = self(x, label)
        loss = F.cross_entropy(out.reshape(x.shape[0], 2, -1), x.reshape(x.shape[0], -1).long())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, label = batch
        out = self(x, label)
        loss = F.cross_entropy(out.reshape(x.shape[0], 2, -1), x.reshape(x.shape[0], -1).long())
        self.log('val_loss', loss)
        return loss
   
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=5e-3)


if __name__ == "__main__":

    def binarize(img):
        # Binarize MNIST
        img =  (img > 0.7).float()
        # Plot image
        return img

    # Load Binarized MNIST
    transform = transforms.Compose([transforms.ToTensor(), binarize])
    train_dataset = MNIST(root='data', train=True, download=True, transform=transform)
    val_dataset = MNIST(root='data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    # Train
    model = PixelCNNDecoderBinary(10, 1, 32, 5, 5)

    trainer = pl.Trainer(max_epochs=1, accelerator='cpu')
    trainer.fit(model, train_loader, val_loader)

    # Save model
    torch.save(model.state_dict(), 'pixelcnn.pt')