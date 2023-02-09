"""
    PixelCNN Decoder implementation.
    
    author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
    date: January 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import partial

import logging

logger = logging.getLogger(__name__)

from itertools import product

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

        x_c, y_c = (kernel_size // 2, kernel_size // 2) if isinstance(kernel_size, int) else (kernel_size[1] // 2, kernel_size[0] // 2)


        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self._mask = np.ones(self.conv.weight.size())
        
        for o in range(self.data_channels):
            for i in range(o+1, self.data_channels):
                self._mask[self._color_mask(i, o), y_c, x_c] = 0
        
        if mask_type == 'A':
            for c in range(data_channels):
                self._mask[self._color_mask(c, c), y_c, x_c] = 0
        
        self.device = self.conv.weight.get_device()
        self._mask = torch.from_numpy(self._mask).type(self.conv.weight.dtype)
        self.register_buffer('mask', self._mask)

    def _color_mask(self, in_c, out_c):
        """
            Indices for masking weights.
            For RGB: B is conditioned (G,B), G is conditioned on R.
        """
        a = np.arange(self.out_channels) % self.data_channels == out_c
        b = np.arange(self.in_channels) % self.data_channels == in_c

        return a[:, None] * b[None, :]


    def forward(self, x):
        # self._get_device()
        # self.mask = self.mask.to(self.device).type(self.conv.weight.dtype)
        self.conv.weight = nn.Parameter(self.conv.weight * self.mask)
        return self.conv(x)

    def _get_device(self):
        try:
            self.device = self.conv.weight.get_device()
            if self.device < 0:
                self.device = torch.device('cpu', 0)
        except RuntimeError:
            self.device = torch.device('cpu', 0)


class GatedPixelCNNLayer(nn.Module):
    """
        PixelCNN layer.
    """
    def __init__(self, in_channels, out_channels=32, kernel_size=3, data_channels=1, mask_type='A', residual=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels * data_channels
        self.out_channels = out_channels
        vertical_kernel = (kernel_size // 2, kernel_size)
        horizontal_kernel = (1, kernel_size // 2)
        self.out_channels = data_channels * out_channels
        _out_channels = self.out_channels * 2
        self.vertical_stack = nn.Conv2d(self.in_channels, _out_channels, vertical_kernel, stride=1, padding='same')
        self.horizontal_stack = nn.Conv2d(self.in_channels, _out_channels, horizontal_kernel, stride=1, padding='same')
        self.link = nn.Conv2d(_out_channels, _out_channels , kernel_size=1, stride=1, padding='same')
        self.skip = nn.Conv2d(_out_channels // 2, _out_channels // 2, kernel_size=1, stride=1, padding='same')
        self.residual = nn.Conv2d(_out_channels // 2, _out_channels // 2, kernel_size=1, stride=1, padding='same') if residual else None
        self.conditional = nn.Conv2d(self.in_channels, _out_channels // 2, kernel_size=1, stride=1, padding='same')
        self.channels_conv = MaskedConv2d(self.in_channels, _out_channels, kernel_size=1, stride=1, padding='same', data_channels=data_channels, mask_type=mask_type)


    def forward(self, vertical, horizontal, conditional=None):
        _cond = self.conditional(conditional) if conditional is not None else 0.
        _vertical = self.vertical_stack(vertical)
        _horizontal = self.horizontal_stack(horizontal)

        _vertical = self.__translate_and_crop(_vertical, 0, -self.kernel_size // 2) 
        _horizontal = self.__translate_and_crop(_horizontal, -self.kernel_size // 2, 0)
        
        _link = self.link(_vertical)
        _color_channels = self.channels_conv(horizontal) # TODO: check masked conv is working
        _horizontal = _horizontal + _link + _color_channels
        
        _vertical = torch.sigmoid(_vertical[:, :self.out_channels, :, :] + _cond) * torch.tanh(_vertical[:, self.out_channels:, :, :] + _cond)
        _horizontal = torch.sigmoid(_horizontal[:, :self.out_channels, :, :] + _cond) * torch.tanh(_horizontal[:, self.out_channels:, :, :] + _cond)
        
        _skip = self.skip(_horizontal)
        if self.residual is not None:
            _residual = self.residual(_horizontal) # out_channels
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
    def __init__(self, out_channels, kernel_size=3, n_layers=10, data_channels=3):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.layers = nn.ModuleList([GatedPixelCNNLayer(out_channels, out_channels, kernel_size, data_channels=data_channels) for _ in range(n_layers)])
    
    def forward(self, vertical, horizontal, conditional=None):
        skip = 0.
        for layer in self.layers:
            vertical, horizontal, _skip = layer(vertical, horizontal, conditional)
            skip += _skip
        return vertical, horizontal, skip


# PixelCNN decoder
class DeconvBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Linear(cfg.input_dim, cfg.in_channels)
        self.deconv = nn.ConvTranspose2d(cfg.in_channels, cfg.out_channels * cfg.color_channels, (cfg.out_width, cfg.out_height), stride=1, padding=0)

    def forward(self, x):
        x = self.mlp(x)
        x = x.reshape(x.shape[-2], -1, 1, 1)
        x = self.deconv(x)
        return x


class PixelCNNDistribution(nn.Module):
        def __init__(self, decoder, h):
            super().__init__()
            self.decoder = decoder
            self.h = h

        def log_prob(self, x):
            logger.debug(f'x.shape: {x.shape}, h.shape: {self.h.shape}')
            if len(x.shape) == len(self.h.shape):
                assert x.shape[0] == self.h.shape[0]
                assert x.shape[-2:] == self.h.shape[-2:]
                log_prob = F.log_softmax(self.decoder(x, self.h), dim=1)
                logger.debug(f'log_prob.shape: {log_prob.shape}, x.shape: {x.shape}')
                log_prob = torch.gather(log_prob, dim=1, index=(255 * x.unsqueeze(1)).long())
                logger.debug(f'log_prob.shape: {log_prob.shape}')
                log_prob = log_prob.reshape(x.shape[0], -1).sum(-1)
            elif len(x.shape) > len(self.h.shape):
                assert x.shape[-len(self.h.shape)] == self.h.shape[0]
                assert x.shape[-2:] == self.h.shape[-2:] 
                
                batch_dims = x.shape[:-len(self.h.shape)]
                B = self.h.shape[0]
                N = np.prod(batch_dims)
                repeats = (len(self.h.shape)-1) * [1] 
                _h = self.h.repeat(N, *repeats)
                _x = x.reshape(-1, *x.shape[-len(self.h.shape)+1:])
                logger.debug(f'Flattened batch -> x.shape: {_x.shape}, h.shape: {_h.shape}')
                
                log_prob = F.log_softmax(self.decoder(_x, _h), dim=1) # N x 256 x 3 x w x h
                logger.debug(f'log_prob.shape: {log_prob.shape}, _x.shape: {_x.shape}')
                log_prob = torch.gather(log_prob, dim=1, index=(255 * _x.unsqueeze(1)).long()).squeeze(1)
                
                # iters = product(*list(map(lambda x: list(range(x)), batch_dims)))
                # log_prob_test = torch.zeros(*batch_dims, B, x.shape[-3], x.shape[-2], x.shape[-1])
                # for idx in iters:
                #     log_prob_test[idx] = F.log_softmax(self.decoder(x[idx], self.h), dim=1).gather(dim=1, index=(255 * x[idx].unsqueeze(1)).long()).squeeze(1)
                
                # assert torch.allclose(log_prob.reshape(*batch_dims, B, -1), log_prob_test.reshape(*batch_dims, B, -1))
                # logger.debug('log_prob_test passed!')

                log_prob = log_prob.reshape(*batch_dims, B, -1).sum(-1)
                logger.debug(f'final log_prob.shape: {log_prob.shape}')
            return log_prob

class PixelCNNDecoder(nn.Module):
    def __init__(self, features, cfg):
        
        super().__init__()
        self.features = features
        self.data_channels = cfg.color_channels
        # self.conv = nn.Conv2d(cfg.in_channels, cfg.feats_maps * self.data_channels, kernel_size=1, stride=1, padding='same')
        
        #self.causal_block = GatedPixelCNNLayer(1, cfg.feats_maps, kernel_size=1, data_channels=self.data_channels)
        self.causal_block = MaskedConv2d(cfg.in_channels, cfg.feats_maps * self.data_channels, kernel_size=1, stride=1, padding='same', data_channels=self.data_channels, mask_type='A') 
        self.stack = PixelCNNStack(cfg.feats_maps, cfg.kernel_size, cfg.n_layers-1, data_channels=self.data_channels)
        self.output = MaskedConv2d(cfg.feats_maps * self.data_channels, 256 * self.data_channels, kernel_size=1, stride=1, padding='same', data_channels=self.data_channels, mask_type='A')

    def forward(self, x, h):
        '''
            x: input image/generating image
            h: embedding/latent feature maps, 
        '''
        cond = self.features(h)
        return self._pixelcnn_stack(x, cond)
    
    def _pixelcnn_stack(self, x, cond):
        width, height = cond.shape[-2:]
        x_= self.causal_block(x)
        _, _, skip = self.stack(x_, x_, cond)
        out = self.output(F.relu(skip))
        out = out.reshape(x.shape[0], 256, self.data_channels, width, height) # (batch_size, 256, data_channels, width, height)
        return out 

    def sample(self, h, n=1):
        '''
            h: embedding/latent feature maps.
        '''
        cond = self.features(h)
        width, height = cond.shape[-2:]
        sample = torch.zeros(n, self.data_channels, width, height)
        for i in range(width):
            for j in range(height):
                for c in range(self.data_channels):
                    out = self.forward(sample, h)
                    out = out[:, c, i, j]
                    out = torch.softmax(out, dim=1)
                    sample = torch.multinomial(out, 1)
                    sample[:, c, i, j] = sample
        return sample


    def distribution(self, h):
        '''
            return function to evaluate the distribution log q(x|h).
            h: embedding/latent feature maps, 
        '''
        cond = self.features(h)

        return PixelCNNDistribution(self._pixelcnn_stack, cond)

    

    
    def log_prob(self, x, h):
        '''
            x: input image/generating image
            h: embedding/latent feature maps, 
        '''
        out = self.forward(x, h)
        return F.log_softmax(out, dim=1) # TODO: how to batch this. batch computation over batch of distributions

    @staticmethod
    def PixelCNNDecoderDist(cfg):
        return partial(PixelCNNDecoder, cfg=cfg)

    