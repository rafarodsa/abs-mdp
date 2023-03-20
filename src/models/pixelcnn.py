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
from tqdm import tqdm

import logging
from src.utils.printarr import printarr

logger = logging.getLogger(__name__)

from itertools import product

class MaskedConv2d(nn.Module):
    """
        Implements Masked CNN for 3D inputs.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, data_channels=3, mask_type='A', type='vertical'):
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

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
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
    def __init__(self, in_channels, out_channels=32, kernel_size=3, data_channels=1, mask_type='B', residual=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels = out_channels
       
        _out_channels = self.out_channels * 2
        #TODO: check the type of mask needed.
        self.vertical_stack = MaskedConv2d(self.in_channels, _out_channels, kernel_size=kernel_size, stride=1, padding='same', data_channels=data_channels, type='vertical', mask_type=mask_type)
        self.horizontal_stack = MaskedConv2d(self.in_channels, _out_channels, kernel_size=(1,kernel_size), stride=1, padding='same', data_channels=data_channels, type='horizontal', mask_type=mask_type)
        self.link = nn.Conv2d(_out_channels, _out_channels , kernel_size=1, stride=1, padding='same')
        self.skip = nn.Conv2d(_out_channels // 2, _out_channels // 2, kernel_size=1, stride=1, padding='same')
        self.residual = nn.Conv2d(_out_channels // 2, _out_channels // 2, kernel_size=1, stride=1, padding='same') if residual else None
        self.conditional = nn.Conv2d(self.in_channels, _out_channels, kernel_size=1, stride=1, padding='same')
        self.dropout = nn.Dropout(0.5)


    def forward(self, vertical, horizontal, conditional=None):
        _cond = self.conditional(conditional) if conditional is not None else 0.
        _vertical = self.vertical_stack(vertical)
        _horizontal = self.horizontal_stack(horizontal)

        
        _link = self.link(_vertical)
        _color_channels = 0. #self.channels_conv(horizontal) # TODO: check masked conv is working
        _horizontal = _horizontal + _link + _color_channels
        
        _vin = _vertical + _cond
        _hin = _horizontal + _cond

        _vertical = torch.sigmoid(_vin[:, :self.out_channels, :, :]) * torch.tanh(_vin[:, self.out_channels:, :, :])
        _horizontal = torch.sigmoid(_hin[:, :self.out_channels, :, :]) * torch.tanh(_hin[:, self.out_channels:, :, :])
        
        _skip = self.skip(_horizontal)
        if self.residual is not None:
            _residual = self.residual(_horizontal)# out_channels
            _horizontal = horizontal + _residual

        return _vertical, _horizontal, _skip

class PixelCNNStack(nn.Module):
    def __init__(self, out_channels, kernel_size=3, n_layers=10, data_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.layers = nn.ModuleList([GatedPixelCNNLayer(out_channels, out_channels, kernel_size, data_channels=data_channels) for _ in range(n_layers)])
    
    def forward(self, vertical, horizontal, conditional=None):
        skip = 0.
        _v, _h = vertical, horizontal
        for layer in self.layers:
            _v, _h, _skip = layer(_v, _h, conditional)
            skip += _skip
        return _v, _h, skip #+conditional


# PixelCNN decoder
class DeconvBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.mlp = nn.Linear(cfg.input_dim, cfg.in_channels)

        feat_maps = cfg.out_channels
        hidden_dim = cfg.mlp_hidden
        self.mlp_1 = nn.Linear(cfg.input_dim, hidden_dim)
        self.mlp_2 = nn.Linear(hidden_dim, 12 * 12)
        self.relu = nn.ReLU()
        self.conv0 = nn.Conv2d(1, feat_maps * cfg.color_channels, kernel_size=1, stride=1, padding='same')
        self.deconv1 = nn.ConvTranspose2d(feat_maps * cfg.color_channels, feat_maps * cfg.color_channels, kernel_size=3, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(feat_maps * cfg.color_channels, feat_maps * cfg.color_channels, kernel_size=3, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(feat_maps * cfg.color_channels, feat_maps* cfg.color_channels, kernel_size=3, stride=2, padding=0)
        self.conv1 = nn.Conv2d(feat_maps* cfg.color_channels, feat_maps* cfg.color_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(feat_maps * cfg.color_channels, feat_maps * cfg.color_channels, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(feat_maps * cfg.color_channels, cfg.out_channels * cfg.color_channels, kernel_size=4, stride=1, padding='valid')
    
    def forward(self, x):
       


        # x = x * 100
        # device = x.get_device() if x.is_cuda else 'cpu'
        # i = torch.zeros(x.shape[0], 100,100).to(device)
        # idx = x.long()
        # i[torch.arange(x.shape[0]).long(), idx[:, 0], idx[:, 1]] = 1.
        # print(i[0].argmax(dim=0), i[0].argmax(dim=0).argmax())
        # print(idx[0])
        # t = i.argmax(dim=1)
        # coords_x, coords_y = t.max(dim=1)#, t.argmax(dim=1)
        # print(coords_x, coords_y)
        # assert torch.allclose(idx, torch.cat([coords_x.unsqueeze(1), coords_y.unsqueeze(1)], dim=1))
        # i = i.unsqueeze(1)
        # x = self.relu(self.conv0(i)

        x = self.mlp_2(self.relu(self.mlp_1(x)))
        x = self.relu(self.conv0(x.reshape(x.shape[0], 1, 12, 12)))
        x = self.conv1(self.relu(self.deconv1(x)))
        x = self.conv2(self.relu(self.deconv2(x)))
        x = self.conv3(self.relu(self.deconv3(self.relu(x))))
        return x


class PixelCNNDistribution(nn.Module):
        def __init__(self, decoder, h):
            super().__init__()
            self.decoder = decoder
            self.h = h
        
        def forward(self, x):
            print(self.h)
            return self.decoder._pixelcnn_stack(x, self.h)
            
        def log_prob(self, x):
          
            logger.debug(f'x.shape: {x.shape}, h.shape: {self.h.shape}')
            if len(x.shape) == len(self.h.shape):
                assert x.shape[0] == self.h.shape[0]
                assert x.shape[-2:] == self.h.shape[-2:]
                log_prob = F.log_softmax(self.forward(x), dim=1)
                logger.debug(f'log_prob.shape: {log_prob.shape}, x.shape: {x.shape}')
                log_prob = torch.gather(log_prob, dim=1, index=((self.decoder.color_levels-1) * x.unsqueeze(1)).long())
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
                
                log_prob = F.log_softmax(self(x), dim=1) # N x 256 x 3 x w x h
                logger.debug(f'log_prob.shape: {log_prob.shape}, _x.shape: {_x.shape}')
                print(x)
                log_prob = torch.gather(log_prob, dim=1, index=((self.decoder.color_levels-1) * _x.unsqueeze(1)).long()).squeeze(1)
                
                # iters = product(*list(map(lambda x: list(range(x)), batch_dims)))
                # log_prob_test = torch.zeros(*batch_dims, B, x.shape[-3], x.shape[-2], x.shape[-1])
                # for idx in iters:
                #     log_prob_test[idx] = F.log_softmax(self.decoder(x[idx], self.h), dim=1).gather(dim=1, index=(255 * x[idx].unsqueeze(1)).long()).squeeze(1)
                
                # assert torch.allclose(log_prob.reshape(*batch_dims, B, -1), log_prob_test.reshape(*batch_dims, B, -1))
                # logger.debug('log_prob_test passed!')

                log_prob = log_prob.reshape(*batch_dims, B, -1).sum(-1)
                logger.debug(f'final log_prob.shape: {log_prob.shape}')
            return log_prob
        
        def sample(self, n_samples=1):
            cond = self.h
            if n_samples > 1:
                cond = cond.repeat_interleave(n_samples, dim=0) # batch*n, channels, width, height

            width, height = cond.shape[-2:]
            _device = self.h.get_device()
            sample = torch.zeros(cond.shape[0], self.decoder.data_channels, width, height).to(_device)
            with tqdm(total=width*height) as pbar:
                for i in range(width):
                    for j in range(height):
                        for c in range(self.decoder.data_channels):
                            out = self.decoder._pixelcnn_stack(sample/self.decoder.color_levels, cond)
                            out = out[..., c, i, j]
                            out = torch.softmax(out, dim=1)
                            pixel = torch.multinomial(out, 1).squeeze(-1)
                            sample[..., :, c, i, j] = pixel
                        pbar.update(1)
            if n_samples > 1:
                sample = sample.reshape(n_samples, -1, self.decoder.data_channels, width, height)
            return sample
        


class PixelCNNDecoder(nn.Module):
    def __init__(self, features, cfg):
        
        super().__init__()
        self.features = features
        self.data_channels = cfg.color_channels
        self.width, self.height = cfg.out_width, cfg.out_height
        self.color_levels = cfg.color_levels
        self.causal_block = GatedPixelCNNLayer(cfg.color_channels, cfg.feats_maps * self.data_channels, kernel_size=cfg.kernel_size, data_channels=self.data_channels, residual=False, mask_type='A')
       
        self.stack = PixelCNNStack(cfg.feats_maps * self.data_channels, cfg.kernel_size, cfg.n_layers, data_channels=self.data_channels)
        # TODO: maybe reshape and then 1-conv. or masked 1-conv for output
        self.conv = nn.Conv2d(cfg.feats_maps * self.data_channels, 128, kernel_size=1, stride=1)
        self.output = nn.Conv2d(128, self.color_levels * self.data_channels, kernel_size=1, stride=1, padding='same')
        # self.output = MaskedConv2d(cfg.feats_maps * self.data_channels, 256 * self.data_channels, kernel_size=1, stride=1, padding='same', mask_type='A')
    def forward(self, x, h):
        '''
            x: input image/generating image
            h: embedding/latent feature maps, 
        '''
        cond = self.features(h)
        return self._pixelcnn_stack(x, cond)
    
    def _pixelcnn_stack(self, x, cond):
        # TODO: generalize this for multiple batch dimensions.
        width, height = self.width, self.height #cond.shape[-2:]
        x_v, x_h, _ = self.causal_block(x, x)
        _, _, skip = self.stack(x_v, x_h, cond)
        out = self.output(F.relu(self.conv(F.relu(skip))))
        out = out.reshape(x.shape[0], self.color_levels, self.data_channels, width, height) # (batch_size, 256, data_channels, width, height)
        return out 

    def sample(self, h, n_samples=1, device=None):
        '''
            h: embedding/latent feature maps.
        '''
        cond = self.features(h) # batch, channels, width, height
        if n_samples > 1:
            cond = cond.repeat_interleave(n_samples, dim=0) # batch*n, channels, width, height

        width, height = cond.shape[-2:]
        _device = h.get_device() if device is None else device
        sample = torch.zeros(cond.shape[0], self.data_channels, width, height).to(_device)
        with tqdm(total=width*height) as pbar:
            for i in range(width):
                for j in range(height):
                    for c in range(self.data_channels):
                        out = self.forward(sample/(self.color_levels-1), h)
                        out = out[..., c, i, j]
                        out = torch.softmax(out, dim=1)
                        pixel = torch.multinomial(out, 1).squeeze(-1)
                        sample[..., :, c, i, j] = pixel
                    pbar.update(1)
        if n_samples > 1:
            sample = sample.reshape(n_samples, -1, self.data_channels, width, height)
        return sample


    def distribution(self, h):
        '''
            return function to evaluate the distribution log q(x|h).
            h: embedding/latent feature maps, 
        '''
        cond = self.features(h)

        return PixelCNNDistribution(self, cond)

    
    def log_prob(self, x, h):
        '''
            x: input image/generating image
            h: embedding/latent feature maps, 
        '''
        out = self.forward(x, h)
        out = F.log_softmax(out, dim=1) # N x color_levels x data_channels x H x W
        # print(out.shape)
        idx = (x * (self.color_levels-1)).long() # N x data_channels x H x W
        # printarr(out, idx)
        log_probs = -F.nll_loss(out, idx, reduction='none').reshape(x.shape[0], -1).sum(-1)
        return log_probs
    

    @staticmethod
    def PixelCNNDecoderDist(cfg):
        return partial(PixelCNNDecoder, cfg=cfg)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_ = False
        return self
    