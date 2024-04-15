'''
    Implementation of MADE: Masked Autoencoder for Distribution Estimation
    author: Rafael Rodriguez-Sanchez
    date: February 2023
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, mask=None):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dims, order=None, activation='silu', permute=False, sample=False):
        super(MADE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.layers = nn.ModuleList(self._create_layers(order=order, permute=permute, sample=sample))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def _mask(self, in_dim, out_dim, max_in, max_out):
        '''
            Creates a mask for the linear layer
            in_dim: input dimension
            out_dim: output dimension
        '''

        mask = torch.zeros(out_dim, in_dim)
        for i in range(in_dim):
            for j in range(out_dim):
                mask[j, i] = 1 if  max_out[j] >= max_in[i] else 0
        return mask

    def _create_layers(self, order=None, permute=False, sample=False):
        '''
            Creates the layers of the MADE
            permute: if true, the order of the input layer is permuted
        '''
        layers = []
        in_dim = self.input_dim
        _max_in = np.arange(1, in_dim + 1) if order is None else order
        _max_in =  np.random.permutation(_max_in) if permute else _max_in
        max_in = _max_in
        masks = []
        for i, h_dim in enumerate(self.hidden_dims):
            
            if not sample:
                # distribute equally
                n = in_dim-max_in.min()
                max_out = np.mod(np.arange(h_dim), n) + max_in.min()
            else:
                maxes = np.arange(max_in.min(), in_dim)
                max_out = np.random.choice(maxes, h_dim, replace=True)

            mask = self._mask(in_dim, h_dim, max_in, max_out)
            layers.append(MaskedLinear(in_dim, h_dim, mask=mask))
            layers.append(self._activation(self.activation))
            in_dim = h_dim
            max_in = max_out
            masks.append(mask)
        mask = self._mask(in_dim, self.input_dim, max_in, _max_in-1)
        layers.append(MaskedLinear(in_dim, self.output_dim, mask=mask))

        masks.append(mask)
        ## Test autoregressiveness
        final_mask = masks[0]
        for i in range(len(masks)-1):
            final_mask = torch.matmul(masks[i+1], final_mask)
        assert torch.triu(final_mask[_max_in-1][:, _max_in-1], diagonal=0).sum() == 0, f'The model is not autoregressive {final_mask}'

        return layers
    
    def _activation(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'leaky_relu':
            return nn.LeakyReLU()
        elif name == 'elu':
            return nn.ELU()
        elif name == 'silu':
            return nn.SiLU()
        elif name == 'mish':
            return nn.Mish()
        else:
            raise NotImplementedError('Activation function not implemented')

class ConditionalMADE(MADE):
    def __init__(self, input_dim, hidden_dims, condition_dim, order=None, activation='relu', permute=False, sample=False):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.condition_dim = condition_dim
        self.layers = nn.ModuleList(self._create_layers(order=order, permute=permute, sample=sample))
    
    def forward(self, x, condition):
        x = torch.cat([condition, x], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def _cond_mask(self, hdim):
            
        cond_mask = torch.ones(hdim, self.condition_dim)
        return cond_mask
    
    def _create_layers(self, order=None, permute=False, sample=False):
        '''
            Creates the layers of the MADE
            permute: if true, the order of the input layer is permuted
        '''
        layers = []
        in_dim = self.input_dim
        _max_in = np.arange(1, in_dim + 1) if order is None else order
        _max_in =  np.random.permutation(_max_in) if permute else _max_in
        max_in = _max_in
        masks = []

        for i, h_dim in enumerate(self.hidden_dims):
            
            if not sample:
                # distribute equally
                n = in_dim-max_in.min()
                max_out = np.mod(np.arange(h_dim), n) + max_in.min()
            else:
                maxes = np.arange(max_in.min(), in_dim)
                max_out = np.random.choice(maxes, h_dim, replace=True)

            mask = self._mask(in_dim, h_dim, max_in, max_out)
            if i == 0: # input layer
                cond_mask = self._cond_mask(h_dim)
                mask = torch.cat([cond_mask, mask], dim=1)
                layers.append(MaskedLinear(in_dim + self.condition_dim, h_dim, mask=mask))
            else:
                layers.append(MaskedLinear(in_dim, h_dim, mask=mask))
    
            layers.append(self._activation(self.activation))
            in_dim = h_dim
            max_in = max_out
            masks.append(mask)

        mask = self._mask(in_dim, self.input_dim, max_in, _max_in-1)
        layers.append(MaskedLinear(in_dim, self.output_dim, mask=mask))

        masks.append(mask)
        ## Test autoregressiveness
        final_mask = masks[0]
        for i in range(len(masks)-1):
            final_mask = torch.matmul(masks[i+1], final_mask)
        # assert torch.triu(final_mask[_max_in-1][:, _max_in-1], diagonal=0).sum() == 0, f'The model is not autoregressive {final_mask}'

        return layers


if __name__=='__main__':
   NN=MADE(3, [10,  10], sample=True)
   NN(torch.randn(1, 3))