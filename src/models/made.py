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
    def __init__(self, input_dim, hidden_dims, permute=False, sample=False):
        super(MADE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList(self._create_layers(permute=permute, sample=sample))

    
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

    def _create_layers(self, permute=False, sample=False):
        '''
            Creates the layers of the MADE
            permute: if true, the order of the input layer is permuted
        '''
        layers = []
        in_dim = self.input_dim
        _max_in = np.arange(1, in_dim + 1) if not permute else np.random.permutation(np.arange(1, in_dim + 1))
        max_in = _max_in
        masks = []
        for i, h_dim in enumerate(self.hidden_dims):
            
            if not sample:
                # distribute equally
                n = in_dim-max_in.min()
                max_out = np.arange(h_dim) % n + max_in.min()
            else:
                maxes = np.arange(max_in.min(), in_dim)
                max_out = np.random.choice(maxes, h_dim, replace=True)

            mask = self._mask(in_dim, h_dim, max_in, max_out)
            layers.append(MaskedLinear(in_dim, h_dim, mask=mask))
            layers.append(nn.ReLU())
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
    

if __name__=='__main__':
   NN=MADE(3, [10,  10], sample=True)
   NN(torch.randn(1, 3))