import torch
import torch.nn as nn

def get_act(activation):
    actlayer = None
    if activation == 'relu':
        actlayer = nn.ReLU()
    elif activation == 'sigmoid':
        actlayer = nn.Sigmoid()
    elif activation == 'tanh':
        actlayer = nn.Tanh()
    elif activation == 'leaky_relu':
        actlayer = nn.LeakyReLU()
    elif activation == 'elu':
        actlayer = nn.ELU()
    elif activation == 'selu':
        actlayer = nn.SELU()
    elif activation == 'gelu':
        actlayer = nn.GELU()
    elif activation == 'glu':
        actlayer = nn.GLU()
    elif activation == 'silu':
        actlayer = nn.SiLU()
    elif activation == 'mish':
        actlayer = nn.Mish()
    elif activation == 'none':
        actlayer == nn.Identity()
    else:
        raise ValueError(f'Invalid activation function {activation}')
    
    return actlayer