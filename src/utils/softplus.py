import torch

def Softplus(x, beta=1, threshold=20):
    return torch.log(1 + torch.exp(beta * x)) / beta * (x * beta < threshold).float() + x * (x * beta  >= threshold).float()