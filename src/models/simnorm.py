import torch

class SimNorm(torch.nn.Module):
    def __init__(self, v=8, temp=1.):
        super().__init__()
        self.v = v
        self.temp = temp
    
    def forward(self, z):
        shape = z.shape
        batch_dims = len(shape[:-1])
        z = z.reshape(*shape[:-1], -1, self.v) if batch_dims > 0 else z.reshape(-1, self.v)
        z = torch.softmax(z/self.temp, dim=-1)
        return z.reshape(*shape)