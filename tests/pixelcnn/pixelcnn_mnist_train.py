import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms

from src.models import DiagonalGaussianModule
from src.models import PixelCNNStack, GatedPixelCNNLayer
from src.models.quantized_logistic import QuantizedLogisticMixture
from src.utils.printarr import printarr
import matplotlib.pyplot as plt

class PixelCNNDecoderBinary(pl.LightningModule):
    '''
        PixelCNN for binary images (Binarized MNIST)
    '''
    def __init__(self, n_classes, in_channels, out_channels, kernel_size=3, n_layers=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_classes, out_channels)
        self.causal_block = GatedPixelCNNLayer(in_channels, out_channels, kernel_size, mask_type='A') # causal.
        self.stack = PixelCNNStack(out_channels, kernel_size, n_layers-1)
        self.output = nn.Conv2d(out_channels, 256, kernel_size=1, stride=1, padding='same')

    def forward(self, x, label):
        conditional = self.embed(label).reshape(x.shape[0],-1, 1, 1)
        vertical, horizontal, _ = self.causal_block(x, x, None)
        vertical, horizontal, skip = self.stack(vertical, horizontal, None)
        return self.output(F.relu(skip))

    def training_step(self, batch, batch_idx):
        x, label = batch
        out = F.log_softmax(self(x, label), dim=1)
        idx = (x*255).int()
        # printarr(x, out, idx)
        loss = F.nll_loss(out, idx.squeeze(), reduction='none').reshape(x.shape[0], -1).sum(-1).mean()
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, label = batch
        out = F.log_softmax(self(x, label), dim=1)
        idx = (x*255).int().squeeze()
        loss = F.nll_loss(out, idx, reduction='none').reshape(x.shape[0], -1).sum(-1).mean()
        self.log('val_loss', loss, prog_bar=True)
        return loss
   
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def conv_out_dim(in_dim, kernel_size, stride, padding=0):
    return int((in_dim - kernel_size + 2 * padding) / stride + 1)


class PixelCNNLogistic(pl.LightningModule):
    '''
        PixelCNN for binary images (Binarized MNIST)
    '''
    def __init__(self, n_classes, in_channels, out_channels, kernel_size=3, n_layers=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_components = 10
        self.embed = nn.Embedding(n_classes, out_channels)
        self.causal_block = GatedPixelCNNLayer(in_channels, out_channels, kernel_size, mask_type='A') # causal.
        self.stack = PixelCNNStack(out_channels, kernel_size, n_layers-1)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.output = nn.Conv2d(out_channels, self.n_components * 3, kernel_size=1, stride=1)

    def forward(self, x, label):
        conditional = self.embed(label).reshape(x.shape[0],-1, 1, 1)
        vertical, horizontal, _ = self.causal_block(x, x, None)
        vertical, horizontal, skip = self.stack(vertical, horizontal, conditional)
        out = F.relu(skip)
        return self.output(F.relu(self.conv(out))).permute(0,2,3,1).contiguous()

    def training_step(self, batch, batch_idx):
        x, label = batch
        x = x * 2 - 1
        out = self(x, label).unsqueeze(1) # n_channels
        log_probs = QuantizedLogisticMixture(out[..., :self.n_components], out[..., self.n_components:2*self.n_components], out[..., 2*self.n_components:]).log_prob(x)
        # printarr(x, out, idx)
        # loss = F.nll_loss(out, idx.squeeze(), reduction='none').reshape(x.shape[0], -1).sum(-1).mean()
        loss = -log_probs.mean()
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, label = batch
        x = x * 2 - 1
        out = self(x, label).unsqueeze(1)
        log_probs = QuantizedLogisticMixture(out[..., :self.n_components], out[..., self.n_components:2*self.n_components], out[..., 2*self.n_components:]).log_prob(x)
        # printarr(x, out, idx)
        # loss = F.nll_loss(out, idx.squeeze(), reduction='none').reshape(x.shape[0], -1).sum(-1).mean()
        loss = -log_probs.mean()
        self.log('val_loss', loss, prog_bar=True)
        return loss
   
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=5e-5)


# def encoder_conv_continuous(in_width, in_height, in_channels, kernel_size=3, stride=1, hidden_dim=128, latent_dim=2):
#     out_channels_1, out_channels_2 = 64, 32
#     out_width_1, out_height_1 = conv_out_dim(in_width, kernel_size=kernel_size, stride=stride), conv_out_dim(in_height, kernel_size=kernel_size, padding=0, stride=stride)
#     out_width_2, out_height_2 = conv_out_dim(out_width_1, kernel_size=kernel_size, padding=0, stride=stride), conv_out_dim(out_height_1, kernel_size=kernel_size, padding=0, stride=stride)

#     encoder_feats_1 = nn.Sequential(
#                                 nn.Conv2d(in_channels, out_channels_1, kernel_size=kernel_size, stride=stride, padding='valid'),
#                                 nn.LayerNorm([out_channels_1, out_width_1, out_height_1]),
#                                 nn.ReLU(),
#                                 nn.Conv2d(out_channels_1, out_channels_2, kernel_size=kernel_size, stride=stride, padding='valid'),
#                                 nn.LayerNorm([out_channels_2, out_width_2, out_height_2])
#                     )

#     encoder_feats = nn.Sequential(
#                                 encoder_feats_1,
#                                 ConvResidualLayer(out_width_2, out_height_2, out_channels_2, 32),    
#                                 nn.Flatten(),
#                                 nn.Linear(out_channels_2 * out_width_2 * out_height_2, hidden_dim),
#                                 nn.ReLU()
#                     )

#     encoder_mean = nn.Linear(hidden_dim, latent_dim)

#     encoder_log_var = nn.Linear(hidden_dim, latent_dim)

#     return DiagonalGaussianModule((encoder_feats, encoder_mean, encoder_log_var))

if __name__ == "__main__":

    def binarize(img):
        # Binarize MNIST
        img =  (img > 0.7).float()
        # Plot image
        # plt.imshow(img[0], cmap='gray')
        # plt.show()
        return img

    # Load Binarized MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
    train_dataset = MNIST(root='data', train=True, download=True, transform=transform)
    val_dataset = MNIST(root='data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


    # Train
    model = PixelCNNLogistic(n_classes=10, in_channels=1, out_channels=64, kernel_size=7, n_layers=5)

    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', overfit_batches=0.5)
    trainer.fit(model, train_loader, val_loader)

    # Save model
    torch.save(model.state_dict(), 'cond_pixelcnn.pt')