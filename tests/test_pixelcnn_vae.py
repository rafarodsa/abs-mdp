import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms

from src.abstract_mdp.pixelcnn import GatedPixelCNNLayer, PixelCNNStack, encoder_conv_continuous


class PixelCNN_VAE_Binary(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_layers=5, latent_dim=10):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        

        width, height = 28, 28
        self.encoder = encoder_conv_continuous(width, height, in_channels, kernel_size, latent_dim=latent_dim).float()
        self.linear = nn.Linear(latent_dim, out_channels)
        self.causal_block = GatedPixelCNNLayer(in_channels, out_channels, kernel_size)
        self.stack = PixelCNNStack(out_channels, kernel_size, n_layers-1)
        self.output = nn.Conv2d(out_channels, 2, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        code_sample, code_dist, code_std, _ = self.encoder.sample(x)
        code_sample = self.linear(code_sample).reshape(x.shape[0],-1, 1, 1)
        vertical, horizontal, _ = self.causal_block(x, x, None)
        vertical, horizontal, skip = self.stack(vertical, horizontal, code_sample)
        return self.output(F.relu(skip)), code_dist, code_std

    def training_step(self, batch, batch_idx):
        x, _ = batch
        out, q_z, q_std_z = self(x)
        loss = F.cross_entropy(out.reshape(x.shape[0], 2, -1), x.reshape(x.shape[0], -1).long()) + torch.distributions.kl_divergence(q_z, q_std_z).mean()
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out, q_z, q_std_z = self(x)
        loss = F.cross_entropy(out.reshape(x.shape[0], 2, -1), x.reshape(x.shape[0], -1).long()) + torch.distributions.kl_divergence(q_z, q_std_z).mean()
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
    model = PixelCNN_VAE_Binary(in_channels=1, out_channels=32, kernel_size=5, n_layers=5, latent_dim=5)

    trainer = pl.Trainer(max_epochs=1, accelerator='cpu')
    trainer.fit(model, train_loader, val_loader)

    # Save model
    torch.save(model.state_dict(), 'pixelcnn.pt')