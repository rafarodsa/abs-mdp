import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms

from src.models.pixelcnn import GatedPixelCNNLayer, PixelCNNStack, encoder_conv_continuous


import argparse

class PixelCNN_VAE_Binary(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_layers=5, latent_dim=10, lr=1e-3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.lr = lr

        width, height = 28, 28
        self.encoder = encoder_conv_continuous(width, height, in_channels, kernel_size, latent_dim=latent_dim)
        self.linear = nn.Linear(latent_dim, out_channels)
        self.trans_conv_1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=15)
        self.trans_conv_2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=14)
        self.causal_block = GatedPixelCNNLayer(in_channels, out_channels, kernel_size)
        self.stack = PixelCNNStack(out_channels, kernel_size, n_layers-1)
        self.output = nn.Conv2d(out_channels, 2, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        code_sample, code_dist, code_std, _ = self.encoder.sample(x)
        code_sample = self.trans_conv_2(self.trans_conv_1(self.linear(code_sample).reshape(x.shape[0],-1, 1, 1)))
        vertical, horizontal, _ = self.causal_block(x, x, None)
        vertical, horizontal, skip = self.stack(vertical, horizontal, code_sample)
        return self.output(F.relu(skip)), code_dist, code_std

    def sample(self, x, z):
        z_ = self.trans_conv_2(self.trans_conv_1(self.linear(z).reshape(x.shape[0],-1, 1, 1)))
        vertical, horizontal, _ = self.causal_block(x, x, None)
        vertical, horizontal, skip = self.stack(vertical, horizontal, z_)
        return self.output(F.relu(skip))

    def training_step(self, batch, batch_idx):
        x, _ = batch
        out, q_z, q_std_z = self(x)
        recons_loss = F.cross_entropy(out, x.squeeze().long(), reduction='sum')/x.shape[0]
        kl_loss = torch.distributions.kl_divergence(q_z, q_std_z).sum(-1).mean()
        loss = kl_loss + recons_loss

        log_dict = {
            'train_loss': loss,
            'train_kl_loss': kl_loss,
            'train_recons_loss': recons_loss
        }
        self.log_dict(log_dict)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out, q_z, q_std_z = self(x)
        recons_loss = F.cross_entropy(out.reshape(x.shape[0], 2, -1), x.reshape(x.shape[0], -1).long())
        kl_loss = torch.distributions.kl_divergence(q_z, q_std_z).mean()
        loss = kl_loss + recons_loss

        log_dict = {
            'val_loss': loss,
            'val_kl_loss': kl_loss,
            'val_recons_loss': recons_loss
        }
        self.log_dict(log_dict)
        return loss
   
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":

    def binarize(img):
        # Binarize MNIST
        img =  (img > 0.7).float()
        # Plot image
        return img


    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--cpus', type=int, default=0)
    parser.add_argument('--accelerator', type=str, default='cpu')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)

    parser.add_argument('--latent_dim', type=int, default=100)


    args = parser.parse_args()


    # Load Binarized MNIST
    transform = transforms.Compose([transforms.ToTensor(), binarize])
    train_dataset = MNIST(root='data', train=True, download=True, transform=transform)
    val_dataset = MNIST(root='data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    # Train
    model = PixelCNN_VAE_Binary(in_channels=1, out_channels=32, kernel_size=5, n_layers=5, latent_dim=args.latent_dim, lr=args.lr).float()

    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator=args.accelerator)
    trainer.fit(model, train_loader, val_loader)

    # Save model
    torch.save(model.state_dict(), 'pixelcnn.pt')