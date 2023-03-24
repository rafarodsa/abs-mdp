import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms

from src.models.vqvae import VQVAE

from src.utils.printarr import printarr
import matplotlib.pyplot as plt
import argparse 

from PIL import Image

class MNISTEncoder(nn.Module):
    def __init__(self, hidden_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_channels, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1)

        # 2 3x3 residual blocks
        self.res1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding='same'),
            nn.ReLU()
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding='same'),
            nn.ReLU()
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.res1(x) + x)
        x = self.res2(x) + x
        return x
            

class MNISTDecoder(nn.Module):
    def __init__(self, hidden_channels=128):
        super().__init__()
        # residual blocks
        self.res1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels,  kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels,  kernel_size=1, stride=1, padding='same'),
            nn.ReLU()
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels,  kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels,  kernel_size=1, stride=1, padding='same'),
            nn.ReLU()
        )

        # 2 transposed convs 4x4 
        self.conv1 = nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1)
        self.out = nn.ConvTranspose2d(hidden_channels, 1, 4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.res1(x) + x)
        x = F.relu(self.res2(x) + x)
        x = F.relu(self.conv1(x))
        # x = torch.sigmoid(self.out(x))
        x = self.out(x)
        # x = torch.tanh(self.out(x))
        return x
    

if __name__ == "__main__":

    def binarize(img):
        # Binarize MNIST
        img =  (img > 0.7).float()
        # Plot image
        # plt.imshow(img[0], cmap='gray')
        # plt.show()
        return img
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--reconstruct', action='store_true')
    args, _ = parser.parse_known_args()

    # Load Binarized MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5), transforms.Grayscale(num_output_channels=1)])
    train_dataset = MNIST(root='data', train=True, download=True, transform=transform)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
    test_dataset = MNIST(root='data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    trainer = pl.Trainer(max_epochs=100, accelerator='gpu')
    # Train
    if not args.reconstruct:
        print('Training...')
        model = VQVAE(MNISTEncoder(256), MNISTDecoder(256), codebook_size=20, embedding_dim=256, commitment_const=0.25, lr=2e-4)

        
        trainer.fit(model, train_loader, val_loader)

        # Save model
        torch.save(model.state_dict(), 'vqvae.pt')
    else:
        #load model
        model = VQVAE(MNISTEncoder(256), MNISTDecoder(256), codebook_size=8, embedding_dim=256, commitment_const=0.25, lr=2e-5)
        model.load_state_dict(torch.load('vqvae.pt'))
        model.eval()
        # test
        trainer.test(model, test_loader)

        # reconstruct
        test_batch = next(iter(test_loader))
        imgs = test_batch
        imgs, _ = imgs
        with torch.no_grad():
            recon, z, z_q = model(imgs)
    
        to_plot = torch.cat([imgs, recon], dim=-1).reshape(-1, 56).numpy()
        to_plot = Image.fromarray((to_plot * 255).astype('uint8'))
        to_plot.save('recon.png')

# reconstruct
