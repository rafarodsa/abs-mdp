"""
    PixelCNN Decoder implementation.
    
    author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
    date: January 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms

from tqdm import tqdm
import matplotlib.pyplot as plt

from tests.pixelcnn.pixelcnn_mnist_train import PixelCNNDecoderBinary

if __name__ == "__main__":
    # Sample Pixel CNN
    def sample(model, label, n_samples=1):
        model.eval()
        with torch.no_grad():
            x = torch.zeros(n_samples, 1, 28, 28)
            for i in tqdm(range(28)):
                for j in range(28):
                    out = model(x, label)
                    probs = F.softmax(out[:, :, i, j], dim=1)
                    x[:, :, i, j] = torch.bernoulli(probs[:, 1]).reshape(n_samples, -1)
        return x
    

    # load model
    model = PixelCNNDecoderBinary(n_classes=10, in_channels=1, out_channels=64, kernel_size=7, n_layers=5)
    # model.load_state_dict(torch.load('pixelcnn.pt'))
    path = 'lightning_logs/version_50/checkpoints/epoch=2-step=1407.ckpt'
    model = model.load_from_checkpoint(path, n_classes=10, in_channels=1, out_channels=64, kernel_size=7, n_layers=5)

    # Sample 4 images of number 7
    samples = sample(model, torch.tensor([0, 2, 3, 4]), 4)


    print(torch.max(samples))
    # Plot
    for img in range(4):
        plt.subplot(2, 2, img+1)
        plt.imshow(samples[img, 0], cmap='gray')

    plt.show()


    # TODO: Test Color Channel Masking on GPU