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


import matplotlib.pyplot as plt

from src.models.pixelcnn import PixelCNNDecoderBinary

if __name__ == "__main__":
    # Sample Pixel CNN
    def sample(model, label, n_samples=1):
        model.eval()
        with torch.no_grad():
            x = torch.zeros(n_samples, 1, 28, 28)
            for i in range(28):
                for j in range(28):
                    out = model(x, label)
                    probs = F.softmax(out, dim=1)
                    x[:, :, i, j] = torch.bernoulli(probs[:, 1, i, j]).reshape(n_samples, -1)
        return x
    

    # load model
    model = PixelCNNDecoderBinary(10, 1, 32, 5, 5)
    model.load_state_dict(torch.load('pixelcnn.pt'))

    # Sample 4 images of number 7
    samples = sample(model, 7+torch.zeros(4).long(), 4)


    print(torch.max(samples))
    # Plot
    for img in range(4):
        plt.subplot(2, 2, img+1)
        plt.imshow(samples[img, 0]*255, cmap='gray')

    plt.show()


    # TODO: Test Color Channel Masking on GPU