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

from tests.pixelcnn.pixelcnn_mnist_train import PixelCNNDecoderBinary, PixelCNNLogistic
from src.utils.printarr import printarr
from src.models.quantized_logistic import QuantizedLogisticMixture

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

def mix_logistic_sample(locs, log_scales, mix_logits, n_samples=1):
    '''
        locs: [batch, n_components]
        log_scales: [batch, n_components]
        mix_logits: [batch, n_components]
    '''
    # printarr(locs, log_scales, mix_logits)
    m = locs.shape[0] # batch
    scales = torch.exp(torch.clamp(log_scales, min=-7))
    probs = torch.softmax(mix_logits, dim=-1) # batch x n_components
    mix_idx = torch.multinomial(probs, n_samples)# batch x n_samples
    u = torch.rand(m, n_samples).to(probs.get_device()) # batch x n_samples x 1
    u = (1-2e-5) * u + 1e-5
    _mu = torch.gather(locs, dim=-1, index=mix_idx.long()) # batch x n_samples
    _s = torch.gather(scales, dim=-1, index=mix_idx.long())
    samples = _mu + _s * (torch.log(u) - torch.log(1.-u))
    return torch.clamp(samples, min=-1, max=1)

def sigmoid(x):
    return np.where(
            x >= 0, # condition
            1 / (1 + np.exp(-x)), # For positive values
            np.exp(x) / (1 + np.exp(x)) # For negative values
    )

def plot_quantized_logistic_mixture(locs, log_scales, mix_logits, name='logistic.png'):
    x = np.linspace(-0.999, 0.999, 255)[None, :, None]
    inv_s = torch.exp(-torch.clamp(log_scales.unsqueeze(1), min=-7)).cpu().numpy()
    mus = locs.cpu().unsqueeze(1).numpy()
    mix_log_probs = torch.log_softmax(mix_logits.unsqueeze(1), dim=-1).cpu().numpy()
    
    y = (x-mus) * inv_s
    printarr(x, inv_s, mus, mix_log_probs, y, 1/inv_s)
    

    cdf_up = sigmoid((x-mus+1/255) * inv_s)
    cdf_down = sigmoid((x-mus-1/255) * inv_s)
    # log_cdf_up = np.log(cdf_up)
    # log_cdf_down = np.log(cdf_down)
    # log_prob = logsumexp(np.concatenate([log_cdf_up, 1/log_cdf_down], axis=-1), axis=-1)
    cdf_delta = np.where(cdf_up-cdf_down > 1e-12, cdf_up-cdf_down, 1e-12)
    log_prob = np.log(cdf_delta)
    log_prob_weighted = mix_log_probs + log_prob
    prob = np.exp(logsumexp(log_prob, axis=-1))

    _x_axis = (x[0,:,0]+1)/2 * 255
    printarr(cdf_up, cdf_down, cdf_delta, log_prob, log_prob_weighted, prob)
    plt.figure()
    for i in range(mus.shape[0]):
        plt.subplot(2,2, i+1)
        plt.bar(_x_axis, prob[i]) 
    plt.savefig(name)

if __name__ == "__main__":
    # Sample Pixel CNN
    def sample(model, label, n_samples=1):
        model.eval().to('cuda')
        with torch.no_grad():
            x = torch.zeros(n_samples, 1, 28, 28).to('cuda')
            for i in tqdm(range(28)):
                for j in range(28):
                    out = model(x/255, label)
                    probs = F.softmax(out[:, :, i, j], dim=1)
                    # printarr(out, probs)
                    x[:, :, i, j] = torch.multinomial(probs, 1)
        log_probs = F.log_softmax(model(x.clone(), label), dim=1)
        nll_loss = F.nll_loss(log_probs, x.squeeze().long(), reduction='none').reshape(x.shape[0], -1).sum(-1)
        print(nll_loss.shape, log_probs.shape)
        print(nll_loss)
        return x, nll_loss
    
    def sample_logistic(model, label, n_samples=1):
        model.eval().to('cuda')
        with torch.no_grad():
            x = torch.zeros(n_samples, 1, 28, 28).to('cuda')-1
            for i in tqdm(range(28)):
                for j in range(28):
                    out = model(x, label).squeeze() # batch x channels x height x width x mix
                    locs, log_scales, mix_logits = out[..., i,j,:model.n_components], out[...,i,j, model.n_components:2*model.n_components], out[...,i,j,2*model.n_components:]
                    # printarr(out, probs)
                    x[:, :, i, j] = mix_logistic_sample(locs, log_scales, mix_logits, 1)
        
            out = model(x, label)
            locs, log_scales, mix_logits = out[..., :model.n_components], out[...,model.n_components:2*model.n_components], out[...,2*model.n_components:]
            log_probs = QuantizedLogisticMixture(locs, log_scales, mix_logits).log_prob(x)
            nll_loss = -log_probs.mean()
            print(nll_loss.shape, log_probs.shape)
            print(nll_loss)
        return (x+1)/2, nll_loss
    

    def complete_logistic(model, batch, p=0.3, n_samples=1):
        model.eval().to('cuda')
        imgs, labels = batch
        pixels = int(28 * 28 * p)
        max_y, max_x = pixels // 28, pixels % 28
        imgs = imgs[:n_samples] * 2 -1
        labels = labels[:n_samples]
        imgs = imgs.to('cuda')
        labels = labels.to('cuda')
        x = torch.zeros(n_samples, 1, 28, 28).to('cuda')+1
        x[:, :, :max_y] = imgs[:, :, :max_y]
        x[:, :, max_y, :max_x] = imgs[:, :, max_y, :max_x]
        smpls = []
        with torch.no_grad():
            for i in tqdm(range(max_y, 28)):
                for j in range(28):
                    out = model(x, labels).squeeze() # batch x channels x height x width x mix
                    locs, log_scales, mix_logits = out[...,i,j, :model.n_components], out[...,i,j,model.n_components:2*model.n_components], out[...,i,j,2*model.n_components:]
                    # printarr(out, probs)
                    y  = mix_logistic_sample(locs, log_scales, mix_logits, 1)
                    smpls.append(y)
                    x[:, :, i, j] = y

            samples = x[:,:, max_y:]                        
            printarr(x, imgs, samples)
            # print(smpls)
            out = model(x, labels)#.unsqueeze(1)
            locs, log_scales, mix_logits = out[..., :model.n_components], out[...,model.n_components:2*model.n_components], out[...,2*model.n_components:]
            log_probs = QuantizedLogisticMixture(locs, log_scales, mix_logits).log_prob(x)
            nll_loss = -log_probs.mean()

            out = model(imgs, labels)#.unsqueeze(1)
            locs, log_scales, mix_logits = out[..., :model.n_components], out[...,model.n_components:2*model.n_components], out[...,2*model.n_components:]
            
            i, j = 13, 13
            # printarr(locs, log_scales, mix_logits)
            plot_quantized_logistic_mixture(locs[:,0, i, j], log_scales[:,0, i, j], mix_logits[:,0, i, j])
            
            log_probs_imgs = QuantizedLogisticMixture(locs, log_scales, mix_logits).log_prob(imgs)
            nll_loss_imgs = -log_probs_imgs.mean()
        print(nll_loss, nll_loss_imgs)
        nll_loss = 0
        return (x+1)/2, nll_loss
    
    def complete(model, batch, p=0.3, n_samples=1):
        model.eval().to('cuda')
        imgs, labels = batch
        pixels = int(28 * 28 * p)
        max_y, max_x = pixels // 28, pixels % 28
        imgs = imgs[:n_samples] * 255
        labels = labels[:n_samples]
        imgs = imgs.to('cuda')
        labels = labels.to('cuda')
        x = torch.zeros(n_samples, 1, 28, 28).to('cuda')
        x[:, :, :max_y] = imgs[:, :, :max_y]
        x[:, :, max_y, :max_x] = imgs[:, :, max_y, :max_x]

        printarr(x, imgs, labels)
        with torch.no_grad():
            for i in tqdm(range(max_y, 28)):
                for j in range(28):
                    out = model(x/255, labels)
                    o_ij = out[:, :, i, j]
                    probs = F.softmax(o_ij, dim=1)
                    # printarr(o_ij, probs)
                    pixels = torch.multinomial(probs, 1)
                    x[:, :, i, j] = pixels
                                                    
        
        log_probs = F.log_softmax(model(x.clone()/255, labels), dim=1)
        log_probs_imgs = F.log_softmax(model(imgs.clone()/255, labels), dim=1)
        nll_loss = F.nll_loss(log_probs, x.squeeze().long(), reduction='none').reshape(x.shape[0], -1).sum(-1).mean()
        nll_loss_imgs = F.nll_loss(log_probs_imgs, imgs.squeeze().long(), reduction='none').reshape(x.shape[0], -1).sum(-1)
        print(nll_loss, nll_loss_imgs)
        return x, nll_loss
    

    # load model
    # model = PixelCNNDecoderBinary(n_classes=10, in_channels=1, out_channels=64, kernel_size=7, n_layers=5)
    # # model.load_state_dict(torch.load('pixelcnn.pt'))
    # path = 'lightning_logs/version_9263421/checkpoints/epoch=19-step=9380-v1.ckpt'
    # model = PixelCNNDecoderBinary.load_from_checkpoint(path, n_classes=10, in_channels=1, out_channels=128, kernel_size=7, n_layers=10)
    path = 'lightning_logs/version_9278222/checkpoints/epoch=19-step=4680.ckpt'
    model = PixelCNNLogistic.load_from_checkpoint(path, n_classes=10, in_channels=1, out_channels=256, kernel_size=3, n_layers=10)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
    val_dataset = MNIST(root='data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    batch = next(iter(val_loader))
    img, label = batch
    img, label = img.to('cuda'), label.to('cuda')
   
    loss = model.to('cuda').validation_step((img, label), 0)
    # complete_samples = complete(model, batch, n_samples=4)
    complete_samples = complete_logistic(model, batch, n_samples=4)
    
    print(f'Val Loss {loss.item()}')

    # samples, loss = sample(model, torch.tensor([0, 2, 3, 4]).to('cuda'), 4)
    samples, loss = sample_logistic(model, torch.tensor([0, 2, 3, 4]).to('cuda'), 4)
    # printarr(samples)
    # Plot
    plt.figure()
    samples = samples.cpu()
    for img in range(4):
        plt.subplot(2, 2, img+1)
        plt.imshow(samples[img, 0], cmap='gray')

    plt.savefig('./sample.png')

    samples = complete_samples[0].cpu()
    plt.figure()
    for img in range(4):
        plt.subplot(2, 2, img+1)
        plt.imshow(samples[img, 0], cmap='gray')

    plt.savefig('./inpainting.png')