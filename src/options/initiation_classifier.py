'''
    Learn the initiation classifier for a set of options
    author: Rafael Rodriguez-Sanchez
    date: February 2023
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..models.factories import build_model

class InitsetModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # build initiation classifier model
        self.initiation_class = build_model(cfg.initset)
        # build expected length model
        self.duration = build_model(cfg.duration)
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        return self.initiation_class(x)
    
    def training_step(self, batch, batch_idx):
        obs, action, executed, duration = batch
        logits = self(obs)
        loss = self.loss(logits, executed)
        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(logits, executed))
        return loss
    
    def validation_step(self, batch, batch_idx):
        obs, action, executed, duration = batch
        logits = self(obs)
        loss = self.loss(logits, executed)
       
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(logits, executed))
        return loss
    
    def test_step(self, batch, batch_idx):
        obs, action, executed, duration = batch
        logits = self(obs)
        loss = self.loss(logits, executed)
        self.log("test_loss", loss)
        self.log("test_acc", self.accuracy(logits, executed))
        return loss

    def accuracy(self, logits, target):
        with torch.no_grad():
            chosen = F.sigmoid(logits) > 0.5 
            accuracy = (chosen == target).sum()/target.shape[0]
        return accuracy
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)