'''
    Train binary classifier for ground truth initset
    author: Rafael Rodriguez-Sanchez
    date: 
'''
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from src.models import ModuleFactory

import pandas as pd
from experiments.antmaze.utils import OptionExecution
from typing import List, Dict
import numpy as np

import argparse
from omegaconf import OmegaConf as oc

def get_initset_data(data):
    return [(d.s, float(d.success)) for d in data]

def preprocess_data(df: pd.DataFrame) -> (List[OptionExecution], List[Dict]):
    '''
    Preprocess data for initset learning
    '''
    # each row is data from an option
    data, name_to_idx = [], {}
    for i in range(len(df)):
        d = df.iloc[i]
        data.append(get_initset_data(d['data'])) # List of Lists of (s, success)
        name_to_idx[d['name']] = i

    data = [(elem[0][0], np.array([e[-1] for e in elem])) for elem in zip(*data)]
    return data, name_to_idx    


class Initset(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = ModuleFactory.build(cfg.initset)

    def forward(self, x, opt_name=None):
        return self.model(x) if opt_name else self.model(x)[:, self.name_to_idx[opt_name]]

    def pos_weight(self, y):
        return (y.shape[0] - y.sum(0, keepdim=True)) / y.sum(0, keepdim=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.pos_weight(y))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.pos_weight(y))
        tpr = (y_hat.sigmoid() > 0.5).eq(y).float().mean()
        fpr = (y_hat.sigmoid() > 0.5).ne(y).float().mean()
        self.log_dict({"val_loss": loss, "tpr": tpr, "fpr": fpr}, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.pos_weight(y))
        tpr = (y_hat.sigmoid() > 0.5).eq(y).float().mean()
        fpr = (y_hat.sigmoid() > 0.5).ne(y).float().mean()
        self.log_dict({"test_loss": loss, "tpr": tpr, "fpr": fpr}, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.trainer.lr)

    def load_dataset(self):
        data, name_to_idx = preprocess_data(pd.read_pickle(self.cfg.data.data_path))
        self.name_to_idx = name_to_idx
        # split data
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(data, [int(self.cfg.data.train_split*len(data)), int(self.cfg.data.val_split*len(data)), int(self.cfg.data.test_split*len(data))])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.cfg.data.batch_size, shuffle=self.cfg.data.shuffle, num_workers=self.cfg.data.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.cfg.data.batch_size, shuffle=False, num_workers=self.cfg.data.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.cfg.data.batch_size, shuffle=False, num_workers=self.cfg.data.num_workers)
    

def train_initset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="experiments/antmaze/configs/ground_initset.yaml")
    args = parser.parse_args()
    cfg = oc.load(args.cfg)

    model = Initset(cfg)
    model.load_dataset()


    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{cfg.trainer.save_path}/ckpt',
        save_top_k=1,
        save_last=True,
        monitor='val_loss',
        mode='min'
    )


    trainer = pl.Trainer(
                        accelerator=cfg.trainer.accelerator,
                        devices=cfg.trainer.devices,
                        max_epochs=cfg.trainer.epochs, 
                        default_root_dir=f'{cfg.trainer.save_path}/initset_train',
                        log_every_n_steps=15,
                        callbacks=[checkpoint_callback]
                    )
    trainer.fit(model)


if __name__=='__main__':
    train_initset()