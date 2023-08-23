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
from experiments.antmaze.utils import OptionExecution, parse_oc_args
from typing import List, Dict
import numpy as np

import argparse
from omegaconf import OmegaConf as oc
import yaml
from functools import reduce

from src.utils import printarr
# def get_initset_data(data):
#     return [(d.s, float(d.success)) for d in data]

# def preprocess_data(df: pd.DataFrame) -> (List[OptionExecution], List[Dict]):
#     '''
#         Preprocess data for initset learning
#     '''
#     # each row is data from an option
#     data, name_to_idx = [], {}
#     for i in range(len(df)):
#         d = df.iloc[i]
#         data.append(get_initset_data(d['data'])) # List of Lists of (s, success)
#         name_to_idx[d['name']] = i

#     data = [(elem[0][0], np.array([e[-1] for e in elem])) for elem in zip(*data)]
#     return data, name_to_idx    


def preprocess_data(df):
    '''
        Preprocess data for initset learning
        Each tuple has the form (s, a, success)
    '''
    # each row is data from an option
    data, name_to_idx = [], {}
    for i in range(len(df)):
        d = df.iloc[i]
        data.append([(_d.s.astype(np.float32), int(i), np.float32(_d.success), d.name) for _d in d['data']]) # List of Lists of (s, a_idx, success, option_name)
        name_to_idx[d['name']] = i

    data = reduce(lambda x, y: x+y, data)
    return data, name_to_idx
        


class Initset(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = ModuleFactory.build(cfg.initset)
        self.name_to_idx = {}
        self.save_hyperparameters()

    def forward(self, x, opt_name=None, opt_idx=None):
        batched = len(x.shape) > 1
        if not batched:
            x = x.unsqueeze(0)
        logits = self.model(x)
        # collect logits for options in opt_idx
        if opt_idx is not None:
            logits = torch.gather(logits, -1, opt_idx.unsqueeze(-1)).squeeze(-1)
        elif opt_name is not None:
            logits = torch.gather(logits, -1, torch.tensor([self.name_to_idx[opt_name]]).to(self.device).unsqueeze(-1)).squeeze(-1)
        if not batched:
            logits = logits.squeeze(0)
        return logits

    def pos_weight(self, y):
        return (y.shape[0] - y.sum(0, keepdim=True)) / y.sum(0, keepdim=True)

    def training_step(self, batch, batch_idx):
        x, idx, y, _ = batch
        y_hat = self(x, opt_idx=idx)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.pos_weight(y))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def compute_stats(self, y, y_hat):
        accuracy = (y_hat.sigmoid() > 0.5).eq(y).float().mean()
        tpr = ((y_hat.sigmoid() > 0.5).eq(y) * y).sum() / y.sum()
        fpr = ((y_hat.sigmoid() > 0.5).ne(y) * (1-y)).sum() / (1-y).sum()
        return accuracy, tpr, fpr

    def validation_step(self, batch, batch_idx):
        x, idx, y, _ = batch
        y_hat = self(x, opt_idx=idx)
        accuracy, tpr, fpr = self.compute_stats(y, y_hat)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.pos_weight(y))
        self.log_dict({"val_loss": loss, "tpr": tpr, "fpr": fpr, 'acc': accuracy}, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, idx, y, _ = batch
        y_hat = self(x, opt_idx=idx)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.pos_weight(y))
        accuracy, tpr, fpr = self.compute_stats(y, y_hat)
        test_ = {"test_loss": float(loss), "tpr": float(tpr), "fpr": float(fpr), 'acc': float(accuracy)}
        self.log_dict(test_, on_epoch=True, prog_bar=True)
        yaml.dump(test_, open(f"{self.cfg.trainer.save_path}/initset_test.yaml", "w"))
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
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.cfg.data.batch_size, shuffle=self.cfg.data.shuffle, num_workers=self.cfg.data.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.cfg.data.batch_size, shuffle=False, num_workers=self.cfg.data.num_workers)
    
    def on_save_checkpoint(self, checkpoint) -> None:
        "Objects to include in checkpoint file"
        checkpoint["name_to_idx"] = self.name_to_idx

    def on_load_checkpoint(self, checkpoint) -> None:
        "Objects to retrieve from checkpoint file"
        self.name_to_idx= checkpoint["name_to_idx"]
    

def train_initset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="experiments/antmaze/configs/ground_initset.yaml")
    args, unknown = parser.parse_known_args()
    cli_config = parse_oc_args(unknown)
    cfg = oc.load(args.cfg)
    cfg = oc.merge(cfg, cli_config)

    model = Initset(cfg)
    model.load_dataset()


    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{cfg.trainer.save_path}/ckpt',
        save_top_k=1,
        save_last=True,
        monitor='fpr',
        mode='min',
        filename='best'
    )


    trainer = pl.Trainer(
                        accelerator=cfg.trainer.accelerator,
                        devices=cfg.trainer.devices,
                        max_epochs=cfg.trainer.epochs, 
                        default_root_dir=f'{cfg.trainer.save_path}/initset_train',
                        log_every_n_steps=15,
                        callbacks=[checkpoint_callback],
                    )
    trainer.fit(model)
    trainer.test(model)


if __name__=='__main__':
    train_initset()