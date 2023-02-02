import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

# import model
from src.abstract_mdp.abs_mdp_vae import AbstractMDPTrainer

import argparse
import numpy as np

# callback to give Tune the metrics to optimize
metric_callback = TuneReportCallback(
                {
                    "loss": "grounding_loss",
                },
                on="validation_end")

def hyperparam_tune(config, num_epochs=10, num_gpus=0, data_dir="~/data"):
    data_dir = os.path.expanduser(data_dir)
    model = LightningMNISTClassifier(config, data_dir)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            metric_callback
        ])
    trainer.fit(model)


if __name__=='__main__':
    ## Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data/pinball_no_obstacles_rewards.pt')
    parser.add_argument('--save-path', type=str, default='mdps/abs_mdp.pt')
    parser = AbstractMDPTrainer.add_model_specific_args(parser)
    args = parser.parse_args()