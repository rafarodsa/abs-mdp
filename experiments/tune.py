import argparse
from omegaconf import OmegaConf as oc
import os, shutil
from tqdm import tqdm
from typing import List
from typing import Optional

from functools import partial

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.absmdp.ibtrainer import AbstractMDPTrainer
from src.absmdp.datasets import PinballDataset
from src.absmdp.utils import CyclicalKLAnnealing


def prepare_config(trial, cfg, variables):
    
    for var in variables:
        name, min, max, _type = var.name, var.min, var.max, var.type
        _path = name.split('.')
        x = cfg
        if _type == 'float':
            val = trial.suggest_float(name, min, max)
            for node in _path[:-1]:
                x = x[node] # get to the parent of leave node
            x[_path[-1]] = val
        else:
            raise NotImplementedError
    return cfg

def objective(cfg, trial, variables, metrics=None):
    # Search Space
    cfg = prepare_config(trial, cfg, variables)
    model = AbstractMDPTrainer(cfg) 
    dataset = PinballDataset(cfg.data)

    trainer = pl.Trainer(
                        accelerator=cfg.accelerator,
                        gpus=cfg.devices if cfg.accelerator == "gpu" else None,
                        max_epochs=cfg.epochs, 
                        callbacks=[
                                    PyTorchLightningPruningCallback(trial, monitor="nll_loss"),
                                    EarlyStopping(monitor='nll_loss', mode='min'),
                                    CyclicalKLAnnealing()
                                ],
                        default_root_dir=cfg.save_path + f'/{trial.number}'
                    )
    
    trainer.logger.log_hyperparams(model.hparams)
    trainer.fit(model, dataset)
    return trainer.callback_metrics["nll_loss"].item()

def tune():
    default_config_path = "experiments/pb_no_obs/fullstate/config/config.yaml"

    ## Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune-config', type=str, default='./tune.yaml')
    parser.add_argument('--num-trials', type=int, default=10)
    parser.add_argument('--process', type=int, default=0)
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    parser.add_argument('--best-trial', action="store_true")

    args, _ = parser.parse_known_args()
    
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )
    
    # allow to overwrite config from command line
    tune_cfg = oc.load(args.tune_config)

    cfg = oc.load(tune_cfg.config)
    cfg_cli = oc.from_cli()
    cfg = oc.merge(cfg, cfg_cli)


    # storage for optuna parallelization
    optuna_db_path = f'{cfg.experiment_cwd}/{tune_cfg.db}'
    storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(optuna_db_path)
    )

    study = optuna.create_study(
                        direction=tune_cfg.direction,
                        pruner=pruner, 
                        storage=storage,
                        study_name=tune_cfg.study_name, 
                        load_if_exists=True
                    )
    if not args.best_trial:
        study.optimize(
                        partial(objective, cfg, variables=tune_cfg.variables), 
                        n_trials=args.num_trials
                    )
    else:
        print("Number of finished trials: {}".format(len(study.trials)))
        trial = study.best_trial
        print(f"Best trial: {trial.number}")
        

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # clean up
        clean_up(trial, cfg)

def clean_up(trial, cfg):
    print('cleaning up...')
    root, dirs, files = next(os.walk(cfg.save_path))
    for dir in tqdm(dirs):
        if dir != str(trial.number):
            shutil.rmtree(f"{root}/{dir}")
 
if __name__ == "__main__":
    tune()