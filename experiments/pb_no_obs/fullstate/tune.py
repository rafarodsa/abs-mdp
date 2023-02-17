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

from src.absmdp.trainer import AbstractMDPTrainer
from src.absmdp.datasets import PinballDataset


def prepare_config(cfg, lr, grounding_const, kl_const, transition_const, kl_balance):
    
    # manually override config
    # TODO can we generalize this?
    cfg.lr = lr
    cfg.loss.grounding_const = grounding_const
    cfg.loss.kl_const = kl_const
    cfg.loss.transition_const = transition_const
    cfg.loss.kl_balance = kl_balance
    return cfg

def objective(cfg, trial):
    # Search Space
    
    lr = trial.suggest_float('lr', 1e-5, 1e-2)
    grounding_const = trial.suggest_float('grounding_const', 0.1, 100)
    kl_const = trial.suggest_float('kl_const', 1e-4, 10)
    transition_const = trial.suggest_float('transition_const', 1e-4, 10)
    kl_balance = trial.suggest_float('kl_balance', 0., 1.)

    cfg = prepare_config(cfg, lr, grounding_const, kl_const, transition_const, kl_balance)
    model = AbstractMDPTrainer(cfg) 
    dataset = PinballDataset(cfg.data)

    trainer = pl.Trainer(
                        accelerator=cfg.accelerator,
                        gpus=cfg.devices if cfg.accelerator == "gpu" else None,
                        max_epochs=cfg.epochs, 
                        callbacks=[PyTorchLightningPruningCallback(trial, monitor="nll_loss")],
                        default_root_dir=cfg.save_path + f'/{trial.number}'
                    )
    
    trainer.logger.log_hyperparams(model.hparams)
    trainer.fit(model, dataset)
    return trainer.callback_metrics["nll_loss"].item()

def tune():
    default_config_path = "experiments/pb_no_obs/fullstate/config/config.yaml"

    ## Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=default_config_path)
    parser.add_argument('--num-trials', type=int, default=10)
    parser.add_argument('--study-name', type=str, default="pinball_simple")
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
    cfg = oc.load(args.config)
    cfg_cli = oc.from_cli()
    cfg = oc.merge(cfg, cfg_cli)

    # storage for optuna parallelization
    optuna_db_path = cfg.experiment_cwd + "/tune.db"
    storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(optuna_db_path)
        )

    study = optuna.create_study(
                        direction="minimize",
                        pruner=pruner, 
                        storage=storage,
                        study_name=args.study_name, 
                        load_if_exists=True
                    )
    if not args.best_trial:
        study.optimize(
                        partial(objective, cfg), 
                        n_trials=args.num_trials, 
                        timeout=600
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