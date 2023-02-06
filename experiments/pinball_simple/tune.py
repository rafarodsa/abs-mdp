import argparse
from omegaconf import OmegaConf as oc


from typing import List
from typing import Optional

from functools import partial

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl

from src.abstract_mdp.abs_mdp_vae import AbstractMDPTrainer

def objective(cfg, trial):
    # Search Space
    lr = trial.suggest_float('lr', 1e-5, 1e-2)
    grounding_const = trial.suggest_float('grounding_loss_const', 0.1, 100)
    kl_const = trial.suggest_float('kl_loss_const', 1e-4, 10)
    transition_const = trial.suggest_float('transition_loss_const', 1e-4, 10)

    # manually override config
    # TODO can we generalize this?
    cfg.lr = lr
    cfg.loss.grounding_const = grounding_const
    cfg.loss.kl_const = kl_const
    cfg.loss.transition_const = transition_const

    model = AbstractMDPTrainer(cfg) 

    trainer = pl.Trainer(
                        accelerator=cfg.accelerator,
                        gpus=cfg.devices if cfg.accelerator == "gpu" else None,
                        max_epochs=cfg.epochs, 
                        auto_scale_batch_size=True,
                        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_NLL")]
                    )
    
    trainer.logger.log_hyperparams(model.hparams)
    trainer.fit(model)
    return trainer.callback_metrics["val_NLL"].item()

def tune():
    default_config_path = "experiments/pinball_simple/config/config.yaml"

    ## Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=default_config_path)
    parser.add_argument('--num-trials', type=int, default=10)
    parser.add_argument('--study-name', type=str, default="pinball_simple")
    
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args, _ = parser.parse_known_args()
    
    
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )
    
    # allow to overwrite config from command line
    cfg = oc.load(args.config)
    cfg_cli = oc.from_cli()
    cfg = oc.merge(cfg, cfg_cli)

    # storage for optuna parallelization
    optuna_db_path = "experiments/pinball_simple/tune.db"
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

    study.optimize(
                    partial(objective, cfg), 
                    n_trials=args.num_trials, 
                    timeout=600
                )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    tune()