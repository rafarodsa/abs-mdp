import argparse
import os, gc
from typing import List
from typing import Optional

from functools import partial

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl

from src.abstract_mdp.abs_mdp_vae import AbstractMDPTrainer

if version.parse(pl.__version__) < version.parse("1.0.2"):
    raise RuntimeError("PyTorch Lightning>=1.0.2 is required for this example.")

def objective(trial: optuna.trial.Trial, data_path='data/pinball_no_obstacle_rewards.pt', gpus=1, cpus=1, epochs=10) -> float:

    lr = trial.suggest_float('lr', 1e-5, 1e-2)
    grounding_loss_const = trial.suggest_float('grounding_loss_const', 0.1, 100)
    kl_loss_const = trial.suggest_float('kl_loss_const', 1e-4, 10)
    transition_loss_const = trial.suggest_float('transition_loss_const', 1e-4, 10)
    
    config = dict(lr=lr, grounding_loss_const=grounding_loss_const, kl_loss_const=kl_loss_const, transition_loss_const=transition_loss_const)
    model = AbstractMDPTrainer(config, data_path).double()
    
    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=epochs,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices= gpus if gpus > 0 else cpus,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_grounding_loss")],
    )
    
    trainer.logger.log_hyperparams(model.hparams)
    trainer.fit(model)

    return trainer.callback_metrics["val_grounding_loss"].item()


if __name__ == "__main__":
    ## Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/pinball_no_obstacle_rewards.pt')
    parser.add_argument('--save-path', type=str, default='mdps/abs_mdp.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gpus', type=float, default=1)
    parser.add_argument('--cpus', type=float, default=8)
    parser.add_argument('--num-trials', type=int, default=10)
    parser = AbstractMDPTrainer.add_model_specific_args(parser)
    
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="minimize", pruner=pruner)
    _objective = partial(objective, gpus=args.gpus, cpus=args.cpus, data_path=args.data_dir, epochs=args.epochs)
    study.optimize(_objective, n_trials=args.num_trials, timeout=600, callbacks=[lambda study, trial: gc.collect()])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))