import pytorch_lightning as pl

from src.absmdp.infomax_attn import InfomaxAbstraction, AbstractMDPTrainer
from src.absmdp.mdp import AbstractMDP


from src.absmdp.datasets import PinballDataset
from omegaconf import OmegaConf as oc
import argparse
import torch
import logging
from pytorch_lightning.callbacks import ModelCheckpoint

# load the config
def save_model(trainer, save_path):
    pass

def run(cfg, ckpt=None):
    
    checkpoint_callback = ModelCheckpoint(
        monitor='nll_loss',
        dirpath=f'{cfg.save_path}/phi_train/ckpts/',
        filename='infomax-pb-{epoch:02d}-{nll_loss:.2f}',
        save_top_k=3
    )
    model = InfomaxAbstraction(cfg) 
    data = PinballDataset(cfg.data)
    # training
    trainer = pl.Trainer(
                        accelerator=cfg.accelerator,
                        devices=cfg.devices,
                        max_epochs=cfg.epochs, 
                        auto_scale_batch_size=True,
                        default_root_dir=f'{cfg.save_path}/phi_train',
                        log_every_n_steps=15,
                        callbacks=[checkpoint_callback]
                    )
    trainer.fit(model, data, ckpt_path=ckpt)
    return model

def train_mdp(cfg, ckpt):
    if ckpt is None:
        raise ValueError("Must provide an abstraction checkpoint to train MDP. Run abstraction first.")

    phi = InfomaxAbstraction.load_from_checkpoint(ckpt)
    model = AbstractMDPTrainer(cfg, phi)
    data = PinballDataset(cfg.data)
    # training
    trainer = pl.Trainer(
                        accelerator=cfg.accelerator,
                        devices=cfg.devices,
                        num_nodes=1,
                        max_epochs=cfg.epochs, 
                        auto_scale_batch_size=True,
                        default_root_dir=f'{cfg.save_path}/mdp_train',
                        log_every_n_steps=15,
                    )

    trainer.fit(model, data)
    return model


def main():
    torch.autograd.set_detect_anomaly(True)
    default_config_path = "experiments/pb_no_obs/fullstate/config/config.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=default_config_path)
    parser.add_argument('--debug', action='store_const', const=logging.DEBUG, dest='loglevel', default=logging.WARNING)
    parser.add_argument('--verbose', action='store_const', const=logging.INFO, dest='loglevel')
    parser.add_argument('--from-ckpt', type=str, default=None)


    parser.add_argument('--train-mdp', action='store_true', default=False)

    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=args.loglevel)

    cli_config = oc.from_cli(unknown)
    cfg = InfomaxAbstraction.load_config(args.config)
    cfg = oc.merge(cfg, cli_config)

    if not args.train_mdp:
        run(cfg, args.from_ckpt)
    else:
        train_mdp(cfg, args.from_ckpt)

if __name__ == "__main__":
    main()