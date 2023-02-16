import pytorch_lightning as pl
from src.absmdp.trainer_iaf import AbstractMDPTrainer
from src.absmdp.datasets import PinballDataset
from omegaconf import OmegaConf as oc
import argparse

import logging
# load the config
def run(cfg):
    
    model = AbstractMDPTrainer(cfg) 
    data = PinballDataset(cfg.data)
    # training
    trainer = pl.Trainer(
                        accelerator=cfg.accelerator,
                        gpus=cfg.devices if cfg.accelerator == "gpu" else None,
                        max_epochs=cfg.epochs, 
                        auto_scale_batch_size=True,
                        default_root_dir=cfg.save_path
                    )
    trainer.fit(model, data)
    return model

def main():
    default_config_path = "experiments/pb_no_obs/fullstate/config/config.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=default_config_path)
    parser.add_argument('--debug', action='store_const', const=logging.DEBUG, dest='loglevel', default=logging.WARNING)
    parser.add_argument('--verbose', action='store_const', const=logging.INFO, dest='loglevel')
    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=args.loglevel)

    cli_config = oc.from_cli(unknown)
    cfg = AbstractMDPTrainer.load_config(args.config)
    cfg = oc.merge(cfg, cli_config)
    run(cfg)

if __name__ == "__main__":
    main()