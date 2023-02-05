import pytorch_lightning as pl
from src.abstract_mdp.abs_mdp_vae import AbstractMDPTrainer

from omegaconf import OmegaConf as oc
import argparse


# load the config
def run(cfg):
    
    model = AbstractMDPTrainer(cfg) 
    # training
    trainer = pl.Trainer(
                        accelerator=cfg.accelerator,
                        gpus=cfg.devices if cfg.accelerator == "gpu" else None,
                        max_epochs=cfg.epochs, 
                        auto_scale_batch_size=True
                    )
    trainer.fit(model)
    return model

def main():
    default_config_path = "experiments/pinball_simple/config/config.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=default_config_path)
    args, _ = parser.parse_known_args()

    cli_config = oc.from_cli()
    cfg = AbstractMDPTrainer.load_config(args.config)
    cfg = oc.merge(cfg, cli_config)
    run(cfg)

if __name__ == "__main__":
    main()