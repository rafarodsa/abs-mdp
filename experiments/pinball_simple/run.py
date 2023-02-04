import pytorch_lightning as pl
from src.abstract_mdp.abs_mdp_vae import AbstractMDPTrainer


# load the config
cfg = AbstractMDPTrainer.load_config("experiments/pinball_simple/config/config.yaml")
model = AbstractMDPTrainer(cfg).double() # fix this double requirement.
# training

trainer = pl.Trainer(
                    accelerator=cfg.accelerator,
                    gpus=cfg.devices if cfg.accelerator == "gpu" else None,
                    max_epochs=cfg.epochs, 
                    auto_scale_batch_size=True
                )
trainer.fit(model)