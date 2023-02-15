'''
    Train initiation set classifier & Duration model
    author: Rafael Rodriguez-Sanchez (rrs@brown.edu)
    date: February 2023
'''
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf as oc
import argparse
from src.absmdp.datasets import InitiationDataset
from src.options.initiation_classifier import InitsetModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='scripts/configs/initset.yaml')
    parser.add_argument('--dataset', type=str, default='experiments/pb_obstacles/fullstate/data/obstacles.pt')
    parser.add_argument('--save-path', type=str, default='experiments/pb_obstacles/initset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args, unknown = parser.parse_known_args()

    cli_config = oc.from_cli(unknown)
    cfg = oc.load(args.config)
    cfg = oc.merge(cfg, cli_config)

    dataset = InitiationDataset(args.dataset, cfg.data)
    model = InitsetModel(cfg)
    trainer = pl.Trainer(accelerator=cfg.accelerator, max_epochs=args.epochs)
    trainer.fit(model, dataset)
    
    # save model
    trainer.save_checkpoint(f"{args.save_path}/initset.ckpt")

    # save model params
    torch.save(model.state_dict(), f"{args.save_path}/initset.pt")




