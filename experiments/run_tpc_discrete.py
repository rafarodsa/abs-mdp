import lightning as pl

from src.absmdp.infomax_attn import AbstractMDPTrainer
from src.absmdp.discrete_tpc_critic import DiscreteInfoNCEAbstraction as TPCAbstraction

from src.absmdp.datasets import PinballDataset
from omegaconf import OmegaConf as oc
import argparse
import torch, random, numpy as np
import logging
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import yaml, os

def set_seeds(seed):
    print(f'Seed set to {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)

def parse_oc_args(oc_args):
    assert len(oc_args)%2==0
    oc_args = ['='.join([oc_args[i].split('--')[-1], oc_args[i+1]]) for i in range(len(oc_args)) if i%2==0]
    cli_config = oc.from_cli(oc_args)
    return cli_config


# load the config
def run(cfg, ckpt=None, args=None):
    # torch.set_float32_matmul_precision('medium')
    set_seeds(cfg.seed)
    save_path = f'{cfg.save_path}/{args.tag}'
    os.makedirs(save_path, exist_ok=True)
    cfg.save_path = save_path
    checkpoint_callback = ModelCheckpoint(
        monitor='val_nll',
        dirpath=f'{save_path}/phi_train/ckpts/',
        filename='infomax-pb-{epoch:02d}-{val_nll:.2f}',
        save_top_k=3,
        save_last=True
    )

    logger = TensorBoardLogger(
        save_dir=f'{save_path}/phi_train/logs/',
        name='infomax-pb',
    )

    csv_logger = CSVLogger(
        save_dir=f'{save_path}/phi_train/csv_logs/',
        name='infomax-pb',
    )    

    
    cfg.data.save_path = save_path

    model = torch.compile(TPCAbstraction(cfg))
    data = PinballDataset(cfg.data)
    
    torch._dynamo.config.verbose=True
    
    # training
    trainer = pl.Trainer(
                        accelerator=cfg.accelerator,
                        devices=cfg.devices,
                        num_nodes=args.num_nodes,
                        strategy=args.strategy,
                        max_epochs=cfg.epochs, 
                        default_root_dir=f'{save_path}/phi_train',
                        log_every_n_steps=15,
                        callbacks=[checkpoint_callback], 
                        logger=[logger, csv_logger],
                        detect_anomaly=False, 
                    )
    trainer.fit(model, data, ckpt_path=ckpt)
    test_results = trainer.test(model, data)
    yaml.dump(test_results, open(f'{save_path}/phi_train/test_results.yaml', 'w'))
    return model

def train_mdp(cfg, ckpt, args):
    if ckpt is None:
        raise ValueError("Must provide an abstraction checkpoint to train MDP. Run abstraction first.")

    phi = TPCAbstraction.load_from_checkpoint(ckpt)
    model = AbstractMDPTrainer(cfg, phi)
    data = PinballDataset(cfg.data)
    # training
    trainer = pl.Trainer(
                        accelerator=cfg.accelerator,
                        devices=cfg.devices,
                        num_nodes=args.num_nodes,
                        strategy=args.strategy,
                        max_epochs=cfg.epochs, 
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
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--tag', type=str, default='')

    parser.add_argument('--train-mdp', action='store_true', default=False)

    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=args.loglevel)

    cli_cfg = parse_oc_args(unknown)
    cfg = TPCAbstraction.load_config(args.config)
    cfg = oc.merge(cfg, cli_cfg)

    if not args.train_mdp:
        run(cfg, args.from_ckpt, args)
    else:
        train_mdp(cfg, args.from_ckpt, args)

if __name__ == "__main__":
    main()