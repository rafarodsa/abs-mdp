import torch, numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F

from src.models import PixelCNNDecoder, DeconvBlock

import argparse
from omegaconf import OmegaConf as oc


class PinballModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.deconv = DeconvBlock(cfg.features)
        self.decoder = PixelCNNDecoder(self.deconv, cfg.dist)
        self.lr = cfg.lr


    def forward(self, obs, s):
        return self.decoder(obs, s)

    def _run_step(self, obs, s):
        
        log_probs = F.log_softmax(self.forward(obs, s), dim=1)
        log_probs = log_probs.gather(1, obs.unsqueeze(1).long()).squeeze(1)
        loss = -log_probs.reshape(obs.shape[0], -1).sum(1).mean()
        return loss

    def training_step(self, batch, batch_idx):
        obs, s = batch
        loss = self._run_step(obs, s)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        obs, s = batch
        loss = self._run_step(obs, s)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        obs, s = batch
        loss = self._run_step(obs, s)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        
def load_decoder_cfg(path='experiments/pb_obstacles/pixel/config/config.yaml'):
    cfg = oc.load(path)
    return cfg

def load_states(debug):
    '''
        Load states from debug dict for PinballDataset
        return: s, next_s (np.ndarray)
    '''
    latent_states = debug['latent_states']
    s = np.array(list(map(lambda x: x['state'], latent_states)))
    next_s = np.array(list(map(lambda x: x['next_state'], latent_states)))
    return s, next_s

if __name__=='__main__':


    default_dataset = 'experiments/pb_obstacles/pixel/data/obstacles.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--dataset', type=str, default=default_dataset)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save-path', type=str, default='experiments/pb_obstacles/pixel/checkpoints')
    parser.add_argument('--config', type=str, default='experiments/pb_obstacles/pixel/config/config.yaml')
    args = parser.parse_args()

    # load dataset.
    dataset, _ = torch.load(args.dataset)
    debug = torch.load(args.dataset.replace('.pt', '_debug.pt'))
    obs, _, next_obs, _, _, _, _ = zip(*dataset)
    obs = np.vstack([obs, next_obs]).transpose((0, 3, 1, 2)).astype(np.float32)
    s, next_s = load_states(debug)
    states = np.vstack([s, next_s]).astype(np.float32)[:, :2] # only use x, y
    data = list(zip(list(obs), list(states)))
    
    print(f'Loaded dataset with {obs.shape[0]} samples.')
    # randomly split dataset into train and test
    train_set, val_test, test_set = random_split(data, [0.7, 0.2, 0.1])

    # create dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_test, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # create model
    cfg = load_decoder_cfg()
    decoder_cfg = cfg.model.decoder
    decoder_cfg.lr = cfg.lr

    model = PinballModel(decoder_cfg)

    # train model
    trainer = pl.Trainer(accelerator=args.accelerator, gpus=1 if args.accelerator=='gpu' else None, max_epochs=10)

    trainer.fit(model, train_loader, val_loader)

    # save model
    trainer.save_checkpoint(args.save_path)
    torch.save(model.state_dict(), args.save_path)