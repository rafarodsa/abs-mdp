import os
import torch, numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F

from src.models import PixelCNNDecoder, DeconvBlock, ResidualConvEncoder

import argparse
from omegaconf import OmegaConf as oc

import logging
from PIL import Image

from src.absmdp.datasets import PinballDataset_, PinballDataset
import zipfile

def load_states(debug):
    '''
        Load states from debug dict for PinballDataset
        return: s, next_s (np.ndarray)
    '''
    latent_states = debug['latent_states']
    s = np.array(list(map(lambda x: x['state'], latent_states))).astype(np.float32)
    next_s = np.array(list(map(lambda x: x['next_state'], latent_states))).astype(np.float32)
    return s, next_s

class TestDataset(PinballDataset_):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = self.load_debug()
        self.s, self.next_s = load_states(self.debug)

    def load_debug(self):
        with zipfile.ZipFile(self.zfile_name, 'r') as z:
            with z.open('debug.pt', 'r') as f:
                debug = torch.load(f)
        return debug


    def __getitem__(self, idx):
        datum = super().__getitem__(idx)
        return (datum.obs, self.s[idx])#, datum.next_obs, self.next_s[idx])


logger = logging.getLogger(__name__)

class PinballDecoderModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.deconv = DeconvBlock(cfg.features)
        self.decoder = PixelCNNDecoder(self.deconv, cfg.dist)
        self.lr = cfg.lr
        
    def forward(self, obs, s):
        # print(obs.shape, s.shape)
        # cond = self.deconv(s)
        return self.decoder(obs, s)

    def _run_step(self, obs, s):
        out = self.forward(obs, s)
        idx = (obs * 255).long()
        loss = F.cross_entropy(out.reshape(obs.shape[0], 256, -1), idx.reshape(obs.shape[0], -1), reduction='sum')/obs.shape[0]
        # log_probs = F.log_softmax(out, dim=1)
        # log_probs = log_probs.gather(1, idx.unsqueeze(1).long()).squeeze(1)
        # loss = -log_probs.reshape(x.shape[0], -1).sum(1).mean()
        # assert torch.allclose(loss, _loss)
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
        logger.info(f'Validation Loss: {loss}')
        return loss
    
    def test_step(self, batch, batch_idx):
        obs, s = batch
        loss = self._run_step(obs, s)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class PinballEncoderModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ResidualConvEncoder(cfg.features)
        self.lr = cfg.lr
        
    def forward(self, obs):
        # print(obs.shape, s.shape)
        # cond = self.deconv(s)
        return self.encoder(obs)

    def _run_step(self, obs, s):
        s = s[..., :2]
        out = self.forward(obs)
        loss = F.mse_loss(out, s)
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
        logger.info(f'Validation Loss: {loss}')
        return loss
    
    def test_step(self, batch, batch_idx):
        obs, s = batch
        loss = self._run_step(obs, s)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        
def load_cfg(path='experiments/pb_obstacles/pixel/config/config.yaml'):
    cfg = oc.load(path)
    return cfg



def grayscale(obs):
    return 0.2125 * obs[:, 0] + 0.7154 * obs[:, 1] + 0.0721 * obs[:, 2]

if __name__=='__main__':


    default_dataset = 'experiments/pb_obstacles/pixel/data/obstacles.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--dataset', type=str, default=default_dataset)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save-path', type=str, default='experiments/pb_obstacles/pixel/checkpoints')
    parser.add_argument('--config', type=str, default='experiments/pb_obstacles/pixel/config/config.yaml')
    parser.add_argument('--from-ckpt', type=str, default=None)
    parser.add_argument('--inference', action='store_true', default=False)
    parser.add_argument('--grayscale', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='decoder')
    parser.add_argument('--n-samples', type=int, default=2)
    parser.add_argument('-n', type=int, default=1)
    args, _ = parser.parse_known_args()

    cfg = load_cfg(args.config)
    

    data = TestDataset(path_to_file=cfg.data.data_path, obs_type=cfg.data.obs_type, transforms=[])

    print(f'Loaded dataset with {len(data)} samples.')
    # randomly split dataset into train and test
    train_set, val_test, test_set = random_split(data, [0.7, 0.2, 0.1])

    # create dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_test, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    
    # create model
    if args.model == 'decoder':
        model_cfg = cfg.model.decoder
        model_cfg.lr = cfg.lr
        model = PinballDecoderModel(model_cfg)
    else:
        model_cfg = cfg.model.encoder
        model_cfg.lr = cfg.lr
        model = PinballEncoderModel(model_cfg)

    if args.from_ckpt is not None:
        print(f'Loading checkpoint at {args.from_ckpt}')
    # train model
    if not args.inference:
        trainer = pl.Trainer(
                            accelerator=args.accelerator, 
                            gpus=1 if args.accelerator=='gpu' else None, 
                            max_epochs=args.epochs,
                        )

        trainer.fit(model, train_loader, val_loader, ckpt_path=args.from_ckpt)

        # save model
        os.makedirs(args.save_path, exist_ok=True)
        
        trainer.save_checkpoint(f'{args.save_path}/pinball.ckpt')
        print(f'Checkpoint saved at {args.save_path}/pinball.ckpt')
        torch.save(model.state_dict(), f'{args.save_path}/pinball.pt')
        print(f'Model params saved at {args.save_path}/pinball.pt')
        # TODO: this is failing to resolve.
        # oc.save(oc.resolve(cfg), f'{args.save_path}/config.yaml')
        # print(f'Model config saved at {args.save_path}/config.yaml')
    else:
        if args.from_ckpt is None:
            model.load_state_dict(torch.load(f'{args.save_path}/pinball.pt'))
        else:
            model.load_from_checkpoint(args.from_ckpt, cfg=model_cfg)
        
        # sample
        device = torch.device('cuda')
        model.to(device)
        decoder = model.decoder
        
        obs, states = list(zip(*list(test_set)))
        states = np.vstack(states)
        obs = np.array(obs)*255
        choice = np.random.choice(states.shape[0], args.n_samples)
        
        s_obs = obs[choice].astype(np.uint8).transpose((0,2,3,1))
        if s_obs.shape[-1] == 1:
            s_obs = s_obs[..., 0]
            
        samples_path = f'{args.save_path}/samples'
        os.makedirs(samples_path, exist_ok=True)
        for i in range(s_obs.shape[0]):
            img = Image.fromarray(s_obs[i])
            img.save(f'{samples_path}/obs_{i}.png')
            print(f'Ground obs {i} save as {samples_path}/obs_{i}.png')


        print(f'Sampling... {args.n_samples} samples')
        with torch.no_grad():
            h = torch.from_numpy(states[choice]).to(device)
            s = decoder.sample(h)
        
        
        imgs = s.cpu().permute(0,2,3,1).numpy().astype(np.uint8)
        if imgs.shape[-1] == 1:
            imgs = imgs[..., 0]
        for i in range(imgs.shape[0]):
            img = Image.fromarray(imgs[i])
            img.save(f'{samples_path}/image_{i}.png')
            print(f'Sample {i} save as {samples_path}/image_{i}.png')

       