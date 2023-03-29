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
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils.printarr import printarr
import src.absmdp.configs

import matplotlib.pyplot as plt


def load_states(debug):
    '''
        Load states from debug dict for PinballDataset
        return: s, next_s (np.ndarray)
    '''
    latent_states = debug['latent_states']
    s = np.array(list(map(lambda x: x['state'], latent_states))).astype(np.float32)
    next_s = np.array(list(map(lambda x: x['next_state'], latent_states))).astype(np.float32)
    return s, next_s


def save_images_as_one(samples, obs):
    _obs = [obs[i] for i in range(obs.shape[0])]
    _samples = [samples[i] for i in range(samples.shape[0])]
    _obs = np.concatenate(_obs, axis=1)
    _samples = np.concatenate(_samples, axis=1)
    return np.concatenate([_obs, _samples], axis=0)

class TestDataset(PinballDataset_):
    MEAN, STD = 0.716529905796051, 0.3007461726665497
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = self.load_debug()
        self.s, self.next_s = load_states(self.debug)
        self._data = list(zip(self.data, self.s, self.next_s))
        n = len(self.data)
        self.data = self.data[:n]

    def load_debug(self):
        with zipfile.ZipFile(self.zfile_name, 'r') as z:
            with z.open('debug.pt', 'r') as f:
                debug = torch.load(f)
        return debug

    def __getitem__(self, idx):

        datum = super().__getitem__(idx)
        s = datum.info['state'].astype(np.float32)
        noise = torch.randn_like(datum.obs) * 2 / 255
        obs = (datum.obs - self.MEAN)/self.STD 
        # obs = (datum.obs - self.mean)/self.std
        obs = datum.obs
        idx = datum.obs
        return (torch.clamp(obs+noise, min=0, max=1), s, idx)#, datum.next_obs, self.next_s[idx])
    

logger = logging.getLogger(__name__)


class PinballDecoderModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.deconv = DeconvBlock(cfg.features)
        self.decoder = PixelCNNDecoder(self.deconv, cfg.dist)
        self.lr = cfg.lr
        self.save_hyperparameters()

    def forward(self, obs, s):
        return self.decoder(obs, s)

    def _run_step(self, obs, s, idx):
        s = s[..., :2]
        # loss = -self.decoder.log_prob(obs, s).mean()
        mean, std = obs.mean(0).detach(), obs.std(0).detach()
        # _obs = (obs - mean)/(std + 1e-12)
        _obs = obs
        out = self.decoder(_obs, s)
        log_probs = F.log_softmax(out, dim=1)
        idx = (_obs * (self.decoder.color_levels-1)).long()
        log_probs = -F.nll_loss(log_probs, idx, reduction='none').reshape(obs.shape[0], -1).sum(-1)
        return -log_probs.mean()

    def training_step(self, batch, batch_idx):
        obs, s, idx = batch
        loss = self._run_step(obs, s, idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        obs, s, idx = batch
        loss = self._run_step(obs, s, idx)
        self.log('val_loss', loss,prog_bar=True)
        logger.info(f'Validation Loss: {loss}')
        return loss
    
    def test_step(self, batch, batch_idx):
        obs, s, idx = batch
        loss = self._run_step(obs, s, idx)
        self.log('test_loss', loss, prog_bar=True)
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
        self.log('val_loss', loss, prog_bar=True)
        logger.info(f'Validation Loss: {loss}')
        return loss
    
    def test_step(self, batch, batch_idx):
        obs, s = batch
        loss = self._run_step(obs, s)
        self.log('test_loss', loss,prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        print(self.lr)
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class PixelCNNAutoencoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        cfg.model.decoder.lr = cfg.lr
        cfg.model.encoder.lr = cfg.lr
        self.lr = cfg.lr
        self.decoder = PinballDecoderModel(cfg.model.decoder)
        self.encoder = PinballEncoderModel(cfg.model.encoder)
        self.save_hyperparameters()

    def forward(self, obs):
        z = self.encoder(obs)
        out_logits = self.decoder(obs, z)
        return out_logits

    def _run_step(self, obs):
        z = self.encoder(obs)
        return -self.decoder.decoder.log_prob(obs, z).mean()

    def training_step(self, batch, batch_idx):
        obs, s, idx = batch
        loss = self._run_step(obs)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        obs, s, idx = batch
        loss = self._run_step(obs)
        self.log('val_loss', loss,prog_bar=True)
        logger.info(f'Validation Loss: {loss}')
        return loss
    
    def test_step(self, batch, batch_idx):
        obs, s, idx = batch
        loss = self._run_step(obs)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        print(self.lr)
        return torch.optim.Adam(self.parameters(), lr=self.lr)
     
def load_cfg(unknown_args, path='experiments/pb_obstacles/pixel/config/config.yaml'):
    cfg = oc.load(path)
    cli_config = oc.from_cli(unknown_args)
    cfg = oc.merge(cfg, cli_config)
    return cfg

def grayscale(obs):
    return 0.2125 * obs[:, 0] + 0.7154 * obs[:, 1] + 0.0721 * obs[:, 2]

if __name__=='__main__':
    torch.set_float32_matmul_precision('medium')
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
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--n-samples', type=int, default=2)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--strategy', type=str, default=None)
    parser.add_argument('-n', type=int, default=1)
    args, unknown_args = parser.parse_known_args()

    torch.manual_seed(args.seed)

    cfg = load_cfg(unknown_args, args.config)
    save_path = f'{args.save_path}/{args.tag}' if args.tag is not None else args.save_path

    data = TestDataset(path_to_file=cfg.data.data_path, obs_type=cfg.data.obs_type, transforms=[])

    print(f'Loaded dataset with {len(data)} samples.')
    # randomly split dataset into train and test
    train_set, val_test, test_set = random_split(data, [0.7, 0.2, 0.1])

    # create dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_test, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    

    # random choice of images to sanity check
    
    batch = next(iter(val_loader))
    choice = np.random.choice(batch[0].shape[0], 3)
    grid_size = 50
    coords = (batch[1][choice, :2] * grid_size).byte().numpy()


    obs = (batch[0][choice] * 255).numpy().astype(np.uint8)[:, 0] # remove channel dim

    # plot histogram of grayscale values
    print(f'Pixel values: {torch.from_numpy(obs).reshape(-1).unique(sorted=True)}')

    obs[np.arange(3), coords[:, 1], coords[:, 0]] = 0
    print(obs.shape)
    obs = obs.reshape(-1, obs.shape[-1])
    # make and save image
    print(coords)
    img = Image.fromarray(obs)
    samples_path = f'{save_path}/samples'
    os.makedirs(samples_path, exist_ok=True)
    img.save(f'{samples_path}/sanity.png')
    
    # create model
    if args.model == 'decoder':
        model_cfg = cfg.model.decoder
        model_cfg.lr = args.lr
        model = PinballDecoderModel(model_cfg)
    elif args.model == 'encoder':
        model_cfg = cfg.model.encoder
        model_cfg.lr = args.lr
        model = PinballEncoderModel(model_cfg)
    elif args.model == 'autoencoder':
        oc.resolve(cfg)
        model = PixelCNNAutoencoder(cfg)
    else:
        raise ValueError(f'{args.model} is not known')


    if args.from_ckpt is not None:
        print(f'Loading checkpoint at {args.from_ckpt}')
    # train model
   

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'{save_path}/ckpts/',
        filename='pixelcnn-pb-{epoch:02d}-{val_loss:.2f}'+f'_{args.model}',
        save_top_k=3
    )

    trainer = pl.Trainer(
                        accelerator=args.accelerator, 
                        devices=args.devices,
                        num_nodes=args.num_nodes, 
                        max_epochs=args.epochs,
                        callbacks=[checkpoint_callback],
                        # strategy=args.strategy,
                        # gradient_clip_val=1.0,
                        detect_anomaly=True,
                        # track_grad_norm=2,
                        # overfit_batches=5
                    )
    
    if not args.inference:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.from_ckpt)
    
        # save model
        os.makedirs(save_path, exist_ok=True)
        
        trainer.save_checkpoint(f'{save_path}/pinball_{args.model}.ckpt')
        print(f'Checkpoint saved at {save_path}/pinball_{args.model}.ckpt')
        torch.save(model.state_dict(), f'{save_path}/pinball_{args.model}.pt')
        print(f'Model params saved at {save_path}/pinball_{args.model}.pt')
        # TODO: this is failing to resolve.
        # oc.save(oc.resolve(cfg), f'{args.save_path}/config.yaml')
        # print(f'Model config saved at {args.save_path}/config.yaml')

        trainer.test(model, test_loader)

    else:
        if args.from_ckpt is None:
            model.load_state_dict(torch.load(f'{save_path}/pinball_{args.model}.pt'))
        else:
            model = model.load_from_checkpoint(args.from_ckpt, cfg=model_cfg)
            
            print(f'Checkpoint loaded from {args.from_ckpt}')
        model.eval()
        # sample
        device = torch.device('cuda')
        model = model.to(device)
        decoder = model.decoder
        
        trainer.test(model, test_loader)

        obs, states, idx = list(zip(*list(test_set)))
        states = np.vstack(states)
        obs = np.array(list(map(lambda x: x.numpy(), obs)))*255
        choice = np.random.choice(states.shape[0], args.n_samples)
        
        _obs = obs[choice]
        s_obs = _obs.astype(np.uint8)[:,0] 
        # s_obs = obs[choice].astype(np.uint8).transpose((0,2,3,1))
        # if s_obs.shape[-1] == 1:
            # s_obs = s_obs[..., 0]
            
        samples_path = f'{save_path}/samples'
        os.makedirs(samples_path, exist_ok=True)
        for i in range(s_obs.shape[0]):
            img = Image.fromarray(s_obs[i])
            img.save(f'{samples_path}/obs_{i}.png')
            print(f'Ground obs {i} save as {samples_path}/obs_{i}.png')

        print(f'Sampling... {args.n_samples} samples')
        with torch.no_grad():
            _obs = torch.from_numpy(_obs/255).to(device)
            if args.model == 'decoder':
                h = torch.from_numpy(states[choice, :2]).to(device)
                s = model.to(device).decoder.sample(h)
                

            elif args.model == 'autoencoder':
                h = model.to(device).encoder(_obs)
                s = model.to(device).decoder.decoder.sample(h)
                probs_samples = torch.softmax(model.to(device).decoder(s, h), dim=1)
                probs_truth = torch.softmax(model.to(device).decoder(_obs, h), dim=1)

                max_probs_samples = probs_samples.max(dim=1).values.reshape(-1, 50)
                max_probs_truth = probs_truth.max(dim=1).values.reshape(-1, 50)
                max_probs = torch.cat([max_probs_samples, max_probs_truth], dim=-1)
                
                # select 'correct' probs
                selected_probs_samples = probs_samples.gather(dim=1, index=(_obs * 255).long().unsqueeze(1)).reshape(-1, 50)
                selected_probs_obs = probs_truth.gather(dim=1, index=(_obs * 255).long().unsqueeze(1)).reshape(-1, 50)
                selected_probs = torch.cat([selected_probs_samples, selected_probs_obs], dim=-1)
                plt.figure()
                im = plt.imshow(selected_probs.cpu().numpy(), cmap='hot', interpolation='nearest')
                plt.colorbar(im)
                plt.savefig('selected_probs.png')

                printarr(max_probs, selected_probs)
                # Image.fromarray((max_probs.cpu().numpy() * 255).astype(np.uint8)).save('probs.png')
                # heatmax
                plt.figure()
                im = plt.imshow(max_probs.cpu().numpy(), cmap='hot', interpolation='nearest')
                plt.colorbar(im)
                plt.savefig('max_probs.png')
                

                # plot histogram
                plt.figure()
                ball_indices = (states[choice, :2] * 50).astype(np.int64)
                ball_probs_sample = probs_samples.cpu()[0, :, 0, ball_indices[0, 1], ball_indices[0, 0]]
                ball_probs_obs = probs_truth.cpu()[0, :, 0, ball_indices[0, 1], ball_indices[0, 0]]
                plt.subplot(2,2,1)
                plt.bar(np.arange(255), ball_probs_sample)
                plt.title('sample dist')
                plt.subplot(2,2,2)
                plt.bar(np.arange(255), ball_probs_obs)
                plt.title('obs dist')
                
                ball_probs_sample = probs_samples.cpu()[0, :, 0, ball_indices[0, 1], ball_indices[0, 0]+1]
                ball_probs_obs = probs_truth.cpu()[0, :, 0, ball_indices[0, 1], ball_indices[0, 0]+1]
                plt.subplot(2,2,3)
                plt.bar(np.arange(255), ball_probs_sample)
                plt.subplot(2,2,4)
                plt.bar(np.arange(255), ball_probs_obs)

                plt.savefig('ball_histograms.png')


        
        s = s/(cfg.model.decoder.dist.color_levels-1) * 255
        imgs = s.cpu().permute(0,2,3,1).numpy().astype(np.uint8)
        printarr(s, s_obs)
        if imgs.shape[-1] == 1:
            imgs = imgs[..., 0]
        for i in range(imgs.shape[0]):
            img = Image.fromarray(imgs[i])
            img.save(f'{samples_path}/image_{i}.png')
            print(f'Sample {i} save as {samples_path}/image_{i}.png')

        img = save_images_as_one(s.cpu().numpy()[:,0], s_obs).astype(np.uint8)
        printarr(img)
        img = Image.fromarray(img)
        img.save(f'{samples_path}/sample.png')