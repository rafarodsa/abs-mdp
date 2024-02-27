
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as pl


from src.models.factories import build_distribution, build_model
from src.models.optimizer_factories import build_optimizer, build_scheduler
from src.utils.symlog import symlog
from src.absmdp.configs import TrainerConfig

from omegaconf import OmegaConf as oc

import logging

from src.utils.printarr import printarr
from lightning.pytorch.utilities import grad_norm

from src.models.simnorm import SimNorm
import jax
from dm_control.utils.transformations import quat_rotate


def logit(z, V=8):
    shape = z.shape
    batch_dims = len(shape[:-1])
    z = z.reshape(*shape[:-1], -1, V) if batch_dims > 0 else z.reshape(-1, V)
    _lgt = torch.log((z + 1e-8)/(1-z + 1e-8))
    return _lgt.reshape(*shape)


class S2SAbstraction(pl.LightningModule):
    INITSET_CLASS_THRESH = 0.5

    def __init__(self, cfg: TrainerConfig):
        super().__init__()
        # oc.resolve(cfg)
        model_cfg = oc.masked_copy(cfg, 'model') 
        oc.resolve(model_cfg)
        cfg.model = model_cfg
        oc.update(cfg, 'model', model_cfg.model)
        self.save_hyperparameters()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.obs_dim = cfg.model.obs_dims
        self.latent_dim = cfg.model.latent_dim
        self.n_options = cfg.model.n_options
        self.encoder = build_model(cfg.model.encoder.features)
        self.transition = build_distribution(cfg.model.transition)
        cfg.model.init_class.output_dim = 3
        self.initsets = build_model(cfg.model.init_class)
        
        self.abstract_state = build_model(cfg.model.abstract_state_mlp)
        self.critic = nn.ModuleDict({'encoder': build_model(cfg.model.critic.encoder), 'mlp': build_model(cfg.model.critic.mlp)})
        
        

        cfg.model.init_class.output_dim = 4
        self.position_regressor = build_model(cfg.model.init_class)

        self.tau = build_model(cfg.model.tau)
        self.reward_fn = build_model(cfg.model.reward)


        self.world_model = nn.ModuleList([self.encoder, self.transition, self.initsets, self.abstract_state, self.critic, self.reward_fn, self.tau])
        self.lr = cfg.optimizer.params.lr
        self.hyperparams = cfg.loss
        self.kl_const =  self.hyperparams.kl_const
        self.alpha = 0.01
        # self.update_target(1.) # hard update

        self.initialized = False
        self.recurrent = cfg.model.transition.features.type=='rssm'

        self.automatic_optimization=False

    def forward(self, state, action):
        z = self.encoder(state)
        t_in = torch.cat([z, action], dim=-1)
        next_z = self.transition.distribution(t_in).mean + z
        return next_z
    
    def _get_device(self, x):
        d = x.get_device()
        return 'cpu' if d < 0 else f'cuda:{d}'

    def critic_forward(self, state, abstract_state):
        z = self.critic['encoder'](state)
        return self.critic['mlp'](torch.cat([z, abstract_state], dim=-1))
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)
    
    def _run_step(self, s, a, next_s, reward, duration, masks, success):
        if not self.initialized and isinstance(s, dict):
            with torch.no_grad():
                self.encoder({k: v[0,0] for k, v in s.items()})
                self.critic['encoder'].mlp_keys = self.encoder.mlp_keys
                self.critic['encoder'].cnn_keys = self.encoder.cnn_keys
                self.critic['encoder'].initialized = True

        # sample encoding of (s, s') and add noise
        masks = masks.bool()
        a = torch.nn.functional.one_hot(a.long(), self.cfg.model.n_options)
        z = self.encoder(s)
        z_c = z
        # with torch.no_grad():
        #     next_z_c = self.target(next_s)
        next_z_c = self.encoder(next_s)
        noise_std = 0.
        z = z + torch.randn_like(z) * noise_std
        next_z  =  next_z_c + torch.randn_like(z) * noise_std 

        next_z_dist = self.transition.distribution(torch.cat([z, a], dim=-1))
        # next_z_sample = next_z_dist.sample()[0] + z


        # _mask = torch.arange(length).to(self._get_device(s)).repeat(batch_size, 1) < lengths.unsqueeze(1)
        _mask = masks
        lengths = masks.sum(-1)
        # grounding_loss = self.grounding_loss(next_z[_mask], next_s[_mask])
        # grounding_loss = self.grounding_loss_normal(next_z[_mask], next_z[_mask])

        abstract_state = self.abstract_state(torch.cat([z, a, next_z], dim=-1))
        
        grounding_loss = self.grounding_loss(abstract_state, next_s, _mask)
        reward_loss = self.reward_loss(reward[_mask], z_c[_mask], a[_mask], next_z_c[_mask])
        tau_loss = self.duration_loss(duration[_mask], z_c[_mask], a[_mask])

        
        transition_loss = self.consistency_loss(z, next_z, next_z_dist, mask=_mask)
        tpc_loss = self.tpc_loss(z, next_z, next_z_dist, min_length=lengths.min())

        initset_loss = self.initset_loss_from_executed(z, a, success, _mask)        

        return grounding_loss.mean(), transition_loss.mean(), tpc_loss.mean(), initset_loss.mean(), reward_loss.mean(), tau_loss.mean()

    def update_target(self, alpha):
        '''
            alpha = 0 -> the target is not updated
            alpha = 1 -> hard update
        '''
        for param_q, param_k in zip(self.encoder.parameters(), self.target.parameters()):
                param_k.data.mul_(1-alpha).add_(alpha * param_q.detach().data)

    def initset_loss(self, z, initset_target):
        pos_samples = (initset_target==1).float()
        n_pos = pos_samples.sum(0)
        n_neg = initset_target.shape[0] - n_pos
        
        pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)
        # printarr(pos_samples, n_pos, n_neg, pos_weight)
        initset_pred = self.initsets(z)
        initset_loss = F.binary_cross_entropy_with_logits(initset_pred, initset_target, reduction='none', pos_weight=pos_weight).mean(-1)
        return initset_loss


    def initset_loss_from_executed(self, z, action, executed, mask):
        _act_idx = action.argmax(-1, keepdim=True)
        initset_pred = torch.gather(self.initsets(z), -1, _act_idx).squeeze(-1)
        n_pos = executed.sum(-1, keepdim=True)
        n_neg = executed.shape[1] - n_pos
        if torch.all(n_neg==0):
            pos_weight = torch.tensor(1.)
        else:
            pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)

        # printarr(initset_pred, executed, pos_weight, z, action)
        initset_loss = (F.binary_cross_entropy_with_logits(initset_pred, executed, pos_weight=pos_weight, reduction='none') * mask).sum(-1)
        return initset_loss


    def step(self, batch, batch_idx):
        (s, a, reward, next_s, duration, success, done, masks), info = batch
        grounding_loss, transition_loss, tpc_loss, initset_loss, reward_loss, tau_loss = self._run_step(s, a, next_s, reward, duration, masks, success)

        loss = self.hyperparams.grounding_const * grounding_loss + \
                self.hyperparams.transition_const * transition_loss + \
                self.hyperparams.tpc_const * tpc_loss + \
                self.hyperparams.initset_const * initset_loss + \
                self.hyperparams.reward_const * reward_loss + \
                self.hyperparams.tau_const * tau_loss 

        

        # log std deviations for encoder.
        logs = {
            'grounding_loss': grounding_loss,
            'transition_loss': transition_loss,
            'tpc_loss': tpc_loss,
            'initset_loss': initset_loss,
            'loss': loss,

        }
        # logger.debug(f'Losses: {logs}')
        return loss, logs
	
    def consistency_loss(self, z, next_z, next_z_dist, mask, executed=None):
        '''
            -log T(z'|z, a) 
        '''
        pos_weight = torch.tensor(1.)
        if executed is not None:
            n_pos = executed.sum(-1, keepdim=True)
            n_neg = executed.shape[1] - n_pos
            if torch.any(n_neg==0):
                pos_weight = torch.where(n_pos > 0, n_neg / n_pos, 1.)

        return -(next_z_dist.log_prob(next_z-z) * mask * pos_weight).sum(-1) / mask.sum(-1)
        # return -(next_z_dist.log_prob(logit(next_z)) * mask * pos_weight).sum(-1) / mask.sum(-1)
	
    # def grounding_loss(self, next_z, next_s):
    #     '''
    #         -MI(s'; \phi(s')) 
    #     '''
    #     b = next_z.shape[:-1]
    #     b_size = np.prod(b)
    #     s_shape = next_s.shape[1:]
    #     next_z, next_s = next_z.reshape(b_size, -1), next_s.reshape(b_size, *s_shape)
    #     _next_s = next_s.repeat(b_size, *[1 for _ in range(len(next_s.shape)-1)])
    #     _next_z = torch.repeat_interleave(next_z, b_size, dim=0)
    #     _log_t = torch.tanh(self.grounding(_next_s, _next_z).reshape(b_size, b_size)) * 100 #* np.log(b) * 0.5
    #     _loss = torch.diag(_log_t) - (torch.logsumexp(_log_t, dim=-1) - np.log(b_size))

    #     return -_loss.reshape(*b)

    def grounding_loss(self, abstract_state, next_s, mask):
        '''
            -MI(s'; \bar{s})) 
        '''

        b = abstract_state.shape[:-1]
        b_size = np.prod(b)
        next_z = self.critic['encoder'](next_s)
        next_z, next_s = next_z.reshape(b_size, -1), abstract_state.reshape(b_size, -1)
        _next_s = next_s.repeat(b_size, *[1 for _ in range(len(next_s.shape)-1)])
        _next_z = torch.repeat_interleave(next_z, b_size, dim=0)
        _inp = torch.cat((_next_s, _next_z), dim=-1)
        _log_t = (self.critic['mlp'](_inp).reshape(b_size, b_size)) * 100
        _loss = torch.diag(_log_t) - (torch.logsumexp(_log_t, dim=-1) - np.log(b_size))

        return -(_loss.reshape(*b) * mask).sum() / mask.sum()
    

    def grounding_loss_normal(self, next_z, next_z_t, std=0.1):
        '''
            critic f(s', z') = exp(||\phi(s')-z'||^2/std^2)
        '''
        b = next_z.shape[:-1]
        b_size = np.prod(b)
        # print(b_size)
        next_z = next_z.reshape(int(b_size), -1)

        _norm = ((next_z[:, None, :] - next_z_t[None, :, :]) / std) ** 2 
        _norm = -_norm.sum(-1) # b_size x b_size
        _loss =  torch.diag(_norm) - (torch.logsumexp(_norm, dim=-1) - np.log(b_size))
        return -_loss.reshape(*b)
    

    def tpc_loss(self, z, next_z, next_z_dist, min_length):
        '''
            -MI(z'; z, a)
        '''
        b, z_dim = next_z.shape[:-1], next_z.shape[-1]
        b_size = np.prod(b)
        n_traj, length = b[0], b[1]
        _next_z = torch.repeat_interleave(next_z, n_traj, dim=0)
        _z = z.repeat(n_traj, 1, 1)
        _log_t = next_z_dist.log_prob(_next_z-_z, batched=True).reshape(n_traj, n_traj, length)[..., :min_length]
        # _log_t = next_z_dist.log_prob(logit(_next_z), batched=True).reshape(n_traj, n_traj, length)[..., :min_length]
        _diag = torch.diagonal(_log_t).T 
        _loss = _diag - (torch.logsumexp(_log_t, dim=1) - np.log(n_traj))  # n_traj x length

        return -_loss.sum(-1)/ min_length
    
    def reward_loss(self, r_target, z, a, next_z):
        '''
            MSE(R, R_pred)
        '''
        r = symlog(r_target)
        r_pred = self.reward_fn(torch.cat([z, a, next_z], dim=-1).detach()).squeeze()
        loss = F.mse_loss(r.reshape(-1), r_pred.reshape(-1), reduction='none')
        return loss

    def duration_loss(self, tau_target, z, a):
        tau = torch.log(tau_target)
        t = self.tau(torch.cat([z, a], dim=-1).detach()).squeeze()
        loss = F.mse_loss(tau.reshape(-1), t.reshape(-1), reduction='none')

        return loss

    def training_step(self, batch, batch_idx):
        # self.update_target(self.alpha)
        optimizer, debug_optimizer = self.optimizers()
        wm_loss, wm_logs = self.step(batch, batch_idx)
        optimizer.zero_grad()
        self.manual_backward(wm_loss)
        self.clip_gradients(optimizer, gradient_clip_val=1., gradient_clip_algorithm="norm")
        optimizer.step()
        self.log_dict({f"train_{k}": v for k, v in wm_logs.items()}, on_step=True, on_epoch=False, logger=True)
        # regressor
        debug_loss, debug_logs = self.regressor_loss(batch)
        debug_optimizer.zero_grad()
        self.manual_backward(debug_loss)
        debug_optimizer.step()
        self.log_dict({f"train_{k}": v for k, v in debug_logs.items()}, on_step=False, on_epoch=True, logger=True)

        return wm_loss

    def get_device(self, s):
        return s.get_device() if s.get_device() >= 0 else 'cpu'

    def validation_step(self, batch, batch_idx):
        (s, a, reward, next_s, duration, success, done, masks), info = batch
        # assert torch.all(executed) # check all samples are successful executions.
        grounding_loss, transition_loss, tpc_loss, initset_loss, reward_loss, tau_loss = self._run_step(s, a, next_s, reward, duration, masks.bool(), success)    
        debug_loss, _ = self.regressor_loss(batch)
        
        # batch_size, length = s.shape[0], s.shape[1]
        # _mask = torch.arange(length).to(self.get_device(s)).repeat(batch_size, 1) < lengths.unsqueeze(1)
        _mask = masks.bool()

        z = self.encoder(s)
        next_z_r = self.encoder(next_s)
        a = torch.nn.functional.one_hot(a.long(), self.cfg.model.n_options)
        t_in = torch.cat([z, a], dim=-1)
        next_z = self.transition.distribution(t_in).mean + z
        # next_z = self.transition.distribution(t_in).mean


        # nll_loss = self.grounding_loss(next_z[_mask], next_s[_mask]).mean()
        nll_loss = self.grounding_loss_normal(next_z[_mask], next_z[_mask])

        logs = {
                'val_infomax': grounding_loss.mean(),
                'val_transition': transition_loss.mean(),
                'val_tpc_loss': tpc_loss.mean(),
                'val_initset_loss': initset_loss.mean(),
                'val_reward_loss': reward_loss.mean(), 
                'val_tau_loss': tau_loss.mean(),
                'val_nll_loss': nll_loss.mean(),
                'val_norm': (z ** 2).sum(-1).mean(),
                'val_debug_loss': debug_loss.mean()
            }
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return logs['val_nll_loss']
	
    def test_step(self, batch, batch_idx):
        (s, a, reward, next_s, duration, success, done, masks), info = batch
        grounding_loss, transition_loss, tpc_loss, initset_loss, reward_loss, tau_loss = self._run_step(s, a, next_s, reward, duration, masks.bool(), success)    
        a = torch.nn.functional.one_hot(a, self.cfg.model.n_options)
        _mask = masks.bool()

        z = self.encoder(s)
        t_in = torch.cat([z, a], dim=-1)
        next_z = self.transition.distribution(t_in).mean + z

        # nll_loss = self.grounding_loss(next_z[_mask], next_s[_mask]).mean()
        nll_loss = self.grounding_loss_normal(next_z[_mask]).mean()
        self.log_dict({
                       'nll_loss': nll_loss,
                       'initset_loss': initset_loss.mean(),
                       'reward_loss': reward_loss.mean(),
                       'transition_loss': transition_loss.mean(),
                       'tau_loss': tau_loss.mean()
                       }
                       ,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return nll_loss
		
    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg.optimizer, self.world_model.parameters())
        regressor_optimizer = torch.optim.Adam(self.position_regressor.parameters(), lr=3e-4)
        return optimizer, regressor_optimizer
	
    @staticmethod
    def load_config(path):
        try:
            with open(path, "r") as f:
                #TODO add structured configs when they have settled.
                # cfg = oc.merge(oc.structured(TrainerConfig), oc.load(f))
                cfg = oc.load(f)
                return cfg
        except FileNotFoundError:
            raise ValueError(f"Could not find config file at {path}")


    def compute_orientation(self, quat):

        _quat = quat.cpu().numpy()
        x = np.array([quat_rotate(_quat[i,j], np.array([1., 0., 0.])) for i in range(_quat.shape[0]) for j in range(_quat.shape[1])]).reshape(*_quat.shape[:-1], -1)
        angle = np.arctan2(x[..., 1], x[..., 0])
        feats = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
        return torch.Tensor(feats).float().to(quat.device)

    def regressor_loss(self, batch):
        (s, action, reward, next_s, duration, success, done, masks), info = batch

        # remake info 

        info = jax.tree_map(lambda *x: torch.stack(x, dim=1), *info)
        target = torch.cat([info['log_global_pos'][..., :2].float(), self.compute_orientation(info['log_global_orientation'])], dim=-1)[masks.bool()]
        with torch.no_grad():
            z = self.encoder(next_s)[masks.bool()]
        prediction = self.position_regressor(z)

        loss = torch.nn.functional.mse_loss(prediction, target)
        logs = {'debug_regressor_loss': loss}

        return loss, logs
    

def run(cfg, ckpt=None, args=None):
    from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
    from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
    from src.absmdp.buffer import TrajectoryReplayBufferStored
    from torch.utils.data import DataLoader
    import os
    torch.set_float32_matmul_precision('medium')

    save_path = f'{args.save_path}'
    os.makedirs(save_path, exist_ok=True)
    cfg.save_path = save_path
    checkpoint_callback = ModelCheckpoint(
        monitor='val_nll_loss',
        dirpath=f'{save_path}/phi_train/ckpts/',
        filename='worldmodel-{epoch:02d}-{val_infomax:.2f}',
        save_top_k=3,
        save_last=True,
    )

    logger = TensorBoardLogger(
        save_dir=f'{save_path}/logs/',
        name='world_model',
    )

    csv_logger = CSVLogger(
        save_dir=f'{save_path}/csv_logs/',
        name='world_model',
        flush_logs_every_n_steps=25
    )    

    
    cfg.world_model.data.save_path = save_path
    model = S2SAbstraction(cfg.world_model)
    print(model)
    data = TrajectoryReplayBufferStored(int(cfg.world_model.data.buffer_size), save_path=f'{args.data_path}/data', device='gpu')
    train_data, val_data = torch.utils.data.random_split(data, [0.9, 0.1])
    # load data
    train_dataloader = DataLoader(train_data, cfg.world_model.data.batch_size, num_workers=8)
    val_dataloader = DataLoader(val_data, cfg.world_model.data.batch_size, num_workers=4)
    
    # training
    trainer = pl.Trainer(
                        accelerator=args.accelerator,
                        devices=args.devices,
                        num_nodes=args.num_nodes,
                        max_epochs=args.epochs, 
                        default_root_dir=f'{save_path}',
                        log_every_n_steps=15,
                        callbacks=[checkpoint_callback], 
                        logger=[logger, csv_logger],
                        detect_anomaly=False, 
                        overfit_batches=0.4
                    )
    trainer.fit(model, train_dataloader, val_dataloader)
    # test_results = trainer.test(model, data)
    # yaml.dump(test_results, open(f'{save_path}/phi_train/test_results.yaml', 'w'))
    return model


if __name__ == '__main__':
    import argparse
    from omegaconf import OmegaConf as oc
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/antmaze/egocentric/online_planner.yaml')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    args = parser.parse_args()

    cfg = oc.load(args.config)
    run(cfg, args=args)

