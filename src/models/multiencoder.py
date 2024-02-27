import torch
import torch.nn as nn
import numpy as np
import re
from src.models.factories import build_distribution, build_model, ModuleFactory

class MultiEncoder(nn.Module):
    CONVERSION = {
        np.floating: np.float32,
        np.signedinteger: np.int64,
        np.uint8: np.uint8,
        bool: bool,
    }

    def __init__(self, cfg):
        super().__init__()

        self.cnn_encoder = build_model(cfg.cnn)
        self.mlp_encoder = build_model(cfg.mlp)
        self.output_mlp = build_model(cfg.final_mlp)
        
        self.cnn_pattern = cfg.cnn.obs_keys
        self.mlp_pattern = cfg.mlp.obs_keys
        self.initialized = False
        self.mlp_keys = []
        self.cnn_keys = []
        self._register_load_state_dict_pre_hook(self.load_state_hook, with_module=True)

    def forward(self, obs):
        assert isinstance(obs, dict), f'Observation is a {type(obs)}'

        cnn_in, mlp_in = self._get_inputs(obs)
        z_mlp = self.mlp_encoder(mlp_in)
        z_cnn = self.cnn_encoder(cnn_in)
        return self.output_mlp(torch.cat([z_cnn, z_mlp], dim=-1))
    
    def _get_inputs(self, obs):
        if not self.initialized:
            self.mlp_keys = []
            self.cnn_keys = []
            for k, v in obs.items():
                if re.match(self.cnn_pattern, k) and 'log' not in k:
                    if isinstance(v, (np.ndarray, torch.Tensor)) and len(v.shape) == 3:
                        self.cnn_keys.append(k)
                    else:
                        print(f'Ignoring {k} for CNN inputs because is != 3 dims')
                elif re.match(self.mlp_pattern, k) and k not in ('is_first', 'is_terminal', 'is_last', 'log_coverage', 'log_global_pos'):
                    if isinstance(v, (np.ndarray, torch.Tensor)) and len(v.shape) in (0, 1, 2) or isinstance(v, (float, int, bool)):
                        self.mlp_keys.append(k)
                    else:
                        print(f'Ignoring {k} for MLP inputs')
                else:
                    print(f'Ignoring {k}')
            print(f'MLP keys {self.mlp_keys}')
            print(f'CNN keys {self.cnn_keys}')
            self.initialized=True

        mlp_inputs = [self.convert(v) for k, v in obs.items() if k in self.mlp_keys]
        cnn_inputs = [self.preprocess_image(self.convert(v)) for k, v in obs.items() if k in self.cnn_keys]
        
        if len(mlp_inputs) == 1:
            mlp_inputs = mlp_inputs[0]
        else:
            try:
                mlp_inputs = torch.cat(mlp_inputs, axis=-1)
            except:
                import ipdb; ipdb.set_trace()
        if len(cnn_inputs) == 1:
            cnn_inputs = cnn_inputs[0]
        else:
            cnn_inputs = torch.cat(cnn_inputs, axis=-3)
        return cnn_inputs, mlp_inputs
        # return torch.from_numpy(cnn_inputs.copy()), torch.from_numpy(mlp_inputs.copy())


    def preprocess_image(self, value):
        assert torch.all(torch.logical_and(value <= 255, value >= 0)), f'{value.min()}, {value.max()}'
        value = value.float() / 255 - 0.5 
        value = value + torch.randn_like(value) * 1./255./5 
        value = torch.clamp(value, -0.5, 0.5)
        n_dims = len(value.shape)
        batch_dims = n_dims - 3
        value = value.permute(list(range(batch_dims)) + [2+batch_dims, batch_dims, 1+batch_dims]) # make channel dim first.
        return value


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dict[f'{prefix}mlp_keys'] = self.mlp_keys
        state_dict[f'{prefix}cnn_keys'] = self.cnn_keys
        return state_dict
    
    # def load_state_dict(self, state_dict, strict=True, assign=False):
        
    #     if 'mlp_keys' in state_dict and 'cnn_keys' in state_dict:
    #         self.mlp_keys = state_dict['mlp_keys']
    #         self.cnn_keys = state_dict['cnn_keys']
    #         self.initialized = True

    #         del state_dict['mlp_keys']
    #         del state_dict['cnn_keys']
        
    #     super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def load_state_hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if f'{prefix}mlp_keys' in state_dict and '{prefix}cnn_keys':
            module.mlp_keys = state_dict[f'{prefix}mlp_keys']
            module.cnn_keys = state_dict[f'{prefix}cnn_keys']
            module.initialized = True
            del state_dict[f'{prefix}mlp_keys']
            del state_dict[f'{prefix}cnn_keys']
    
    def convert(self, value):
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        if isinstance(value, torch.Tensor):
            return value.float()
        else:
            raise ValueError(f'Unknown data type!')


class MultiCNNEncoder(MultiEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        cnn_encoders = [self.cnn_encoder]
        for _ in range(cfg.n_cnns-1):
            cnn_encoders.append(build_model(cfg.cnn))
        self.cnn_encoder = nn.ModuleList(cnn_encoders)
    
    def _get_inputs(self, obs):
        if not self.initialized:
            self.mlp_keys = []
            self.cnn_keys = []
            for k, v in obs.items():
                if re.match(self.cnn_pattern, k) and 'log' not in k:
                    if isinstance(v, (np.ndarray, torch.Tensor)) and len(v.shape) == 3:
                        self.cnn_keys.append(k)
                    else:
                        print(f'Ignoring {k} for CNN inputs because is < 3 dims')
                elif re.match(self.mlp_pattern, k) and k not in ('is_first', 'is_terminal', 'is_last', 'log_coverage', 'log_global_pos'):
                    if isinstance(v, (np.ndarray, torch.Tensor)) and len(v.shape) in (0, 1, 2) or isinstance(v, (float, int, bool)):
                        self.mlp_keys.append(k)
                    else:
                        print(f'Ignoring {k} for MLP inputs')
                else:
                    print(f'Ignoring {k}')
            print(f'MLP keys {self.mlp_keys}')
            print(f'CNN keys {self.cnn_keys}')
            self.initialized=True

        mlp_inputs = [self.convert(v) for k, v in obs.items() if k in self.mlp_keys]
        cnn_inputs = [self.preprocess_image(self.convert(v)) for k, v in obs.items() if k in self.cnn_keys]
        
        if len(mlp_inputs) == 1:
            mlp_inputs = mlp_inputs[0]
        else:
            try:
                mlp_inputs = torch.cat(mlp_inputs, axis=-1)
            except:
                import ipdb; ipdb.set_trace()
        return cnn_inputs, mlp_inputs
    
    def forward(self, obs):
        assert isinstance(obs, dict), f'Observation is a {type(obs)}'

        cnn_in, mlp_in = self._get_inputs(obs)

        assert len(cnn_in) == len(self.cnn_encoder)
        z_cnn = torch.cat([cnn(_in) for cnn, _in in zip(self.cnn_encoder, cnn_in)], dim=-1)
        z_mlp = self.mlp_encoder(mlp_in)
        return self.output_mlp(torch.cat([z_cnn, z_mlp], dim=-1))

def build_multiencoder(cfg):
    cfg.cnn.outdim = cfg.outdim // 2
    cfg.mlp.output_dim = cfg.outdim - cfg.cnn.outdim
    return MultiEncoder(cfg)

def build_multicnnencoder(cfg):
    cfg.final_mlp.input_dim = cfg.mlp.output_dim + cfg.cnn.outdim * cfg.n_cnns
    return MultiCNNEncoder(cfg)


ModuleFactory.register('multicnnencoder', build_multicnnencoder)