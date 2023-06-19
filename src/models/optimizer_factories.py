'''
    Optimizer Factories
    author: Rafael Rodriguez-Sanchez
    date: April 2023
'''
import torch

class OptimizerFactory:
    factories ={
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "radam": torch.optim.RAdam,
        "adamw": torch.optim.AdamW,
        "linear_scheduler": torch.optim.lr_scheduler.LinearLR,
        "cosine_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
        "exponential_scheduler": torch.optim.lr_scheduler.ExponentialLR,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau
    }
    
    @staticmethod
    def build(cfg):
        return OptimizerFactory.factories[cfg.type](**cfg.params)
    
    @staticmethod
    def register(type: str, factory):
        OptimizerFactory.factories[type] = factory



def build_optimizer(cfg, model_params):
    return OptimizerFactory.factories[cfg.type](model_params, **cfg.params)

def build_scheduler(cfg, optimizer):
    return OptimizerFactory.factories[cfg.type](optimizer, **cfg.params)