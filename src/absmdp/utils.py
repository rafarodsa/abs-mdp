import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class CyclicalKLAnnealing(Callback):

    def __init__(self, num_cycles=4, rate=0.5):
        self.num_cycles = num_cycles
        self.rate = rate

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.current_t = self.current_t % self.iter_per_cycle
        if self.current_t < self.peak_time:
            kl_const = self.linear_rate(self.current_t)
        else:
            kl_const = self.max_kl_const
        pl_module.kl_const = kl_const
        self.current_t += 1
    
    def linear_rate(self, t):
        return self.max_kl_const * (t / self.peak_time)

    def setup(self, trainer, pl_module, stage=None):
        self.max_kl_const = pl_module.hyperparams.kl_const
        self.n_iterations = trainer.estimated_stepping_batches
        self.iter_per_cycle = self.n_iterations // self.num_cycles
        self.current_t = 0
        self.peak_time = int(self.iter_per_cycle * self.rate)
        self.trainer = trainer
        self.pl_module = pl_module
        self.pl_module.kl_const = 0.
        print(f'Peak time: {self.peak_time}, iter per cycle: {self.iter_per_cycle}, n_iterations: {self.n_iterations}')
    

class Ratio:
    def __init__(self, ratio):
        self.ratio = ratio
        self.prev_t = None
    
    def __call__(self, t):
        if self.ratio == 0:
            return 0
        if self.prev_t is None:
            self.prev_t = t
            return 1
        repeat = int((t-self.prev_t) * self.ratio)
        self.prev_t += repeat / self.ratio
        return repeat

class Every:
    def __init__(self, every):
        self.every = every
        self.prev_t = None
    
    def __call__(self, t):
        if self.prev_t is None:
            self.prev_t = t
            return True
        
        if t - self.prev_t >= self.every:
            self.prev_t = t
            return True
        return False
