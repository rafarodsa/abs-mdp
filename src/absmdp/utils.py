import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


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

class IBScheduler(Callback):

    def __init__(self, warmup_epochs):
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
    
    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch >= self.warmup_epochs:
            pl_module.kl_const += (self.epoch - self.warmup_epochs) * self.rate
        self.epoch += 1

    def setup(self, trainer, pl_module, stage=None):
        self.max_kl_const = 1.
        self.total_epochs = trainer.max_epochs
        self.rate = self.max_kl_const / (trainer.max_epochs - self.warmup_epochs) 
        pl_module.kl_const = 0.
        print(f'Max epochs {self.total_epochs}, Warmup period {self.warmup_epochs}')