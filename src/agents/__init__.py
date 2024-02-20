from typing import Any
import torch

class DictTensor(dict):
    @property
    def shape(self):
        key = list(self.keys())[-1]
        return self.__getitem__(key).shape
    
    @property
    def device(self):
        key = list(self.keys())[-1]
        return self.__getitem__(key).device