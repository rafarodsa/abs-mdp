import numpy as np
from dataclasses import dataclass


@dataclass
class OptionExecution:
    s: np.ndarray
    next_state: np.ndarray
    goal: np.ndarray
    success: bool
    reward: np.ndarray
    steps: int
