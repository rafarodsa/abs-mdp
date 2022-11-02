"""
    Option Class
    author: Rafael Rodriguez-Sanchez
    date: November 2022
"""

import numpy as np


class Option:
    def __init__(self, initiation_classifier, policy_func, termination_prob, name=''):
        self.initiation = initiation_classifier
        self.policy = policy_func
        self.termination_prob = termination_prob
        self.name = name
    
    def is_executable(self, state):
        return self.initiation(state)

    def act(self, state):
        return self.policy(state), np.random.binomial(1, self.termination_prob(state)) == 1
    
    def __repr__(self) -> str:
        return self.name


