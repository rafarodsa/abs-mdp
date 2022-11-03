"""
    Option Class
    author: Rafael Rodriguez-Sanchez
    date: November 2022
"""

import numpy as np


class Option:
    def __init__(self, initiation_classifier, policy_func_factory, termination_prob, name=''):
        self.initiation = initiation_classifier
        self.policy = policy_func_factory
        self.executing = False
        self.termination_prob = termination_prob
        self.name = name
    
    def execute(self, state):
        self.executing = self.initiation(state)
        return self.executing

    def is_executing(self):
        return self.executing

    def act(self, state):
        self.executing = np.random.binomial(1, self.termination_prob(state)) == 1
        return self.policy(state) if self.executing else None
    
    def __repr__(self) -> str:
        return self.name


