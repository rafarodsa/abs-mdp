"""
    Option Class
    author: Rafael Rodriguez-Sanchez
    date: November 2022
"""

import numpy as np


class Option:
    def __init__(self, initiation_classifier, policy_func_factory, termination_prob, max_executing_time=100, check_can_execute=True, name=''):
        self.initiation = initiation_classifier
        self.policy_factory = policy_func_factory
        self.executing = False
        self.termination_prob_factory = termination_prob
        self.termination_prob = None
        self.name = name
        self.policy = None
        self._step = 0.
        self.max_executing_time = max_executing_time
        self.check_can_execute = check_can_execute

    def execute(self, state):
        self.executing = self.initiation(state) if self.check_can_execute else True
        self.policy = self.policy_factory(state)
        self.termination_prob = self.termination_prob_factory(state)
        self._step = 0.
        return self.executing

    def is_executing(self):
        return self.executing

    def act(self, state):
        # self.executing = np.random.binomial(1, self.termination_prob(state)) == 1
        if self._step != 0: # skip first state termination
            self.executing = not(np.random.random(1) < self.termination_prob(state))
        self._step += 1
        return self.policy(state) if self.executing else None
    
    def __repr__(self) -> str:
        return self.name


