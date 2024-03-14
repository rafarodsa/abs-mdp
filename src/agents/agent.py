class Agent:
    def __init__(self, cfg):
        pass

    def act(self, obs):
        '''
            Act in ground env
        '''
        raise NotImplemented
    
    def evaluate(self):
        '''
            Evaluate agent in ground enviroment
        '''
        pass
        
    def train(self, world_model, steps):
        '''
            Rollout in abstract model
        '''
        pass

