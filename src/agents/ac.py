
import torch
import numpy as np

from src.models.factories import build_model
from jax import tree_map

from contextlib import contextmanager
from src.utils.symlog import symlog, symexp

from copy import deepcopy


def twohot(value, nbins, bmin=-1, bmax=1):
    '''
        value: torch.Tensor (B0,B1...,1)
        return: (B0,B1,...,nbins)
    '''
    assert bmin < bmax
    delta = 1/(nbins-1)
    nbatchdims = len(value.shape[:-1])
    bins = torch.arange(0, 1+delta/2, delta) 
    for _ in range(nbatchdims):
        bins = bins.unsqueeze(0)
    normalized_value = (value - bmin) / (bmax-bmin) # [0,1]
    normalized_value = torch.abs(normalized_value - bins) 
    twohot = (1-normalized_value / delta) * (normalized_value  < delta)
    # TODO what happens in the extremes?
    assert torch.allclose(twohot.sum(-1, keepdims=True), torch.ones_like(value)), twohot.sum(-1)
    return twohot


def compute_lambda_return(rs, dones, values, gamma=0.99, lambd=0.95):
    '''
        rs: rewards (B, H)
        dones: termination flags (B, H)
        values: value prediction for bootstrapping (B, H+1)
        return: \lambda-return_t BxT
    '''
    cont = torch.cumprod(1-dones, dim=-1)
    _, H = rs.shape
    G = torch.zeros_like(rs)
    G[..., -1] = rs[..., -1] +  cont[..., -1] * gamma * (1-lambd) * values[..., -1]
    for t in reversed(range(H-1)):
        G[..., t] = rs[..., t] + cont[..., t] * gamma * ((1-lambd) * values[..., t+1] + lambd * G[..., t+1])
    return G


class EMA:
    def __init__(self, alpha, initialization):
        self.alpha = alpha
        self._mean = initialization
    def __call__(self, data):
        self._mean = tree_map(lambda current, target: self.alpha * current + (1-self.alpha) * target, self._mean, data)
    def get_value(self):
        return self._mean
    


class DistributionalCritic:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nbins = cfg.nbins
        self.bmin = cfg.bmin
        self.bmax = cfg.bmax
        self.delta = (self.bmax - self.bmin) / (self.nbins - 1)
        self.bins = torch.arange(self.bmin, self.bmax+self.delta/2, self.delta)
        self.reg_const = cfg.reg_const
        self.gamma = cfg.gamma
        self.lambd = cfg.lambd
        self.twohot = lambda values: twohot(values, self.nbins, self.bmin, self.bmax)
        assert self.bmin < self.bmax
        self.slow_critic = build_model(cfg.critic)
        self.fast_critic = build_model(cfg.critic)
        self.optimizer = torch.optim.Adam(self.fast_critic.parameters(), lr=cfg.lr, eps=1e-5)
        self.param_updater = EMA(cfg.slow_ema_decay, self.slow_critic.state_dict())

    def logprobs(self, states):
        return torch.nn.functional.log_softmax(self.fast_critic(states), dim=-1)

    def __call__(self, states):
       '''
            return predicted value
       '''
       return symexp(self.compute_values(states, self.fast_critic))
    
    def compute_values(self, states, critic):
        nbatchdims = len(states.shape[:-1])
        for _ in range(nbatchdims):
            bins = self.bins.unsqueeze(0)
        probs = torch.softmax(critic(states), dim=-1)
        values = (probs * bins).sum(-1)
        return values

    def loss(self, states, rewards, dones):
        '''
            states: torch.Tensor (B, H+1, D)
            rewards: torch.Tensor (B, H, 1)
            dones: torch.Tensor(B, H)
            return: (loss, return, value)

        '''        
        # compute values v(s)
        
        values = self.compute_values(states, self.fast_critic).detach()

        # compute target returns G^lambda & 2hot encode y_t
        returns = compute_lambda_return(rewards, dones, values, self.gamma, self.lambd)
        y_t = self.twohot(symlog(returns.unsqueeze(-1))).detach()

        # compute logprobs of bins ln P(b|s_t)
        logprobs = self.logprobs(states)
        # regularizer
        # regularizer: (1) cross-entropy: slow_y_t^T ln P(b|s_t) or (2) -log P(slow_v(s_t)|s_t)
        # choosing option (2)
        slow_values = self.twohot(self.compute_values(states, self.slow_critic).unsqueeze(-1)).detach()
        reg = -(slow_values * logprobs).sum(-1)

        # loss = -y_t^T ln P(b_i|s_t) 
        loss = (-y_t * logprobs[:, :-1]).sum(-1).sum(-1) + (self.reg_const * reg).sum(-1)

        return loss.mean(), returns, symexp(values)
    
    def update_critic(self):
        with torch.no_grad():
            self.param_updater(deepcopy(self.fast_critic.state_dict()))
            self.slow_critic.load_state_dict(self.param_updater.get_value())

class Actor:

    def __init__(self, cfg):
        self.policy = build_model(cfg.actor)
        # batch statistics
        self.perclow = EMA(cfg.normalizer_ema_decay, 0.)
        self.perchigh = EMA(cfg.normalizer_ema_decay, 0.)

    def logpi(self, obs):
        pass

    def loss(self, states, actions, returns, values):
        '''

        
        '''
        pass
        # values v(s_t)
        # compute returns G^lambda_t & Normalize
        # Adv_t = G^\lambda_t - V_t
        # compute ln\pi(a_t|s_t) * adv_t
        # compute entropy 

class DistributionalActorCritic:

    def __init__(self, model_cfg):

        self.training = True
        # build model
        # initialize critic to predict zeros
        self.critic = DistributionalCritic(model_cfg)
        self.actor = build_model(model_cfg.actor)

        # stats
        # self.return_percentiles = ??


    def setup(self, outdir):
        pass
        # output logger
        # setup optimizers

    def act(self, obs, encoder):
        pass

    def act_in_imagination(self, obs, initset=None):
        pass

    def compute_metrics(self, episodes):
        logs = []
        for ep in episodes:
            obss, actions, rewards, next_obss, dones, infos = zip(*ep)
            ep = np.array(obss + next_obss[-1:])
            norm2 = (ep ** 2).sum(-1, keepdims=True)
            residual = ((ep[1:]-ep[:-1])**2).sum(-1, keepdims=True)


            logs.append({
                'sim_rollout/norm2_mean': norm2.mean(),
                'sim_rollout/norm2_std': norm2.std(),
                'sim_rollout/norm2_max': norm2.max(),
                'sim_rollout/norm2_min': norm2.min(),
                'sim_rollout/residual_norm2_mean': residual.mean(),
                'sim_rollout/residual_norm2_std': residual.std(),
                'sim_rollout/residual_norm2_max': residual.max(),
                'sim_rollout/residual_norm2_min': residual.min(),
            })  
        stats = tree_map(lambda *ep_stats: np.median(np.array(ep_stats)), *logs)
        stats['n_episodes'] = len(episodes)
        return stats
    
    def compute_eval_stats(self, episodes):
        scores, lengths, exec_time, discounted_scores = [], [], [], []
        for ep in episodes:
            obss, actions, rewards, next_obss, dones, infos = zip(*ep)
            scores.append(sum(rewards))
            lengths.append(len(actions))
            taus = [info['tau'] for info in infos]
            discounts =  self.agent.gamma ** np.cumsum([0] + taus)
            discounted_rewards = np.array(rewards) * discounts[:-1]
            exec_time.append(sum(taus))
            discounted_scores.append(float(discounted_rewards.sum()))
        
        print(f'Evaluation: mean return {sum(scores) / len(scores)}, mean length {sum(lengths) / len(lengths)}')
        return scores, lengths, exec_time, discounted_scores

    def rollout(self, env, policy_fn, max_episode_len, n_episodes=None, n_steps=None, name='eval', stats_fn=None):
        logger = self.logger
        timestep = 0
        tau_total = 0
        reset = True
        assert (n_steps is None) != (n_episodes is None)
        episodes = []
        terminate = False
        n_steps = n_steps if n_steps else -1
        n_episodes = n_episodes if n_episodes else -1
        
        assert n_steps > 0 or n_episodes > 0

        while not terminate:
            if reset:
                obs = env.reset()
                done = False
                test_r = 0
                discounted_test_r = 0
                episode_len = 0
                info = {}
                tau_total = 0
                ep = []
            
            initset = getattr(env, 'last_initset', None)
            a = policy_fn(obs, initset)
            next_obs, r, done, info = env.step(a)
            
            ep.append([obs, a, r, next_obs, done, info])
            obs = next_obs
            
            episode_len += 1
            timestep += 1
            reset = done or episode_len == max_episode_len or info.get("needs_reset", False)
            if 'tau' not in info:
                info['tau'] = 1 

            self.agent.observe(obs, r, done, reset, info)
            test_r += r
            discounted_test_r += (self.agent.gamma ** tau_total) * r
            tau_total += info['tau']
            if reset:
                logger.info(
                    f"{name} episode {len(episodes)} length:{episode_len} R:{test_r}"
                )
                episodes.append(ep)
            terminate = len(episodes) == n_episodes or timestep == n_steps
                
        # If all steps were used for a single unfinished episode
        if len(episodes) == 0:
            episodes.append(ep) # non terminated episode
        return None if not stats_fn else stats_fn(episodes)
 
    def train(self, world_model, steps_budget):
        self.agent.batch_last_episode = None
        stats = self.rollout(
                            world_model, 
                            self.act_in_imagination, 
                            self.max_episode_len, 
                            n_steps=steps_budget, 
                            stats_fn=self.compute_metrics
                        )
        self.imagination_steps += steps_budget
        self.n_episodes += stats['n_episodes']
        del stats['n_episodes']
        return stats
    
    def agent_train(self, eps):
        pass
        # rollout
        # compute returns & 2hot encode
        # compute & update statistics batch statistics
        # critic train
        # actor train
        # update target 

    @contextmanager
    def eval_mode(self):
        orig_mode = self.training
        try:
            self.training = False
            yield
        finally:
            self.training = orig_mode

############ TESTING ############

def test_twohot():
    nbins = 10
    brange = [-20, 20]

    # generate testcase
    bins = torch.randint(0, nbins, (3,3)).float()
    
    x = (bins + torch.rand_like(bins)) / (nbins-1)
    x = x * (brange[1]-brange[0]) + brange[0]
    # transform 
    _x = twohot(x.unsqueeze(-1), nbins=nbins, bmin=brange[0], bmax=brange[1])
    bins_recons = (_x > 0).float().argmax(-1)
    assert torch.all(bins_recons == bins), f'{bins} != {bins_recons}'

    print('TwoHot Test passed!')

def test_lambda_return():
    horizon = 5
    dones = torch.nn.functional.one_hot(torch.randint(0, horizon, (2, )), horizon)
    rs = torch.randn(2, horizon)
    values = torch.rand(2, horizon+1)
    lambd = 0.97
    gamma = 0.99
    G_lambda = compute_lambda_return(rs, dones, values, gamma, lambd)

    cont = torch.cumprod(1-dones, dim=-1)
    G = []


    for b in range(2):
        returns = []
        for t in reversed(range(horizon)):
            g = 0
            for i in reversed(range(t, horizon)):
                g = rs[b, i] + cont[b, i] * gamma * ((1-lambd) * values[b, i+1] + lambd * g)
            returns.append(g)
        G.append(list(reversed(returns)))

    G = torch.Tensor(G)
    assert torch.allclose(G, G_lambda)
    print('Lambda return test passed!')

def test_critic():
    from omegaconf import OmegaConf as oc

    config = {
        'nbins': 255,
        'bmin': -20,
        'bmax': 20,
        'reg_const': 1.0,
        'gamma': 0.99,
        'lambd': 0.95,
        'lr': 1e-4,
        'slow_ema_decay': 0.98,
        'critic':{
            'type': 'mlp',
            'output_dim': '${nbins}',
            'input_dim': 5,
            'hidden_dims': [128, 128],
            'normalize': True,
            'activation': 'mish'
        }   
    }
    cfg = oc.create(config)
    H = 5
    critic = DistributionalCritic(cfg)
    states = torch.randn((2, H+1, 5))
    dones = torch.nn.functional.one_hot(torch.randint(0, H, (2,)), H)
    rewards = torch.randint(0, 5, (2, H)).float()
    logprob = critic.logprobs(states)
    values = critic(states)
    loss, returns, values = critic.loss(states, rewards, dones)
    print('Critic Forward pass!')

def test_actor():
    pass

if __name__ == '__main__':
    test_critic()
    test_lambda_return()
    test_twohot()
