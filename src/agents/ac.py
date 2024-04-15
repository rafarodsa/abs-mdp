
import torch
from torch.nn import functional as F
import numpy as np


from src.models.factories import build_model
from src.models.factories import ModuleFactory
from jax import tree_map

from contextlib import contextmanager
from src.utils.symlog import symlog, symexp

import statistics
from copy import deepcopy
import time, logging


from src.agents.evaluator import record_stats, record_tb_stats, create_tb_writer, write_header
from src.utils.printarr import printarr


from collections import deque

def mean(lst):
    if len(lst) == 0:
        return 0.
    else:
        return statistics.mean(lst)
    
def ortho_init(layer, gain=1.):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.zeros_(layer.bias)
    return layer
def zeros_init(layer):
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    return layer

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
    distances = torch.abs(normalized_value - bins) 
    twohot = (1-distances / delta) * (distances  < delta) 
    twohot = twohot * (value < bmax) * (value > bmin) + ((value > bmax) + (value < bmin)) * (twohot > 0) 
    assert torch.allclose(twohot.sum(-1, keepdims=True), torch.ones_like(value), rtol=1e-4), twohot.sum(-1)
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

        
        self.slow_critic = ModuleFactory.build(cfg.critic, out_init_fn=zeros_init)
        self.fast_critic = ModuleFactory.build(cfg.critic, out_init_fn=zeros_init)
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

    def loss(self, states, rewards, dones, masks):
        '''
            H is the padded temporal dimension.
            masks.sum(-1) is the last valid transition. 
            masks.sum(-1)+1 is the final state of the ep

            states: torch.Tensor (B, H, D)
            rewards: torch.Tensor (B, H, 1)
            dones: torch.Tensor(B, H)
            masks: torch.Tensor(B, H)
            return: (loss, return, value)

        '''        
        # compute values v(s)
        values = self.compute_values(states, self.fast_critic).detach()

        # compute target returns G^lambda & 2hot encode y_t
        masked_dones = ((dones + (1-masks)) > 0).float()
        returns = compute_lambda_return(rewards, masked_dones, symexp(values), self.gamma, self.lambd)
        y_t = self.twohot(symlog(returns.unsqueeze(-1))).detach()

        # compute logprobs of bins ln P(b|s_t)
        logprobs = self.logprobs(states)
        # regularizer
        # regularizer: (1) cross-entropy: slow_y_t^T ln P(b|s_t) or (2) -log P(slow_v(s_t)|s_t)
        # choosing option (2)
        slow_values = self.twohot(self.compute_values(states, self.slow_critic).unsqueeze(-1)).detach()
        reg = -(slow_values * logprobs).sum(-1)
        # loss = -y_t^T ln P(b_i|s_t) 

        loss = (-y_t * logprobs).sum(-1) + (self.reg_const * reg) ##
        loss = (loss * masks).sum(-1)
        return loss.mean(), (returns, symexp(values))
    
    def update_critic(self):
        with torch.no_grad():
            self.param_updater(deepcopy(self.fast_critic.state_dict()))
            self.slow_critic.load_state_dict(self.param_updater.get_value())
    
    def get_params(self):
        return self.fast_critic.parameters()

class Actor:
    def __init__(self, cfg):
        self.policy = build_model(cfg.actor)
        self.entropy_const = cfg.entropy_const
        # batch statistics
        self.perclow = EMA(cfg.normalizer_ema_decay, 0.)
        self.perchigh = EMA(cfg.normalizer_ema_decay, 0.)

    def logpi(self, obs):
        return torch.nn.functional.log_softmax(self.policy(obs), dim=-1)

    def loss(self, states, actions, returns, values, masks):
        '''
            states: torch.Tensor (B, H, D)
            actions: torch.Tensor (B, H, N_ACTIONS)
            returns: torch.Tensor (B, H)
            values: torch.Tensor (B, H)
            return:
                actor_loss: \E ln\pi(a_t|s_t)A(s_t, a_t) + entropy_const * H[\pi(\dot|s_t)]
        '''
        # compute stats 
        returns = returns.detach().float()


        quantiles = torch.quantile(returns[masks>0], torch.Tensor([0.05, 0.95]))
        
        self.perclow(quantiles[0])
        self.perchigh(quantiles[1])
        
        # normalize
        scale = torch.maximum(torch.tensor(1.0), self.perchigh.get_value() - self.perclow.get_value())
        normalized_returns = returns if scale == 1. else (returns-self.perclow.get_value()) / scale
        normalized_values = values if scale == 1. else (values - self.perclow.get_value()) / scale
        # Adv_t = G^\lambda_t - V_t
        adv = normalized_returns - normalized_values
        # compute ln\pi(a_t|s_t) * adv_t
        logpis = self.logpi(states) # ln\pi(\dot|s_t)

        logpis_a = (logpis * actions).sum(-1) # ln\pi(a_t|s_t)
        reinforce =  logpis_a * adv * masks
        # compute entropy 
        entropy = -(logpis.exp() * logpis).sum(-1) * masks

        loss = -reinforce - self.entropy_const * entropy
        return loss.sum(-1).mean()

    def get_params(self):
        return self.policy.parameters()

class DistributionalActorCritic:
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self.training = True
        # build model
        # initialize critic to predict zeros
        self.critic = DistributionalCritic(model_cfg.planner.dist_ac)
        self.actor = Actor(model_cfg.planner.dist_ac)
        self.actor_lr = model_cfg.planner.dist_ac.actor_lr
        self.critic_lr = model_cfg.planner.dist_ac.critic_lr
        self.n_actions = model_cfg.planner.dist_ac.n_actions
        self.optimizers = []

        # trainer params
        self.max_episode_len = self.cfg.experiment.max_episode_len
        self.eval_max_episode_len = self.cfg.experiment.max_episode_len*2
        self.ground_step = 0
        self.imagination_steps = 0 
        self.n_episodes = 0
        self.n_updates = 0
        self.start_time = time.time()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel('INFO')

        self.value_record = EMA(alpha=0.99, initialization=0.)
        self.entropy_record = EMA(alpha=0.99, initialization=0.)
        self.value_loss_record = EMA(alpha=0.99, initialization=0.)
        self.policy_loss_record = EMA(alpha=0.99, initialization=0.)


    def setup(self, outdir):
        # output logger
        self.outdir = outdir
        self.tb_writer = None
        if self.cfg.experiment.log_tensorboard:
            self.tb_writer = create_tb_writer(self.outdir)
        write_header(self.outdir, self, None)
        # optimizers
        # optimizer_actor = torch.optim.Adam(self.actor.get_params(), lr=self.actor_lr, eps=1e-5)
        # optimizer_critic = torch.optim.Adam(self.critic.get_params(), lr=self.critic_lr, eps=1e-5)
        # self.optimizers = [optimizer_actor, optimizer_critic]
        optimizer = torch.optim.Adam(list(self.critic.get_params()) + list(self.actor.get_params()), lr=self.critic_lr, eps=1e-5)
        self.optimizers = [optimizer]

    def get_statistics(self):
        stats = (
            ('value', float(self.value_record.get_value())),
            ('entropy', float(self.entropy_record.get_value())),
            ('value_loss', float(self.value_loss_record.get_value())),
            ('policy_loss', float(self.policy_loss_record.get_value())),
            ('return_perc_low', float(self.actor.perclow.get_value())),
            ('return_perc_high', float(self.actor.perchigh.get_value())),
            ('n_updates', self.n_updates)
        )
        return stats

    def act(self, obs, encoder):
        if isinstance(obs, list):
            z = torch.stack([encoder(o) for o in obs])
        else:
            z = encoder(obs)
        action_probs = self.actor.logpi(z).exp()
        if self.training:
            # sample
            actions = torch.distributions.Categorical(probs=action_probs).sample((1,))[0]
        else:
            # Greedy
            actions = action_probs.argmax(-1)

        return actions.tolist() if isinstance(obs, list) else actions.item()

    def act_in_imagination(self, obs, initset=None):
        action_probs = self.actor.logpi(torch.from_numpy(obs)).exp()
        if self.training:
            # sample
            self.entropy_record(-(action_probs * torch.log(action_probs)).sum(-1).mean().item())
            actions = torch.distributions.Categorical(probs=action_probs).sample((1,))[0]
        else:
            # Greedy
            actions = actions.argmax(-1)
        return actions.item()

    def compute_metrics(self, episodes, world_model):
        logs = []
        successful_eps_len = []
        for ep in episodes:
            obss, actions, rewards, next_obss, dones, infos = zip(*ep)
            ep = np.array(obss + next_obss[-1:])
            norm2 = (ep ** 2).sum(-1, keepdims=True)
            residual = ((ep[1:]-ep[:-1])**2).sum(-1, keepdims=True)
            if sum(rewards) != 0.:
                successful_eps_len.append(len(rewards))

            acts = torch.nn.functional.one_hot(torch.from_numpy(np.array(actions)).long(), self.n_actions)
            ss = torch.from_numpy(np.array(obss))
            next_ss = torch.from_numpy(np.array(next_obss))
            with torch.no_grad():
                likelihood = world_model.transition.distribution(torch.cat([ss, acts], dim=-1)).log_prob(next_ss-ss).sum(-1) / len(rewards)

            logs.append({
                'sim_rollout/norm2_mean': norm2.mean(),
                'sim_rollout/norm2_std': norm2.std(),
                'sim_rollout/norm2_max': norm2.max(),
                'sim_rollout/norm2_min': norm2.min(),
                'sim_rollout/residual_norm2_mean': residual.mean(),
                'sim_rollout/residual_norm2_std': residual.std(),
                'sim_rollout/residual_norm2_max': residual.max(),
                'sim_rollout/residual_norm2_min': residual.min(),
                'sim_rollout/return': sum(rewards),
                'sim_rollout/lengths': len(rewards),
                'sim_rollout/ep_likelihood': likelihood.cpu().item()
            })
        stats = tree_map(lambda *ep_stats: np.mean(np.array(ep_stats)), *logs)
        stats['n_episodes'] = len(episodes)
        stats['sim_rollout/steps_to_success'] = mean(successful_eps_len)
        return stats
    
    def compute_eval_stats(self, episodes):
        scores, lengths, exec_time, discounted_scores = [], [], [], []
        for ep in episodes:
            obss, actions, rewards, next_obss, dones, infos = zip(*ep)
            scores.append(sum(rewards))
            lengths.append(len(actions))
            taus = [info['tau'] for info in infos]
            discounts =  self.critic.gamma ** np.cumsum([0] + taus)
            discounted_rewards = np.array(rewards) * discounts[:-1]
            exec_time.append(sum(taus))
            discounted_scores.append(float(discounted_rewards.sum()))
        
        print(f'Evaluation: mean return {sum(scores) / len(scores)}, mean length {sum(lengths) / len(lengths)}')
        return scores, lengths, exec_time, discounted_scores

    def rollout(self, env, policy_fn, max_episode_len, n_episodes=None, n_steps=None, name='eval'):
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

            test_r += r
            discounted_test_r += (self.critic.gamma ** tau_total) * r
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
        return episodes
    
    def preprocess(self, episodes, maxlen=64):
        def pad(tensor, padding_len):
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(-1)
            padded = F.pad(tensor, (0,0,0, padding_len))
            return padded
        def totensor(t):
            if isinstance(t, (float, int)):
                return torch.tensor(t)
            if isinstance(t, (np.ndarray, list, tuple)):
                return torch.from_numpy(np.array(t))
            if isinstance(t, torch.Tensor):
                return t
            raise ValueError(f'Unknown type {type(t)}')

        masks = torch.stack(list(map(lambda ep: torch.Tensor([1] * len(ep) + [0] * (maxlen-len(ep))), episodes)))
        episodes = [tree_map(totensor, list(zip(*ep))[:-1]) for ep in episodes]
        episodes = [list(map(torch.stack, ep)) for ep in episodes]
        obss, acts, rews, next_obss, dones = list(zip(*episodes))
        ss = tree_map(lambda ss, next_ss: torch.cat([ss, next_ss[-1:]], dim=0), obss, next_obss) # (maxlen + 1, D)
        acts = tree_map(lambda a: F.one_hot(a, self.n_actions), acts)
        
        ss, acts, rews, dones = tree_map(lambda t: pad(t, maxlen-t.shape[0]), [ss, acts, rews, dones])

        
        return torch.stack(ss), torch.stack(acts), torch.stack(rews).squeeze(-1), torch.stack(dones).float().squeeze(-1), masks

 
    def train(self, world_model, steps_budget, epochs=5, max_ep_len=64):
        # rollout
        _stats = []
        for _ in range(epochs):
            episodes = self.rollout(
                                world_model, 
                                self.act_in_imagination, 
                                self.max_episode_len, 
                                # n_steps=steps_budget // epochs,
                                n_episodes=3
                            )
            self.imagination_steps += steps_budget
            self.n_episodes += len(episodes)
            states, actions, rewards, dones, masks = self.preprocess(episodes, max_ep_len)
            # compute losses
            critic_loss, (returns, values) = self.critic.loss(states, rewards, dones, masks)
            actor_loss = self.actor.loss(states, actions, returns, values, masks)
            # take step
            loss = actor_loss + critic_loss
            [opt.zero_grad() for opt in self.optimizers]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor.get_params(), 100)
            torch.nn.utils.clip_grad_norm_(self.critic.get_params(), 100)

            [opt.step() for opt in self.optimizers]

            # update target
            self.critic.update_critic()

            # agent stats
            self.policy_loss_record(actor_loss.item())
            self.value_loss_record(critic_loss.item())
            self.value_record(returns[masks==1].mean().item())
            stats = self.compute_metrics(episodes, world_model)
            print(f'[agent training] critic_loss: {critic_loss}, actor_loss: {actor_loss}, return {returns[masks==1].mean()}, value_estimates {values[masks==1].mean()}, success rate {rewards.sum(-1).mean().item()}, loglikelihood {stats["sim_rollout/ep_likelihood"]}')
            self.n_updates += 1
            _stats.append({**stats, **{f"agent/{k}":v for k,v in self.get_statistics()}})

            

        return tree_map(lambda *s: mean(s), *_stats)


    def evaluate(self, env, encoder, n_episodes):
        with self.eval_mode():
            episodes = self.rollout(
                                        env,
                                        lambda obs, initset: self.act(obs, encoder),
                                        self.eval_max_episode_len,
                                        n_episodes=n_episodes
                                    )
            
        eval_stats = self.compute_eval_stats(episodes)
        scores, lengths, exec_times, discounted_scores = eval_stats
        option_duration_mean = np.array(exec_times)/np.array(lengths)

        stats = dict(
            episodes=len(scores),
            mean=statistics.mean(scores),
            median=statistics.median(scores),
            stdev=statistics.stdev(scores) if len(scores) >= 2 else 0.0,
            max=np.max(scores),
            min=np.min(scores),
            length_mean=statistics.mean(lengths),
            length_median=statistics.median(lengths),
            length_stdev=statistics.stdev(lengths) if len(lengths) >= 2 else 0,
            length_max=np.max(lengths),
            length_min=np.min(lengths),
            option_len_mean = statistics.mean(option_duration_mean),
            option_len_median = statistics.median(option_duration_mean),
            option_len_std = statistics.stdev(option_duration_mean) if len(option_duration_mean) >= 2 else 0.0,
            option_len_min = option_duration_mean.min(),
            option_len_max = option_duration_mean.max(),
            discounted_mean = statistics.mean(discounted_scores),
            discounted_median = statistics.median(discounted_scores),
            discounted_std = statistics.stdev(discounted_scores) if len(discounted_scores) >= 2 else 0.0,
            discounted_min = np.min(discounted_scores),
            discounted_max = np.max(discounted_scores),
        )

        # agent_stats = [] #self.agent.get_statistics()
        agent_stats = self.get_statistics()
        custom_values = tuple(tup[1] for tup in agent_stats)

        mean = stats["mean"]
        elapsed = time.time() - self.start_time
        values = (
            (
                self.ground_step,
                self.n_episodes,
                elapsed,
                mean,
                stats["median"],
                stats["stdev"],
                stats["max"],
                stats["min"],
            )
            + custom_values
        )

        record_stats(self.outdir, values)
        if self.tb_writer:
            record_tb_stats(self.tb_writer, agent_stats, stats, [], self.ground_step)

        return stats

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
        'normalizer_ema_decay': 0.98,
        'entropy_const': 3e-4,
        'n_actions': 5,
        'actor':{
            'type': 'mlp',
            'output_dim': '${n_actions}',
            'input_dim': 5,
            'hidden_dims': [128, 128],
            'normalize': True,
            'activation': 'mish'
        }   
    }
    cfg = oc.create(config)
    H = 5
    states = torch.randn((2, H+1, 5))
    dones = torch.nn.functional.one_hot(torch.randint(0, H, (2,)), H)
    rewards = torch.randint(0, 5, (2, H)).float()
    values = torch.randn((2, H+1))
    actions = torch.nn.functional.one_hot(torch.randint(0, cfg.n_actions, (2,H)), cfg.n_actions)
    returns = compute_lambda_return(rewards, dones, values, gamma=cfg.gamma, lambd=cfg.lambd)
    actor = Actor(cfg)
    actor.loss(states, actions, returns, values)

    print('Actor Forward pass!')

if __name__ == '__main__':
    test_actor()
    test_critic()
    test_lambda_return()
    test_twohot()
