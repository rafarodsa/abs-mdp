import torch
import jax
import numpy as np
from contextlib import contextmanager


class AbstractMPC:
    def __init__(self, world_model, action_mask=None, device='cpu', recurrent=False):
        self.world_model = world_model
        self.action_mask = action_mask
        self.device = device
        self.recurrent = recurrent
        self.n_actions = world_model.n_options
        self.gamma = world_model.gamma

    
    def mpc(self, obs, N=1000, K=100, horizon=150, device='cpu'):
        
        n_obs = len(obs)
        obs = jax.tree_map(lambda *s: torch.stack(s).repeat_interleave(N, dim=0), *obs)
        # rollout for N trajs
        z = self.world_model.reset(obs) # (NxN_OBS)
        trajs = []
        actions = []
        for t in range(horizon):
            # random action selection
            _action = torch.randint(0, self.n_actions,(N*n_obs,)).long().to(device)
            ret = self.world_model.step(_action)
            trajs.append(ret)
            actions.append(_action)
        

        
        # compute returns
        rewards = jax.tree_map(lambda transition: transition[1], trajs, is_leaf=lambda s: isinstance(s, (tuple,)))
        rewards = torch.stack(rewards).squeeze() # H x N
        dones = jax.tree_map(lambda transition: transition[2].float(), trajs, is_leaf=lambda s: isinstance(s, (tuple,))) 
        dones = torch.stack(dones).squeeze() # H x N
        dones = torch.cat([dones, torch.zeros(1, N*n_obs).to(self.device)], dim=0)
        dones = torch.cumprod(1.-dones, dim=0) # H x N
        discounts = torch.Tensor([self.gamma ** i for i in range(horizon+1)]).unsqueeze(1).to(self.device) * dones # T x N 
        
        # final_value = self.agent.agent.model(torch.Tensor(trajs[-1][0]).to(self.device)).q_values.max(-1)[0] * discounts[-1] # N 
        final_value = 0
        returns = (rewards * discounts[:-1]).sum(0) + final_value # (Nxn_obs)
        

        returns = returns.reshape(n_obs, N) # n_obs x N
        actions = torch.stack(actions).reshape(-1, n_obs, N) # T x n_obs x N
        # pick top K
        top_k, top_k_idx = torch.topk(returns, K)
        top_k_idx = top_k_idx.repeat(horizon, 1, 1)
        actions = torch.gather(actions, dim=-1, index=top_k_idx)
        actions_prob = 1/K * torch.nn.functional.one_hot(actions, self.n_actions).sum(-2) # T x n_obs x N_ACTIONS

        return actions_prob.argmax(-1).cpu()[0]

    def act(self, obs, initset=None):
        # obs = jax.tree_map(self.preprocess, obs)
        obs = self.preprocess(obs)
        if not isinstance(obs, list):
            obs = [obs]
        action = self.mpc(obs, device=self.world_model.device)
        return action if len(obs) > 1 else action[0]
    
    def _preprocess(self, obs):
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs.copy()).to(self.device)
        elif isinstance(obs, torch.Tensor):
            return obs.to(self.device)
        elif isinstance(obs, (np.bool_, bool, np.intc, np.float_, float, int)):
            return torch.tensor(obs)
        return obs

    def preprocess(self, obs):
        return jax.tree_map(self._preprocess, obs)
    
    @contextmanager
    def eval_mode(self):
        # orig_mode = self.agent.agent.training
        try:
            current_z = self.world_model.current_z
            # self.agent.agent.training = False
            yield
        finally:
            # self.agent.agent.training = orig_mode
            self.world_model.current_z = current_z
            