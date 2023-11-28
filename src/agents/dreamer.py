import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
import numpy as np

from dreamerv2 import common
from dreamerv2 import expl
import gym
from src.agents.nets import ResidualEnsembleRSSM as EnsembleRSSM

class GymWrapperOptions(common.envs.GymWrapper):
  @property
  def obs_space(self):
    if self._obs_is_dict:
      spaces = self._env.observation_space.spaces.copy()
    else:
      spaces = {self._obs_key: self._env.observation_space}
    return {
        **spaces,
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        'duration': gym.spaces.Box(0, np.inf, (), dtype=np.float32)
    }
  
  def step(self, action):
    if not self._act_is_dict:
      action = action[self._act_key]
    obs, reward, done, info = self._env.step(action)
    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    obs['reward'] = float(reward)
    obs['is_first'] = False
    obs['is_last'] = done
    obs['is_terminal'] = info.get('is_terminal', done)
    obs['duration'] = float(info['tau'])
    obs['success'] = float(info['success'])
    return obs

  def reset(self):
    obs = self._env.reset()
    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    obs['reward'] = 0.0
    obs['is_first'] = True
    obs['is_last'] = False
    obs['is_terminal'] = False
    obs['duration'] = 0.
    obs['success'] = 1.
    return obs

class Agent(common.Module):

  def __init__(self, config, obs_space, act_space, step):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.tfstep = tf.Variable(int(self.step), tf.int64)
    # self.wm = WorldModel(config, obs_space, self.tfstep)
    self.wm = MarkovianWorldModel(config, obs_space, self.tfstep)
    self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)
    if config.expl_behavior == 'greedy':
      self._expl_behavior = self._task_behavior
    else:
      self._expl_behavior = getattr(expl, config.expl_behavior)(
          self.config, self.act_space, self.wm, self.tfstep,
          lambda seq: self.wm.heads['reward'](seq['feat']).mode())

  @tf.function
  def policy(self, obs, state=None, mode='train'):
    obs = tf.nest.map_structure(tf.tensor, obs)
    tf.py_function(lambda: self.tfstep.assign(
        int(self.step), read_value=False), [], [])
    # if state is None:
    #   # latent = self.wm.rssm.initial(len(obs['reward']))
    #   latent = self.
    #   action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
    #   state = latent, action
    # latent, action = state
    embed = self.wm.encoder(self.wm.preprocess(obs))
    sample = (mode == 'train') or not self.config.eval_state_mean
    # latent, _ = self.wm.rssm.obs_step(
    #     latent, action, embed, obs['is_first'], sample)
    # feat = self.wm.rssm.get_feat(latent)
    feat = embed
    if mode == 'eval':
      actor = self._task_behavior.actor(feat)
      action = actor.mode()
      noise = self.config.eval_noise
    elif mode == 'explore':
      actor = self._expl_behavior.actor(feat)
      action = actor.sample()
      noise = self.config.expl_noise
    elif mode == 'train':
      actor = self._task_behavior.actor(feat)
      action = actor.sample()
      noise = self.config.expl_noise
    action = common.action_noise(action, noise, self.act_space)
    outputs = {'action': action}
    state = (embed, action)
    return outputs, state

  @tf.function
  def train(self, data, state=None):
    metrics = {}
    state, outputs, mets = self.wm.train(data, state)
    metrics.update(mets)
    start = {'feat': outputs['feat'], 'z_a': outputs['z_a']}
    reward = lambda seq: self.wm.heads['reward'](seq['z_a']).mode()
    metrics.update(self._task_behavior.train(
        self.wm, start, data['is_terminal'][:,1:], reward))
    if self.config.expl_behavior != 'greedy':
      mets = self._expl_behavior.train(start, outputs, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    return state, metrics

  @tf.function
  def report(self, data):
    report = {}
    # data = self.wm.preprocess(data)
    # for key in self.wm.heads['decoder'].cnn_keys:
    #   name = key.replace('/', '_')
    #   report[f'openl_{name}'] = self.wm.video_pred(data, key)
    return report


class WorldModel(common.Module):
  SMOOTHING_NOISE_STD = 0.2
  
  def __init__(self, config, obs_space, tfstep):
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    self.config = config
    self.tfstep = tfstep
    self.rssm = common.EnsembleRSSM(**config.rssm)
    self.encoder = common.Encoder(shapes, **config.encoder)
    self.heads = {}
    self.heads['duration'] = common.MLP([], **config.duration_head)
    self.heads['reward'] = common.MLP([], **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    if config.pred_initset:
      self.heads['success'] = common.MLP([], **config.initset_head)
    # for name in config.grad_heads:
    #   assert name in self.heads, name
    self.model_opt = common.Optimizer('model', **config.model_opt)

  def train(self, data, state=None):
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    modules = [self.encoder, self.rssm, *self.heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))
    return state, outputs, metrics
  

  def grounding_loss(self, post, std=0.1):
    # feats = self.rssm.get_feat(post)
    # feats = post['stoch']
    feats = post['stoch']
    feats = feats + tf.random.normal(feats.shape, stddev=self.SMOOTHING_NOISE_STD, dtype=feats.dtype) # batch x len x dim
    # b_size = tf.reduce_prod(feats.shape[:-1])
    # feats = feats.reshape((b_size, -1))
    norm = ((feats[:, tf.newaxis] - post['stoch'][tf.newaxis]) / self.SMOOTHING_NOISE_STD ) ** 2 # batch x batch x len x dim
    norm = -norm.sum(-1) # 

    b_size = tf.cast(feats.shape[0], norm.dtype)
    diag  = tf.transpose(tf.linalg.diag_part(tf.transpose(norm, (2,0,1)))) # batch_size x len
    loss = diag - (tf.math.reduce_logsumexp(norm, axis=1) - tf.math.log(b_size))
    loss = tf.reduce_mean(tf.reduce_mean(tf.cast(loss, tf.float32), axis=-1))
    return -loss
  
  def tpc_loss(self, prior, post):

    transition_dist = self.rssm.get_dist(prior) # batch x len x k
    next_z = post['stoch'] # batch x len x k
    next_z = next_z + tf.random.normal(next_z.shape, stddev=self.SMOOTHING_NOISE_STD, dtype=next_z.dtype)
    delta_z = next_z[:, tf.newaxis] - prior['stoch'][tf.newaxis] # batch x batch x len x k
    
    logprob = transition_dist.log_prob(delta_z) # batch_size x distr_batch_size x len 
    diag  = tf.transpose(tf.linalg.diag_part(tf.transpose(logprob, (2,0,1)))) # batch_size x len
    batch_size = tf.cast(next_z.shape[0], logprob.dtype) 
    loss = diag - (tf.math.reduce_logsumexp(logprob, axis=1) - tf.math.log(batch_size)) # batch_size x len
    return -tf.reduce_mean(tf.reduce_mean(loss, axis=-1))

  def transition_loss(self, prior, post):
    transition_dist = self.rssm.get_dist(prior)
    feats = post['stoch'] - prior['stoch']
    feats = feats + tf.random.normal(feats.shape, stddev=self.SMOOTHING_NOISE_STD, dtype=feats.dtype)
    return -tf.reduce_mean(tf.reduce_mean(transition_dist.log_prob(feats), axis=-1))
  
  def loss(self, data, state=None):
    data = self.preprocess(data)
    embed = self.encoder(data)
    # embed = embed + tf.random.normal(embed.shape, stddev=self.SMOOTHING_NOISE_STD, dtype=embed.dtype)
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
  
    # prior is the T(z'| h, a) and post is q(z'| x', h)
    # deter is the recurrent state and stoch is the sample z state.
    transition_loss = self.transition_loss(prior, post)
    tpc_loss = self.tpc_loss(prior, post)
    grounding_loss = self.grounding_loss(post)

    assert len(transition_loss.shape) == 0 and  len(grounding_loss.shape) == 0 and  len(tpc_loss.shape) == 0
    likes = {}
    losses = {'grounding': grounding_loss, 'tpc': tpc_loss, 'transition': transition_loss}
    # feat = self.rssm.get_feat(prior)
    feat = self.rssm.get_feat(post)

    for name, head in self.heads.items():
      grad_head = (name in self.config.grad_heads)
      inp = feat if grad_head else tf.stop_gradient(feat)
      out = head(inp)
      dists = out if isinstance(out, dict) else {name: out}
      for key, dist in dists.items():
        like = tf.cast(dist.log_prob(data[key]), tf.float32)
        likes[key] = like
        losses[key] = -like.mean()

    model_loss = sum(
        self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, grounding_loss=grounding_loss, tpc_loss=tpc_loss, transition_loss=transition_loss)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    metrics['reward_ent'] = self.heads['reward'](feat).entropy().mean()
    metrics['reward_pos'] = data['reward'].sum()
    metrics['reward_neg'] = (1-data['reward']).sum()
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss, last_state, outs, metrics

  def imagine(self, policy, start, is_terminal, horizon):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = self.rssm.get_feat(start)
    start['action'] = tf.zeros_like(policy(start['feat']).mode())
    seq = {k: [v] for k, v in start.items()}
    for _ in range(horizon):
      action = policy(tf.stop_gradient(seq['feat'][-1])).sample()
      state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
      feat = self.rssm.get_feat(state)
      for key, value in {**state, 'action': action, 'feat': feat}.items():
        seq[key].append(value)
    seq = {k: tf.stack(v, 0) for k, v in seq.items()}
    if 'duration' in self.heads:
      # seq['duration_pred'] = tf.math.exp(self.heads['duration'](seq['feat']).mean())
      seq['duration_pred'] = self.heads['duration'](seq['feat']).mean()
      disc = tf.pow(self.config.discount, (seq['duration_pred']))
      if 'discount' in self.heads:
        disc = disc * tf.cast(self.heads['discount'](seq['feat']).mode(), disc.dtype)
      if is_terminal is not None:
        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
        true_first *= 0.99
        disc = tf.concat([true_first[None], disc[1:]], 0)
    else:
      disc = self.config.discount * tf.ones(seq['feat'].shape[:-1])
    seq['discount'] = disc
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    seq['weight'] = tf.math.cumprod(
        tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0)
    return seq

  @tf.function
  def preprocess(self, obs):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_'):
        continue
      if value.dtype == tf.int32:
        value = value.astype(dtype)
      if value.dtype == tf.uint8:
        value = value.astype(dtype) / 255.0 - 0.5
      obs[key] = value
    obs['reward'] = {
        'identity': tf.identity,
        'sign': tf.sign,
        'tanh': tf.tanh,
    }[self.config.clip_rewards](obs['reward'])
    obs['discount'] = 1.0 - obs['is_terminal'].astype(dtype)
    # obs['discount'] *= self.config.discount
    obs['duration'] = tf.math.log(obs['duration'])
    return obs

  @tf.function
  def video_pred(self, data, key):
    decoder = self.heads['decoder']
    truth = data[key][:6] + 0.5
    embed = self.encoder(data)
    states, _ = self.rssm.observe(
        embed[:6, :5], data['action'][:6, :5], data['is_first'][:6, :5])
    recon = decoder(self.rssm.get_feat(states))[key].mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:6, 5:], init)
    openl = decoder(self.rssm.get_feat(prior))[key].mode()
    model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    video = tf.concat([truth, model, error], 2)
    B, T, H, W, C = video.shape
    return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))



class MarkovianWorldModel(WorldModel):
  SMOOTHING_NOISE_STD = 0.2
  
  def __init__(self, config, obs_space, tfstep):
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    self.config = config
    self.tfstep = tfstep
    
    self.encoder = common.Encoder(shapes, **config.encoder)
    self.transition = common.MLP(config.encoder.mlp_layers[-1:], **config.transition) # TODO fix shape if CNN used
    self.heads = {}
    self.heads['duration'] = common.MLP([], **config.duration_head)
    self.heads['reward'] = common.MLP([], **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    if config.pred_initset:
      self.heads['success'] = common.MLP([], **config.success_head)
    # for name in config.grad_heads:
    #   assert name in self.heads, name
    self.model_opt = common.Optimizer('model', **config.model_opt)

  def train(self, data, state=None):
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    modules = [self.encoder, self.transition, *self.heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))
    return state, outputs, metrics

  def grounding_loss(self, next_z, std=0.1):
    feats = next_z
    feats = feats + tf.random.normal(feats.shape, stddev=self.SMOOTHING_NOISE_STD, dtype=feats.dtype) # batch x len x dim
    norm = ((feats[:, tf.newaxis] - next_z[tf.newaxis]) / self.SMOOTHING_NOISE_STD ) ** 2 # batch x batch x len x dim
    norm = -norm.sum(-1) # 

    b_size = tf.cast(feats.shape[0], norm.dtype)
    diag  = tf.transpose(tf.linalg.diag_part(tf.transpose(norm, (2,0,1)))) # batch_size x len
    loss = diag - (tf.math.reduce_logsumexp(norm, axis=1) - tf.math.log(b_size))
    loss = tf.reduce_mean(tf.reduce_mean(tf.cast(loss, tf.float32), axis=-1))
    return -loss
  
  def tpc_loss(self, z, next_z, transition_dist, is_first):
    next_z = next_z + tf.random.normal(next_z.shape, stddev=self.SMOOTHING_NOISE_STD, dtype=next_z.dtype)
    delta_z = next_z[:, tf.newaxis] - z[tf.newaxis] # batch x batch x len x k
    mask = 1.- tf.cast(is_first[:, 1:], tf.float32)
    delta_z = tf.cast(delta_z, tf.float32)
    logprob = transition_dist.log_prob(delta_z) # batch_size x distr_batch_size x len 
    diag  = tf.transpose(tf.linalg.diag_part(tf.transpose(logprob, (2,0,1)))) # batch_size x len
    batch_size = tf.cast(next_z.shape[0], logprob.dtype) 
    loss = diag - (tf.math.reduce_logsumexp(logprob, axis=1) - tf.math.log(batch_size)) # batch_size x len
    return -tf.reduce_mean((loss * mask).sum(-1) / mask.sum(-1))

  def transition_loss(self, z, next_z, transition_dist, is_first):
    next_z = next_z + tf.random.normal(next_z.shape, stddev=self.SMOOTHING_NOISE_STD, dtype=next_z.dtype)
    feats = tf.cast(next_z - z, tf.float32)
    mask = 1.- tf.cast(is_first[:, 1:], tf.float32)
    # feats = feats + tf.random.normal(feats.shape, stddev=self.SMOOTHING_NOISE_STD, dtype=feats.dtype)
    logprob = transition_dist.log_prob(feats)
    loss = ((logprob * mask).sum(-1) / mask.sum(-1)).mean()
    return -loss
  
  def loss(self, data, state=None):
    data = self.preprocess(data)
    embed = self.encoder(data)
    # obs, action, reward, is_terminal, is_first (batch x len)
    is_first = data['is_first'] # batch x len
    z = embed[:, :-1]
    action = tf.cast(data['action'][:, 1:], embed.dtype)
    z_a = tf.concat([embed[:, :-1], action], axis=-1)
    next_z = embed[:, 1:]
    transition_dist = self.transition(z_a)
    transition_loss = self.transition_loss(z, next_z, transition_dist, is_first)
    tpc_loss = self.tpc_loss(z, next_z, transition_dist, is_first)
    grounding_loss = self.grounding_loss(next_z)

    assert len(transition_loss.shape) == 0 and  len(grounding_loss.shape) == 0 and  len(tpc_loss.shape) == 0
    likes = {}
    losses = {'grounding': grounding_loss, 'tpc': tpc_loss, 'transition': transition_loss}

    feat = z_a
    mask = 1. - tf.cast(is_first, tf.float32)[:, 1:]
    for name, head in self.heads.items():
      grad_head = (name in self.config.grad_heads)
      inp = feat if grad_head else tf.stop_gradient(feat)
      if name not in ['discount']:
        out = head(inp)
      else:
        out = head(tf.concat([inp, next_z], axis=-1))
      dists = out if isinstance(out, dict) else {name: out}
      for key, dist in dists.items():
        _d = data[key][:, 1:]
        like = tf.cast(dist.log_prob(_d), tf.float32)
        likes[key] = like * mask
        losses[key] = -like.sum() / mask.sum()
    
    # import ipdb; ipdb.set_trace()

    model_loss = sum(
        self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    outs = dict(
        embed=embed, feat=next_z, z_a=z_a, likes=likes, grounding_loss=grounding_loss, tpc_loss=tpc_loss, transition_loss=transition_loss)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['post_ent'] = transition_dist.entropy().mean()
    # _dur = tf.math.exp(data['duration'])
    _dur = data['duration']
    metrics['batch_duration_mean'] =_dur.mean()
    metrics['batch_duration_min'] = _dur.min()
    metrics['batch_duration_max'] = _dur.max()
    # last_state = {k: v[:, -1] for k, v in post.items()}
    last_state = {'feat': next_z[:, -1], 'z_a': z_a[:, -1]}
    return model_loss, last_state, outs, metrics
  
  def _cast(self, x):
    return tf.cast(x, prec.global_policy().compute_dtype)
  
  def img_step(self, state, action):
    # import ipdb; ipdb.set_trace()
    action = tf.cast(action, prec.global_policy().compute_dtype)
    z_a = tf.concat([state['feat'], action], axis=-1)
    delta_z_dist = self.transition(z_a)
    delta_z_sample = tf.cast(delta_z_dist.sample(), prec.global_policy().compute_dtype)
    new_state = {'z_a': z_a, 'feat': state['feat'] + delta_z_sample}
    return new_state

  def imagine(self, policy, start, is_terminal, horizon):
    # start: dict (z_a, feat=next_z)

    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    start['action'] = tf.zeros_like(policy(start['feat']).mode())

    seq = {k: [v] for k, v in start.items()}
    for _ in range(horizon):
      action = policy(tf.stop_gradient(seq['feat'][-1])).sample()
      state = self.img_step({k: v[-1] for k, v in seq.items()}, action)
      for key, value in {**state, 'action': action}.items():
        seq[key].append(value)
    seq = {k: tf.stack(v, 0) for k, v in seq.items()}

    # z_a = tf.concat([seq['feat'][:-1], self._cast(seq['action'][1:])], axis=-1)
    z_a = seq['z_a']
    if 'duration' in self.heads:
      # max_duration = tf.math.log(200.)
      # seq['duration_pred'] = tf.math.exp(max_duration - tf.nn.softplus(max_duration - self.heads['duration'](z_a).mean()))
      # seq['duration_pred'] = tf.math.exp(self.heads['duration'](z_a).mean())-1
      seq['duration_pred'] = self.heads['duration'](z_a).mean()
      disc = tf.pow(self.config.discount, seq['duration_pred'])
      # disc = seq['duration_pred']
      # disc = self.config.discount * tf.ones(z_a.shape[:-1])
      # seq['duration_pred'] = disc
      if 'discount' in self.heads:
        cont_pred = tf.cast(self.heads['discount'](tf.concat([z_a, seq['feat']], axis=-1)).mode(), disc.dtype)
        seq['cont_pred'] = cont_pred
      if is_terminal is not None:
        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
        true_first *= disc[0]
        seq['cont_pred'] = tf.concat([true_first[tf.newaxis], cont_pred[1:]], 0)
        disc = tf.concat([true_first[tf.newaxis], disc[1:] * cont_pred[1:]], 0)
      else:
        disc = disc * cont_pred
    else:
      disc = self.config.discount * tf.ones(z_a.shape[:-1])
    seq['discount'] = disc
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.

    seq['weight'] = tf.math.cumprod(
        tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0)
    return seq

  def preprocess(self, obs):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_'):
        continue
      if value.dtype == tf.int32:
        value = value.astype(dtype)
      if value.dtype == tf.uint8:
        value = value.astype(dtype) / 255.0 - 0.5
      obs[key] = value
    obs['reward'] = {
        'identity': tf.identity,
        'sign': tf.sign,
        'tanh': tf.tanh,
    }[self.config.clip_rewards](obs['reward'])
    obs['discount'] = 1.0 - obs['is_terminal'].astype(dtype)
    # obs['discount'] *= self.config.discount
    # obs['duration'] = tf.math.log(tf.cast(obs['duration'], tf.float32))
    obs['duration'] = tf.cast(obs['duration'], tf.float32)
    obs['success'] = obs['success'].astype(dtype)
    return obs



class ActorCritic(common.Module):

  def __init__(self, config, act_space, tfstep):
    self.config = config
    self.act_space = act_space
    self.tfstep = tfstep
    discrete = hasattr(act_space, 'n')
    if self.config.actor.dist == 'auto':
      self.config = self.config.update({
          'actor.dist': 'onehot' if discrete else 'trunc_normal'})
    if self.config.actor_grad == 'auto':
      self.config = self.config.update({
          'actor_grad': 'reinforce' if discrete else 'dynamics'})
    self.actor = common.MLP(act_space.shape[0], **self.config.actor)
    self.critic = common.MLP([], **self.config.critic)
    if self.config.slow_target:
      self._target_critic = common.MLP([], **self.config.critic)
      self._updates = tf.Variable(0, tf.int64)
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('actor', **self.config.actor_opt)
    self.critic_opt = common.Optimizer('critic', **self.config.critic_opt)
    self.rewnorm = common.StreamNorm(**self.config.reward_norm)

  def train(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    hor = self.config.imag_horizon
    # The weights are is_terminal flags for the imagination start states.
    # Technically, they should multiply the losses from the second trajectory
    # step onwards, which is the first imagined step. However, we are not
    # training the action that led into the first step anyway, so we can use
    # them to scale the whole sequence.

    with tf.GradientTape() as actor_tape:
      seq = world_model.imagine(self.actor, start, is_terminal, hor)
      reward = reward_fn(seq)
      seq['reward'], mets1 = self.rewnorm(tf.cast(reward, tf.float32))
      mets1 = {f'reward_{k}': v for k, v in mets1.items()}
      mask = tf.math.cumprod(seq['cont_pred'], axis=0)
      masked_discount = seq['discount'] * mask
      masked_duration = seq['duration_pred'] * mask
      mets_img = {
                  'img_discount_mean': masked_discount.sum() / mask.sum(), 
                  'img_discount_max': masked_discount.max(), 
                  'img_discount_min': masked_discount.min(), 
                  'img_duration':  masked_duration.sum() / mask.sum(),
                  'img_duration_max': masked_duration.max(), 
                  'img_duration_min': masked_duration.min(),
                  }
      print(mets_img)
      target, mets2 = self.target(seq)
      actor_loss, mets3 = self.actor_loss(seq, target)
    with tf.GradientTape() as critic_tape:
      critic_loss, mets4 = self.critic_loss(seq, target)
    metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
    metrics.update(**mets1, **mets2, **mets3, **mets4, **mets_img)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target):
    # Actions:      0   [a1]  [a2]   a3
    #                  ^  |  ^  |  ^  |
    #                 /   v /   v /   v
    # States:     [z0]->[z1]-> z2 -> z3
    # Targets:     t0   [t1]  [t2]
    # Baselines:  [v0]  [v1]   v2    v3
    # Entropies:        [e1]  [e2]
    # Weights:    [ 1]  [w1]   w2    w3
    # Loss:              l1    l2
    metrics = {}
    # Two states are lost at the end of the trajectory, one for the boostrap
    # value prediction and one because the corresponding action does not lead
    # anywhere anymore. One target is lost at the start of the trajectory
    # because the initial state comes from the replay buffer.
    policy = self.actor(tf.stop_gradient(seq['feat'][:-2]))
    if self.config.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.config.actor_grad == 'reinforce':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      action = tf.stop_gradient(seq['action'][1:-1])
      objective = policy.log_prob(action) * advantage
    elif self.config.actor_grad == 'both':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(seq['action'][1:-1]) * advantage
      mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.config.actor_grad)
    ent = policy.entropy()
    ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = tf.stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    # States:     [z0]  [z1]  [z2]   z3
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]   v3
    # Weights:    [ 1]  [w1]  [w2]   w3
    # Targets:    [t0]  [t1]  [t2]
    # Loss:        l0    l1    l2
    dist = self.critic(seq['feat'][:-1])
    target = tf.stop_gradient(target)
    weight = tf.stop_gradient(seq['weight'])
    critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
    metrics = {'critic': dist.mode().mean()}
    return critic_loss, metrics

  def target(self, seq):
    # States:     [z0]  [z1]  [z2]  [z3]
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]  [v3]
    # Discount:   [d0]  [d1]  [d2]   d3
    # Targets:     t0    t1    t2
    reward = tf.cast(seq['reward'], tf.float32)
    disc = tf.cast(seq['discount'], tf.float32)
    value = self._target_critic(seq['feat']).mode()
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.config.discount_lambda,
        axis=0)
    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, metrics

  def update_slow_target(self):
    if self.config.slow_target:
      if self._updates % self.config.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.config.slow_target_fraction)
        for s, d in zip(self.critic.variables, self._target_critic.variables):
          d.assign(mix * s + (1 - mix) * d)
      self._updates.assign_add(1)
