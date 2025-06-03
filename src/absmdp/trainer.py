import pathlib
import os
import pfrl
import time, datetime

from src.absmdp.utils import Every
import numpy as np


def makeoutdirs(config):
    basedir = f'{config.experiment_cwd}/{config.experiment_name}'
    os.makedirs(basedir, exist_ok=True)
    # make world model outdir
    world_model_outdir = f'{basedir}/{config.exp_id}/world_model/'
    os.makedirs(world_model_outdir, exist_ok=True)
    # make agent outdir
    agent_outdir = f'{basedir}/{config.exp_id}/agent'
    os.makedirs(agent_outdir, exist_ok=True)
    agent_outdir = pfrl.experiments.prepare_output_dir(None, agent_outdir, exp_id=config.planner.agent_id, make_backup=False)
    return basedir, world_model_outdir, agent_outdir


class TicToc:
    def __init__(self, alpha=0.98):
        self._init_time = time.perf_counter()
        self._tic = time.perf_counter()
        self._avg = 0.
        self.alpha = alpha

    def tic(self):
        self._tic = time.perf_counter()

    def toc(self):
        self._avg = self.alpha * self._avg + (1-self.alpha) * (time.perf_counter()-self._tic)
    
    def __call__(self):
        return self._avg

class Trainer:
    def __init__(self, config, train_env, test_env, agent, world_model, offline_data=None):
        self.train_env = train_env
        self.test_env = test_env
        self.agent = agent
        self.world_model = world_model
        self.config = config
        self.offline_data=offline_data

        self.max_rollout_len = config.experiment.max_rollout_len
        self.prefill = config.experiment.prefill
        self.gradient_steps = config.experiment.gradient_steps if 'gradient_steps' in config.experiment else 1

        self.should_train_wm = Every(config.world_model.train_every)
        self.should_train_agent = Every(config.planner.train_every)
        self.should_checkpoint = Every(config.experiment.checkpoint_frequency)
        self.should_evaluate = Every(config.experiment.eval_interval)

        # stats
        self.episode_len = 0
        self.episode_count = 0
        self.max_steps = config.experiment.steps

    def setup(self):
        # mkdirs
        device = f'cuda:{self.config.experiment.gpu}' if self.config.experiment.gpu >= 0 else 'cpu'
        _, world_model_outdir, agent_outdir = makeoutdirs(self.config)
        self.load_checkpoint(world_model_outdir, device=self.config.fabric.accelerator)

        self.agent.setup(agent_outdir)
        self.world_model.set_outdir(world_model_outdir)
        self.world_model.setup_trainer(self.config)
        self.world_model.setup_replay(self.offline_data)
        
        self.agent.to(device)

    def checkpoint(self):
        self.world_model.save_checkpoint()

    def load_checkpoint(self, ckpt_path, device):
        if (pathlib.Path(ckpt_path) / 'checkpoints/world_model.ckpt').exists():
            warmup_steps = self.world_model.warmup_steps
            sample_transition = self.world_model.sample_transition
            self.world_model = self.world_model.load_checkpoint(pathlib.Path(ckpt_path) / 'checkpoints/world_model.ckpt', device=device)
            print(f"Loading checkpoint at {pathlib.Path(ckpt_path) / 'checkpoints/world_model.ckpt'}")
            self.world_model.warmup_steps = warmup_steps
            self.world_model.sample_transition = sample_transition
    
    def log(self):
        pass
        

    def train(self):
        episode_return = []
        episode_len = []
        episode_count = 0
        timer = TicToc()

        ss = self.train_env.reset()
        timer.tic()

        while self.world_model.timestep < self.max_steps:
            
            # rollout & observe
            with self.agent.eval_mode():
                actions = self.agent.act(ss, self.world_model.encode)
            next_ss, rs, dones, infos = self.train_env.step(actions) 

            if len(episode_return) == 0:
                # initialize
                episode_return = [0. for _ in range(len(rs))]
                episode_len = [0. for _ in range(len(rs))]
            episode_return = [rs + r for rs, r in zip(episode_return, rs)]
            episode_len = [ep_len + 1 for ep_len in episode_len]
            env_rewards = [info['env_reward'] for info in infos]
            taus = [info['tau'] for info in infos]
            successes = [info['success'] for info in infos]

            last = np.logical_or(dones, np.array(episode_len) >= self.max_rollout_len)
            self.world_model.observe(ss, actions, env_rewards, next_ss, dones, taus, successes, info=infos, last=last)

            ss = self.train_env.reset(~last)

            for i in range(len(episode_len)):
                self.world_model.timestep += 1
                if last[i]:
                    episode_count += 1
                    ground_log = {
                        'ground_env/episode_return': episode_return[i],
                        'ground_env/episode_length': episode_len[i],
                        'ground_env/success': float(infos[i]['goal_reached'])
                    }
                    # log 
                    print(f'[rollout] timestep {self.world_model.timestep} episodes {len(self.world_model.data)}, length {episode_len[i]}, return {episode_return[i]}')
                    self.world_model.log(ground_log, self.world_model.timestep)
                    episode_len[i] = 0
                    episode_return[i] = 0

            
            # train
            if self.should_train_wm(self.world_model.timestep) and len(self.world_model.data) > self.prefill:
                timer.toc()
                estimate = timer() * (self.max_steps - self.world_model.timestep) / self.config.world_model.train_every
                print(f'Average time per iteration {timer():.2f}s. Estimated time {datetime.timedelta(seconds=estimate)}')
                timer.tic()

                self.world_model.train_world_model(timestep=self.world_model.timestep, steps=self.gradient_steps)
            
            if self.should_train_agent(self.world_model.timestep) and self.world_model.n_gradient_steps > self.world_model.warmup_steps:
                ep_logs = self.agent.train(self.world_model, steps_budget=self.config.planner.agent.rollout_len)
                self.world_model.log(ep_logs, step=self.world_model.timestep)
                print(f'[simulation stats] {" | ".join([f"{k}: {v}" for k,v in ep_logs.items()])}')
            # evaluate
            if self.should_evaluate(self.world_model.timestep):
                stats = self.agent.evaluate(self.test_env, self.world_model.encode, self.config.experiment.eval_n_runs, timestep=self.world_model.timestep)
                print(f"[evaluation stats] reward {stats['mean']} | length {stats['length_mean']}")
            # stats
            if self.should_checkpoint(self.world_model.timestep):
                self.checkpoint()
            
            
        # finalize