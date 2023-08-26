'''
    Train abstract world model online
    author: Rafael Rodriguez-Sanchez
    date: 24 August 2023
'''



import logging
import os

from pfrl.experiments.evaluator import save_agent
from pfrl.utils.ask_yes_no import ask_yes_no

from src.agents.evaluator import Evaluator
from src.utils.printarr import printarr


''' 
    ground_env: ground environment is an environment with options
    world_model: abstract world model
    task_reward: reward function for the task
    config: configuration object
'''

def evaluate_agent_if_necessary(grounded_agent, ground_env, n_episodes):
    """
        run evaluation episodes
    """
    pass

def rollout(grounded_agent, ground_env, world_model, task_reward, n_steps):
    """
        act in ground environment
        store transitions in world model
    """
    timestep = 0
    s = ground_env.reset()
    while timestep < n_steps:
       
       
        a = grounded_agent.act(s)
        next_s, r, done, info = ground_env.step(a)
        r = task_reward(s, a, next_s, info)
        info['task_reward'] = r
        world_model.observe(s, a, r, next_s, done, info['tau'], info['success'], info=info)

        s = next_s
        timestep += 1
        if done:
            s = ground_env.reset()
            world_model.end_episode()


        # checkpointing and logging

    return timestep



def train_world_model(world_model, gradient_steps):
    '''
        Run training steps on world model
    '''
    for _ in range(gradient_steps):
        world_model.train()


def train_agent_in_simulation(agent, world_model, steps):
    """
        train agent in simulation using world model
    """
    timestep = 0

    # update agent
    z = world_model.reset()
    initset_fn = world_model.get_initset_fn()
    
    while timestep < steps:
        initset_z = initset_fn(z)
        a = agent.act(z, initset_z)
        next_z, r, done, info = world_model.step(a)
        agent.observe(z, a, r, next_z, done, info)
        z = next_z
        timestep += 1
        if done:
            z = world_model.reset()
            agent.stop_episode()




def train_agent_with_evaluation(
                                grounded_agent, 
                                ground_env, 
                                world_model, 
                                task_reward,
                                max_steps,
                                config
                            ):
    
    ## Boilerplater: set up logging, evaluator, and checkpointing

    ## main training loop
    agent = grounded_agent.agent
    timesteps = 0  # abstract timesteps
    while timesteps < max_steps:
        
        ## rollout agent in ground environment
        t = rollout(grounded_agent, ground_env, world_model, task_reward, episodes=1)
        timesteps += t
        
        ## train world model
        train_world_model(world_model, gradient_steps=config.world_model_training_steps)
        
        ## train agent
        train_agent_in_simulation(agent, world_model, steps=config.agent_training_steps)
        
        ## evaluate agent        
        evaluate_agent_if_necessary(grounded_agent, ground_env, n_episodes=config.eval_n_episodes)
