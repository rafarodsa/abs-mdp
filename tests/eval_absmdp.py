'''
    Test learned abstract MDP
    author: Rafael Rodriguez-Sanchez
    date: June 12, 2023
'''

import argparse
import torch
import numpy as np
from envs.pinball.pinball_gym import PinballEnvContinuous
from envs.pinball.controllers_pinball import PinballGridOptions, create_position_options
from envs.env_options import EnvOptionWrapper
from scripts.utils import compute_initiation_masks
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--n-traj', type=int, default=10)
    parser.add_argument('--traj-len', type=int, default=10)
    args = parser.parse_args()


    # load absmdp
    amdp = torch.load(args.path)

    # create ground env
    gmdp = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg')
    options = create_position_options(gmdp)
    gmdp = EnvOptionWrapper(options, gmdp)

    # sample init states
    init_states = gmdp.env.sample_init_states(args.n_traj).astype(np.float32)
    traj_g, traj_a = [], []
    for i in range(args.n_traj):
        t_a, t_g = [], []
        s = gmdp.reset(init_states[i])
        z = amdp.reset(s.astype(np.float32))
        for j in range(args.traj_len):
            # sample action
            # z = amdp.reset(s.astype(np.float32))
            i_s = compute_initiation_masks(s, options)
            probs = i_s/i_s.sum()
            a = np.random.choice(len(options), p=probs)
            # step
            next_s, g_r, done_g, truncated, info = gmdp.step(a)
            next_z, a_r, done_a, _ = amdp.step(a)
            i_z = amdp.initiation_set(torch.from_numpy(z))
            tau = amdp.tau(torch.from_numpy(z), a)

            # collect data
            t_g.append((s, a, g_r, next_s.astype(np.float32), done_g, i_s, info['execution_length']))
            t_a.append((z, a, a_r, next_z, done_a, i_z, tau.item()))

            s = next_s
            z = next_z

        # add
        traj_g.append(t_g)
        traj_a.append(t_a)
 
    # compute stats per trajectory
    s, a, g_r, next_s, done_g, i_s, t = zip(*[t for traj in traj_g for t in traj])
    z, a, a_r, next_z, done_a, i_z, tau = zip(*[t for traj in traj_a for t in traj])

    reward_error = np.abs(np.array(g_r) - np.array(a_r)).reshape(args.n_traj, -1)
    tau_error = np.abs(np.array(t) - np.array(tau)).reshape(args.n_traj, -1)

    # plot errors
    sns.set_theme(style='darkgrid')
    sns.set_palette('colorblind')

    t = np.arange(args.traj_len)
    # sns.lineplot(x=t, y=, data=reward_error, label='reward error')
    plt.subplot(2,2,1)
    # reward_error = pd.DataFrame(reward_error, columns=t)
    # print(reward_error)
    # sns.lineplot(data=reward_error, label='reward error', x=reward_error.columns, y=reward_error.index)
    plt.plot(t, reward_error.mean(0), label='reward error')
    plt.fill_between(t, reward_error.mean(0) - reward_error.std(0), reward_error.mean(0) + reward_error.std(0), alpha=0.2)
    plt.subplot(2,2,2)
    plt.plot(t, tau_error.mean(0), label='tau error')
    plt.fill_between(t, tau_error.mean(0) - tau_error.std(0), tau_error.mean(0) + tau_error.std(0), alpha=0.2)
    

    groundings = [amdp.ground(torch.from_numpy(z)) for z in next_z]
    # grounding_error = [-g.log_prob(torch.from_numpy(s).unsqueeze(0)).numpy()  for s, g in zip(next_s, groundings)]

    mean_grounding_error = [((g.sample(n=20).squeeze().transpose(0, 1).mean(0) - torch.from_numpy(s))**2).numpy() for s, g in tqdm(zip(next_s, groundings))]
    mean_grounding_error = np.array(mean_grounding_error).reshape(args.n_traj, -1, 4)

    mean_vel_error = mean_grounding_error[..., 2:].sum(-1)
    mean_pos_error = mean_grounding_error[..., :2].sum(-1)

    plt.subplot(2,2,3)
    plt.plot(t, mean_vel_error.mean(0), label='vel error')
    plt.fill_between(t, mean_vel_error.mean(0) - mean_vel_error.std(0), mean_vel_error.mean(0) + mean_vel_error.std(0), alpha=0.2)
    plt.subplot(2,2,4)
    plt.plot(t, mean_pos_error.mean(0), label='pos error')
    plt.fill_between(t, mean_pos_error.mean(0) - mean_pos_error.std(0), mean_pos_error.mean(0) + mean_pos_error.std(0), alpha=0.2)
    # plt.show()

    # per action
    actions = np.array(a).reshape(args.n_traj, -1)
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.title('reward error')
    # sns.histplot(x=reward_error[actions==0].flatten(), label='option 0', color='blue', alpha=0.5)
    sns.histplot(x=np.array(g_r)[actions.flatten()==0], label='option 0', color='blue', alpha=0.5)
    sns.histplot(x=np.array(a_r)[actions.flatten()==0], label='option 0', color='orange', alpha=0.5)
    
    plt.subplot(2,2,2)
    sns.histplot(x=reward_error[actions==1].flatten(), label='option 1', color='orange', alpha=0.5)
    plt.subplot(2,2,3)
    sns.histplot(x=reward_error[actions==2].flatten(), label='option 2', color='green', alpha=0.5)
    plt.subplot(2,2,4)
    sns.histplot(x=reward_error[actions==3].flatten(), label='option 3', color='red', alpha=0.5)
    
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.title('tau error')
    sns.histplot(x=tau_error[actions==0].flatten(), label='option 0', color='blue', alpha=0.5)
    plt.subplot(2,2,2)
    sns.histplot(x=tau_error[actions==1].flatten(), label='option 1', color='orange', alpha=0.5)
    plt.subplot(2,2,3)
    sns.histplot(x=tau_error[actions==2].flatten(), label='option 2', color='green', alpha=0.5)
    plt.subplot(2,2,4)
    sns.histplot(x=tau_error[actions==3].flatten(), label='option 3', color='red', alpha=0.5)
    
    plt.show()

