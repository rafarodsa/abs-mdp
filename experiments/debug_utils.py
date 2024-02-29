import numpy as np
import lightning as L



def load_wm_buffer(ckpt):
    fabric = L.Fabric()
    loaded_ckpt = fabric.load(ckpt, {})
    data = loaded_ckpt['data']
    return data

def get_trajectories_from_buffer(databuffer, low=0, high=10000):
    states = []
    high = min(len(databuffer), high)
    goal_reached = []
    for i in range(low, high):
        traj = databuffer.buffer[i]
        s = [t[0] for t in traj] + [traj[-1][3]]
        states.append(np.array(s))
        goal_reached.append(traj[-1][-1])
    return states

def plot_trajs(states, goal):
    for s in states:
        plt.plot(s[..., 0], s[..., 1], linewidth=0.1, c='g')
        plt.scatter(s[..., 0], s[..., 1], s=0.5, c='k')
        plt.scatter(s[0, 0], s[0, 1], s=5, c='r')
        plt.scatter(s[-1, 0], s[-1, 1], s=5, c='b')
    plt.scatter(goal[0], goal[1], s=20)
    plt.savefig('test.png')

if __name__=='__main__':
    import matplotlib.pyplot as plt
    data = load_wm_buffer('exp_results/antmaze/antmaze-umaze-v2/absmdps/plan_awm_dim2_goals_gaussian_nstep3_50k/world_model/plan_with_model_umaze_dim2_gaussian_nstep3_007__planner.env.goal_6__experiment.seed_31__planner.agent.lr_1e-5/checkpoints/world_model.ckpt')
    states = get_trajectories_from_buffer(data)
    goal = [5., 7.5]
    plot_trajs(states[1500:1510], goal)
    
    import ipdb; ipdb.set_trace()