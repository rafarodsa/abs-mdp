import envs.pinball.controllers_pinball as pb
from envs.pinball.pinball_gym import PinballEnvContinuous as Pinball
import numpy as np
from src.utils.printarr import printarr
import matplotlib.pyplot as plt
import torch


configuration_file = "envs/pinball/configs/pinball_simple_single.cfg"
env = Pinball(config=configuration_file, width=25, height=25, render_mode='rgb_array') 

# s = np.array([[ 0.255,  0.675,  0.008, -0.016],
#             [ 0.262,  0.673, -0.05,  -0.002]])
# predicted_s = np.array([[0.295, 0.675],
#                          [0.302, 0.673]])

s = np.array([[ 2.483e-01,  6.637e-01, 4.614e-03,  6.093e-02],
            [ 2.509e-01,  6.791e-01, -6.687e-05,  9.746e-02],
            [ 2.505e-01,  6.788e-01,  1.577e-03, -2.195e-01],
            [ 2.538e-01,  6.595e-01,  2.912e-10, -2.930e-01],
            [ 2.619e-01,  6.811e-01,  1.574e-03,  1.449e-01]])
predicted_s = np.array([[0.288, 0.664],
                        [0.291, 0.679],
                        [0.29,  0.679],
                        [0.294, 0.66 ],
                        [0.302, 0.681]])


s_no_options = np.array([[0.79190504, 0.29285348, 0.28305147, 0.],
                         [2.80796641e-01,  6.64116683e-01, -1.08741220e-04, -4.37372117e-02]])


errors = torch.load('errors.pt')
s, predicted_s, _ = zip(*errors)

s, predicted_s = np.concatenate(s), np.concatenate(predicted_s)

printarr(s, predicted_s)

_s = np.stack([s[:, :2], predicted_s], axis=-1)
ball_radius = 0.02
dis = _s[..., 1] - _s[..., 0]
dis = dis/np.linalg.norm(dis, axis=-1, keepdims=True)
normal_vector = np.zeros_like(dis)
normal_vector[..., 0] = -dis[..., 1]
normal_vector[..., 1] = dis[..., 0]

printarr(s, predicted_s, _s, normal_vector)

obs = env.vertices[4]
expanded_obs = env.expanded_obs[4]
_, ax = plt.subplots()
plt.plot(obs[:, 0], obs[:, 1])
plt.plot(expanded_obs[:, 0], expanded_obs[:, 1])

points = range(_s.shape[0])
# points = range(3,4)
for i in points:
    plt.scatter(_s[i, 0], _s[i, 1], c='b', s=10)
    plt.plot(_s[i, 0], _s[i, 1], c='r')
    circle = plt.Circle(_s[i, ..., 1], env.pinball.ball.radius, color='k', fill=False)
    ax.add_artist(circle)

# _s[..., 1] = _s[..., 1] + ball_radius*dis

for i in points:
    plt.scatter(_s[i, 0], _s[i, 1], c='b', s=10)
    plt.plot(_s[i, 0], _s[i, 1], 'r--')

# lim_s = _s + ball_radius*normal_vector[..., None]
# lim_i = _s - ball_radius*normal_vector[..., None]
# for i in points:
#     plt.scatter(lim_s[i, 0], lim_s[i, 1], c='b', s=10)
#     plt.plot(lim_s[i, 0], lim_s[i, 1], 'k:')
#     plt.scatter(lim_i[i, 0], lim_i[i, 1], c='b', s=10)
#     plt.plot(lim_i[i, 0], lim_i[i, 1], 'k-.')


edges = np.stack([obs[:-1], obs[1:]], axis=-1)
# coeffs = pb._intersect_batch(edges, s, predicted_s)
print(s, predicted_s)
close_by = np.zeros((s.shape[0],), dtype=bool)
for obs in env.vertices:
    displacement = predicted_s - s[:, :2]
    dis = np.linalg.norm(displacement, axis=-1, keepdims=True)
    dis, coeff = pb.distances_path_to_vertices(obs[:-1], s[:, :2], predicted_s + displacement * 0/dis)
    close_by = np.logical_or(np.any(dis <= ball_radius, axis=0), close_by)
    plt.plot(obs[:, 0], obs[:, 1], 'k--')


distances = []
for obs in env.vertices:
    displacement = predicted_s - s[:, :2]
    dis = np.linalg.norm(displacement, axis=-1, keepdims=True)
    distance_goal_to_edge, intersect_coeff = pb.distances_path_to_vertices(predicted_s + displacement * ball_radius/dis, obs[:-1], obs[1:])
    distances.append(distance_goal_to_edge)
print(distances)


_, ax = plt.subplots()
for obs in env.vertices:
    plt.plot(obs[:, 0], obs[:, 1], c='k')
for obs in env.expanded_obs:
    plt.plot(obs[:, 0], obs[:, 1], c='r')
points = range(s_no_options.shape[0])
step_size = 1/25
ball_radius = 0.01
for i in points:
    circle = plt.Circle(s_no_options[i], ball_radius)
    distance = plt.Circle(s_no_options[i], step_size, color='b', fill=False)
    distance_min = plt.Circle(s_no_options[i], step_size-ball_radius, color='g', fill=False)
    distance_max = plt.Circle(s_no_options[i], step_size+ball_radius, color='g', fill=False)
    ax.add_artist(circle)
    ax.add_artist(distance)
    ax.add_artist(distance_min)
    ax.add_artist(distance_max)
    plt.scatter(s_no_options[i, 0], s_no_options[i, 1], c='b', s=10)
    plt.plot(s_no_options[i, 0], s_no_options[i, 1], c='r')




# print(~close_by)
# print(s[~close_by, :2])
# printarr(coeffs[0])
# coeffs = np.stack(coeffs, axis=-1)
# print(coeffs)
# check_ = pb._intersect_obstable(obs, initial_position=s[:, :2], final_position=predicted_s)
# print(check_)
# plt.show()

import ipdb; ipdb.set_trace()