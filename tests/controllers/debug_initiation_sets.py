import envs.pinball.controllers_pinball as pb
from envs.pinball.pinball_gym import PinballEnvContinuous as Pinball
import numpy as np
from src.utils.printarr import printarr


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

obs = env.vertices[4]
edges = np.stack([obs[:-1], obs[1:]], axis=-1)
coeffs = pb._intersect_batch(edges, s[3:4, :2], predicted_s[3:4])
printarr(coeffs[0])
coeffs = np.stack(coeffs, axis=-1)
print(coeffs)


check_ = pb._intersect_obstable(obs, initial_position=s[:, :2], final_position=predicted_s)
print(check_)

import ipdb
ipdb.set_trace()