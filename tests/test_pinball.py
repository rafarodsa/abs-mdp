from matplotlib import pyplot as plt
from src.envs.pinball.pinball_gym import PinballEnv


pinball = PinballEnv("/Users/rrs/Desktop/abs-mdp/src/envs/pinball/configs/pinball_simple_single.cfg", render_mode="human")
points = pinball.sample_initial_positions(10000)
plt.scatter(points[:, 0], points[:, 1], s=1)
plt.show()