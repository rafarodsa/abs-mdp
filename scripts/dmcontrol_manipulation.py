import dm_control.manipulation as manipulation
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import product

env = manipulation.load('reach_site_vision', seed=42)
action_spec = env.action_spec()

def sample_random_action():
  return env.random_state.uniform(
      low=action_spec.minimum,
      high=action_spec.maximum,
  ).astype(action_spec.dtype, copy=False)

def display_video(frames, fps):
  # frames in numpy array
  for i in range(frames.shape[0]):
    Image.fromarray(frames[i])
    # show at frame rate
    plt.imshow(frames[i])
    plt.axis('off')
    plt.show(block=False)
    plt.pause(1/fps)

def prod(t):
  _r = 1
  for tt in t:
    _r *= tt
  return _r


# print('action_spec', action_spec)
# obs_dim = [prod(o.shape) for o in env.observation_spec().values()]
# print('obs_dim', obs_dim)
# print('obs_dim', sum(obs_dim))


print(manipulation.ALL)

# Step the environment through a full episode using random actions and record
# the camera observations.
frames = []
timestep = env.reset()
frames.append(timestep.observation['front_close'])
while not timestep.last():
  timestep = env.step(sample_random_action())
  print(timestep.observation.keys())
  frames.append(timestep.observation['front_close'])
  # print(timestep.observation)
print(frames[0])
all_frames = np.concatenate(frames, axis=0)
display_video(all_frames, 60)