import numpy as np
from envs.pinball.pinball_gym import PinballEnvContinuous
from envs.pinball.controllers_pinball import initiation_set, initiation_set_batch
from src.utils.printarr import printarr

def test_initiation_set():
    env = PinballEnvContinuous(config='envs/pinball/configs/pinball_simple_single.cfg', render_mode='human')
    # sample initial states
    s = env.sample_initial_positions(1)[0]
    printarr(s)
    # create initiation set
    distance = np.array([1/20, 0])
    init_set = initiation_set(env, distance=distance)
    
    # compute initiation for each state
    if len(s.shape) == 1:
        initiation = init_set(s)
    else:
        initiation = [init_set(s[i]) for i in range(s.shape[0])]

    init_set_batch = initiation_set_batch(env, distance=distance)
    initiation_batch = init_set_batch(s)
    # print(initiation, initiation_batch)
    assert np.all(initiation == initiation_batch)


if __name__=='__main__':
    test_initiation_set()

