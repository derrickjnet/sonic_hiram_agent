import random
import numpy as np
import retrowrapper
from sonic_mod import AllowBacktracking, make_env


level_choice = 1
local_env = True

def move(env, num_steps, left=False, jump_prob=1.0 / 10.0, jump_repeat=4):
    """
    Move right or left for a certain number of steps,
    jumping periodically.
    """
    total_rew = 0.0
    done = False
    steps_taken = 0
    jumping_steps_left = 0
    while not done and steps_taken < num_steps:
        action = np.zeros((12,), dtype=np.bool)
        action[6] = left
        action[7] = not left
        if jumping_steps_left > 0:
            action[0] = True
            jumping_steps_left -= 1
        else:
            if random.random() < jump_prob:
                jumping_steps_left = jump_repeat - 1
                action[0] = True
        _, rew, done, _ = env.step(action)
        total_rew += rew
        steps_taken += 1
        if done:
            break
    return total_rew, done

if __name__ == "__main__":
    env1 = make_env(local=local_env, level_choice=2)
    env1 = AllowBacktracking(env1)
    env1.agent = 'JERK1'
    # env2 = make_env(local=local_env, level_choice=2)
    # env2 = AllowBacktracking(env2)
    # env2.agent = 'JERK2'
    _obs = env1.reset()
    new_ep = True


    while env1.total_steps_ever <= 1000000:
        total_rew1, done1 = move(env1, 15)
        print(env1.done)
        if done1:
            env1.reset()


