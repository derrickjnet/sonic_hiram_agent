#env
from retro_contest.local import make
import gym
import gym_remote.client as grc
import gym_remote.exceptions as gre
import argparse
#visual
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
import imutils
#ml
import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Dropout
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import tensorflow as tf

levels = ['SpringYardZone.Act3',
'SpringYardZone.Act2',
'GreenHillZone.Act3',
'GreenHillZone.Act1',
'StarLightZone.Act2',
'StarLightZone.Act1',
'MarbleZone.Act2',
'MarbleZone.Act1',
'MarbleZone.Act3',
'ScrapBrainZone.Act2',
'LabyrinthZone.Act2',
'LabyrinthZone.Act1',
'LabyrinthZone.Act3']

#global

EMA_RATE = 0.20
EXPLOIT_BIAS = 0.20
TOTAL_TIMESTEPS = int(1e6)
last_obs = []
is_stuck_img = []
is_stuck_pos = [0]
max_rewards = []


def main():
    """Run JERK on the attached environment."""

    LEVEL = levels[random.randrange(0, 13, 1)]
    env = make(game='SonicTheHedgehog-Genesis', state=LEVEL)
#    env = grc.RemoteEnv('tmp/sock')
    env = TrackedEnv(env)
    print (env.observation_space)
    print(LEVEL)
    new_ep = True
    solutions = []
    while True:

        if new_ep:
            if (solutions and
                    random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
                solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                best_pair = solutions[-1]
                new_rew = exploit(env, best_pair[1])
                best_pair[0].append(new_rew)
                max_rewards.append((new_rew))
                print('replayed best with reward %f and max rewards: %f' % (new_rew, max(max_rewards)))

                continue
            else:
                env.reset()
                new_ep = False
        rew, new_ep = move(env, 100)
        if not new_ep and rew <= 0:
            #print('backtracking due to negative reward: %f' % rew)
            _, new_ep = move(env, 70, left=True)
        if new_ep:
            solutions.append(([max(env.reward_history)], env.best_sequence()))

def stand_still(env):
    #used to observe the static environment
    action = np.zeros((12,), dtype=np.bool)
    obs, rew, done, info = env.step(action)
    return obs,rew

def power_up(env):
    #used to power up
    action = np.zeros((12,), dtype=np.bool)
    action[5] = True
    action[0] = True
    obs, rew, done, info = env.step(action)
    return obs,rew

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
        obs, rew, done, info = env.step(action)
        if left:
            im_stuck(env,obs,info)
        env.render() #See the game played
        total_rew += rew
        steps_taken += 1
        if done:
            capture_moment(obs,'died')
            break

    return total_rew, done


def im_in_control(env,obs):
    action = np.zeros((12,), dtype=np.bool)
    obs2, rew, _, _ = env.step(action)
    img1 = obs[1:224, 100:320, :]
    img2 = obs2[1:224, 100:320, :]
    is_fate = np.array_equal(img1,img2)
    print('fate:',is_fate)
    return is_fate

def im_stuck(env, obs,info):
    obs2, _, _, _ = env.step(np.zeros((12,), dtype=np.bool))
    obs2 = obs2[1:224, 100:320, :]
    obs1 = obs[1:224, 100:320, :]
    third_obs1 = obs[1:224, 160:320, :]
    is_stuck_img.append(obs1)

    if not np.array_equal(obs1,obs2):
        if max(is_stuck_pos) <  info['screen_x']:
            is_stuck_pos.append(info['screen_x'])
            is_stuck_img.append(obs1)
            capture_moment(obs1,'stranded')
            capture_moment(third_obs1, 'stranded')
            im_in_control(env,obs)

    if bool(random.getrandbits(1)):
        stand_still(env)
        if bool(random.getrandbits(1)):
            power_up(env)

def capture_moment(img, moment='unknown'):
    plt.imshow(img)
    plt.show()

def exploit(env, sequence):
    """
    Replay an action sequence; pad with NOPs if needed.

    Returns the final cumulative reward.
    """
    env.reset()
    done = False
    idx = 0
    print('replaying this many steps %f' % len(sequence))
    while not done:
        if idx >= len(sequence):
            _, _, done, _ = env.step(np.zeros((12,), dtype='bool'))
        else:
            obs, _, done, _ = env.step(sequence[idx])
        idx += 1
    return env.total_reward


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01

class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        self.total_steps_ever = 0

    def best_sequence(self):
        """
        Get the prefix of the trajectory with the best
        cumulative reward.
        """
        max_cumulative = max(self.reward_history)
        for i, rew in enumerate(self.reward_history):
            if rew == max_cumulative:
                return self.action_history[:i+1]
        raise RuntimeError('unreachable')

    # pylint: disable=E0202
    def reset(self, **kwargs):
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.total_steps_ever += 1
        self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        return obs, rew, done, info

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)