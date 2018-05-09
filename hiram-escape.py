#env
from retro_contest.local import make
import gym
import gym.spaces as spaces
import gym_remote.client as grc
import gym_remote.exceptions as gre
import argparse
#db
#visual
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
import imutils
#ml
import numpy as np
import random
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Dropout
from keras.optimizers import Adam
from collections import deque
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import tensorflow as tf

# Author: Derrick L. Johnson (derrickj@derrickj.net)
#Hiram's Agent - Blends exploration with DQN
#  Initialize Q(s, a) arbitrarily
# Repeat (for each episode):
# 	Initialize s
# 	Choose a from s using policy derived from Q
# 	While (s is not a terminal state):
# 		Take action a, observe r, s'
# 		Choose a' from s' using policy derive from Q
# 		Q(s,a) += alpha * (r + gamma * max,Q(s', a') - Q(s,a))
# 		s = s', a = a'
# Initialize Q(s,a) arbitrarily
# 		Choose a from s using policy derived from Q
# 		Take action a, observe r, s'
# 		Q(s,a) += alpha * (r + gamma * max,Q(s') - Q(s,a))
# 		s = s'


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
is_stuck_img = []
is_stuck_pos = [0]
max_rewards = []
experiences = []
def main():
    LEVEL = levels[random.randrange(0, 13, 1)]
    print('Begining: ',LEVEL)
    #Environment Classes
    stack = True
    scale_rew = True
    warp = True
    env = make(game='SonicTheHedgehog-Genesis', state=LEVEL)
    # env = grc.RemoteEnv('tmp/sock')
    print('initial')
    debug(env)
    env = TrackedEnv(env)

    # ML Varables:
    gamma = 0.9
    epsilon = .95
    jerk_attempts = 15
    trial = 0
    trials = 1000
    trial_len = 500
    updateTargetNetwork = 1000
    steps = []
    new_ep = True
    solutions = []
    done = False
    game_reward = True
    steps_taken = 0
    env.last_observation = []
    while True and steps_taken < TOTAL_TIMESTEPS:

        if new_ep:
            if (solutions and
                    random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
                solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                best_pair = solutions[-1]
                new_rew = exploit(env, best_pair[1])
                # env.replay()
                best_pair[0].append(new_rew)
                max_rewards.append((new_rew))
                print('replayed best with reward %f and max rewards: %f' % (new_rew, max(max_rewards)))
                continue
            else:
                env.reset()
                new_ep = False
        rew, new_ep = move(env,100)
        if not new_ep and rew <= 0:
            #\print('backtracking due to negative reward: %f' % rew)
            _, new_ep = move(env, 70, left=True)
            continue
        if new_ep:
            solutions.append(([max(env.reward_history)], env.best_sequence()))
        trial += 1
        if steps_taken >= 199:
            print("Failed to complete in trial")
            if steps_taken % 10 == 0:
                env.save_model("trial-{}.model")
            else:
                print("Completed in {} trials".format(trial))
                env.save_model("success.model")
                break

def debug(env):
    print ('Obs Size',env.observation_space)
    print('Action Space',env.action_space)

def move(env,num_steps, left=False, jump_prob=0.6 / 10.0, jump_repeat=4):
    """
    Move right or left for a certain number of steps,
    jumping periodically.
    """
    done = False
    steps_taken = 0
    jumping_steps_left = 0
    while not done and steps_taken < num_steps:
        cur_state = env.last_observation # Last captured environment
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
        new_state = obs
        env.remember(cur_state,action,rew,new_state,done) #maybe move?
        env.render()
        if left:
            im_stuck(env,cur_state,new_state)
        if done:
            capture_moment(obs, 'died')
            break

    return env.total_reward,done

def im_stuck(env,cur_state, new_state):
    current_x = env._cur_x
    if max(is_stuck_pos) < current_x:
        is_stuck_pos.append(current_x)
        capture_moment(cur_state)
        capture_moment(new_state)


def capture_moment(img, moment='unknown'):
    plt.imshow(img)
    plt.show()
    return

def exploit(env, sequence):
    """
    Replay an action sequence; pad with NOPs if needed.

    Returns the final cumulative reward.
    """
    env.reset()
    done = False
    idx = 0
    while not done:
        if idx >= len(sequence):
            _, _, done, _ = env.step(np.zeros((12,), dtype='bool'))
        else:
            obs, _, done, _ = env.step(sequence[idx])
        idx += 1
    return env.total_reward


#SEGA
class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        # self.env = env
     #Game History
        self.action_history = []
        self.reward_history = []
        self.sonic_history = []
        self.attempts = 1
        self.attempts_step = 0
        self.total_reward = 0
        self.total_steps_ever = 0
        self.im_stranded = False

    #Backtracking:
        self._cur_x = 0
        self._max_x = 0
    #RL variables for agent init
    # (sourced:https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125
    #Model creation and target creation
        self.model = self.create_model()
        self.target_model = self.create_model()
    #Sonic stuff - Later

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Conv2D(32, (3, 3), input_shape=state_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="binary_crossentropy", #"mean_squared_error"
                      optimizer='rmsprop', #Adam(lr=self.learning_rate)
                      metrics=['accuracy'])
        return model

    #A random act
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    #Learning
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        print('attempting replay')
        batch_size = 8
        if len(self.memory) < batch_size:
            return
        print('replaying')
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state[0])
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    #Training
    def target_train(self):
        print('train')
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    #Saving
    def save_model(self, fn):
        self.model.save(fn)

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
        self._cur_x = 0
        self._max_x = 0
        self.attempts_step = 0
        return self.env.reset(**kwargs)

    #Make Sonic Step
    def step(self, action):
        self.total_steps_ever += 1
        #self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        self.last_observation = obs
        self.attempts_step += 1
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        self._cur_x += rew
        self.rew1 = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        self.sonic_history.append({ 'attempt':self.attempts
        , 'step':self.attempts_step, 'action': [], 'current_x':int(self._cur_x)
        , 'reward':int(rew),'reward_2':int(self.rew1),  'total_reward':int(self.total_reward)
        , 'max_reward':int(self._max_x), 'stuck': self.im_stranded, 'done':done, 'img':'no'})
        if done or self.im_stranded:
            #print(self.sonic_history[-3:])
            print(max(self.reward_history))
            if done:
                self.attempts += 1
                self.attempts_step = 0

        return obs, rew, done, info

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)