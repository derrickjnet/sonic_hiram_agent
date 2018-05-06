#!/usr/bin/env python
# Human Preferences for
# DQN Lib: https://github.com/keon/deep-q-learning/blob/master/dqn.py
# Play Lib: https://raw.githubusercontent.com/openai/gym/master/gym/utils/play.py
# DL from Human Pref: https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/
# DL from Human Pref: https://arxiv.org/abs/1706.03741
# Implicit Imitation: https://www.aaai.org/Papers/JAIR/Vol19/JAIR-1916.pdf

# ENV_LOCAL
from retro_contest.local import make
# ENV_REMOTE
import gym_remote.client as grc
import gym_remote.exceptions as gre
# ENV_PLUS
import gym
import sqlite3
import random
import time
# FIFO
from collections import deque
# ML/RL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage
from scipy import ndimage as ndi
from skimage import feature
from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from anyrl.algos import DQN
from anyrl.models import MLPQNetwork, EpsGreedyQNetwork
from anyrl.rollouts import BasicPlayer, UniformReplayBuffer
from anyrl.spaces import gym_space_vectorizer

# Load ML
# trained_model = load_ml_model('reward.ml')

# GLOBALS
local_env = True
EMA_RATE = 0.2
EXPLOIT_BIAS = 0.15
TOTAL_TIMESTEPS = int(1e6)
completion = 9000  # Estimated End
batches = 4
# References
# https://github.com/keon/deep-q-learning/blob/master/dqn.py


# Create Storage
conn = sqlite3.connect('retro.db')
db = conn.cursor()

# Setup Storage
stats_col = ["level", "episode", "steps", "cur_action", "prev_action", "acts1", "acts3", "acts5", "acts7", "acts9",
             "acts11"
    , "acts33", "safety", "esteem", "belonging", "potential", "human", "total_reward"]
df = pd.DataFrame(columns=stats_col)
df.to_sql('game_stats', conn, if_exists='replace')

sarsa_col = ["level", "action_cluster", "rewards", "last_reward", "esteem"]
sarsa = pd.DataFrame(columns=sarsa_col)
sarsa.to_sql('sarsa', conn, if_exists='replace')


def main():
    if local_env:  # Select Random Level if local
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
        level_choice = levels[random.randrange(0, 13, 1)]
        env = make(game='SonicTheHedgehog-Genesis', state=level_choice)
    else:
        env = grc.RemoteEnv('tmp/sock')
    env = TrackedEnv(env)
    if level_choice:
        env.level_choice = level_choice
    env.reset()  # Initialize Gaming Environment
    new_ep = True  # New Episode Flag
    solutions = []  # Track Solutions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(state_size, action_size)
    env.assist = False
    env.trainer = False  # Begin with mentor led exploration
    env.resume_rl(False)  # Begin with RL exploration
    BUFFER_SIZE = 1024
    MIN_BUFFER_SIZE = 256
    STEPS_PER_UPDATE = 1
    ITERS_PER_LOG = 200
    BATCH_SIZE = 64
    EPSILON = 0.1
    LEARNING_RATE = 0.001

    while env.total_steps_ever <= TOTAL_TIMESTEPS:  # Interact with Retro environment until Total TimeSteps expire.

        if env.resume:
            tf.reset_default_graph()
            print('Running DQN from:', env.total_reward)
            with tf.Session() as sess:
                def make_net(name):
                    return MLPQNetwork(sess,
                                       env.action_space.n,
                                       gym_space_vectorizer(env.observation_space),
                                       name,
                                       layer_sizes=[32])

                dqn = DQN(make_net('online'), make_net('target'))
                player = BasicPlayer(env, EpsGreedyQNetwork(dqn.online_net, EPSILON),
                                     batch_size=STEPS_PER_UPDATE)
                optimize = dqn.optimize(learning_rate=LEARNING_RATE)

                sess.run(tf.global_variables_initializer())

                dqn.train(num_steps=1000,
                          player=player,
                          replay_buffer=UniformReplayBuffer(BUFFER_SIZE),
                          optimize_op=optimize,
                          target_interval=200,
                          batch_size=64,
                          min_buffer_size=200,
                          handle_ep=lambda _, rew: print('got reward: ' + str(rew)))

                sess.close()

        if env.trainer:
            keys = getch()
            if keys == 'A':
                env.control(-1)
            if keys == 'B':
                env.control(4)
            if keys == 'C':
                env.control(3)
            if keys == 'D':
                env.control(2)
            if keys == 'rr':
                env.trainer = False
                continue
            if keys == ' ':
                env.close()
                env = make(game='SonicTheHedgehog-Genesis', state=levels[random.randrange(0, 13, 1)])
                env = TrackedEnv(env)
                env.reset()  # Initialize Gaming Environment
            print('Entering Self Play')

        if new_ep:  # If new episode....
            if (solutions and
                    random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
                solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                best_pair = solutions[-1]
                new_rew = exploit(env, best_pair[1])
                best_pair[0].append(new_rew)
                print('replayed best with reward %f' % new_rew)
                print(best_pair[0])
                continue
            else:
                env.reset()
                new_ep = False
        print("Running Jerk from:", env.total_reward)
        rew, new_ep = move(env, 100)
        if not new_ep and rew <= 0:
            env.resume_rl()
            # _, new_ep = move(env, 70, False)
        if new_ep:
            solutions.append(([max(env.reward_history)], env.best_sequence()))


def getch():  # Enable Keyboard
    import sys, tty, termios
    old_settings = termios.tcgetattr(0)
    new_settings = old_settings[:]
    new_settings[3] &= ~termios.ICANON
    try:
        termios.tcsetattr(0, termios.TCSANOW, new_settings)
        ch = sys.stdin.read(1)
        print(ch)
    finally:
        termios.tcsetattr(0, termios.TCSANOW, old_settings)

    return ch


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
        if left:
            action = env.control(0)
        else:
            action = env.control(1)
        if jumping_steps_left > 0:
            action = env.control(6)
            jumping_steps_left -= 1
        else:
            if random.random() < jump_prob:
                jumping_steps_left = jump_repeat - 1
                action = env.control(6)
        _, rew, done, _ = env.step(action)
        total_rew += rew
        steps_taken += 1
        if done:
            break
    return total_rew, done

def capture_moment(env,frames,done):
    if done:
        env.in_danger_pos.append(env.total_reward)
        img = env.observation_history[-1]
        show_moment(img)
    elif env.total_reward > max(env.is_stuck_pos):
        env.is_stuck_pos.append(env.total_reward)
        img = env.observation_history[-1]
        show_moment(img)
        # img = env.observation_history[frames+5]
        # show_moment(img)

def show_moment(img):
    plt.imshow(img)
    plt.show()


def exploit(env, sequence):
    """
    Replay an action sequence; pad with NOPs if needed.

    Returns the final cumulative reward.
    """
    state = env.reset()
    done = False
    idx = 0
    env.rl = False
    while not done:
        if idx >= len(sequence):
            env.resume_rl()
            done = True
        else:
            new_state, rew, done, _ = env.step(sequence[idx])
            state = new_state
        idx += 1
    env.total_replays += 1
    env.replay_reward_history.append(env.total_reward)
    return env.total_reward

#DEFINE GYM
class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_choices = self._actions
        self.action_space = gym.spaces.Discrete(len(self._actions))

        self.level_choice = ''
        # Toggles
        self.assist = False  # Allow human trainer
        self.trainer = False  # Enables and tracks training
        self.rl = False  # Enables RL exploration
        self.resume = False  # Resumes RL exploration
        self.game_over = False
        # Env Counters
        self.episode = 1
        self.total_steps_ever = 0
        self.total_reward = 0
        self.rl_total_reward = 0
        self.total_replays = 0
        # Position Counters
        self._cur_x = 0
        self._max_x = 0
        # Scene Trackers
        self.action_history = []
        self.reward_history = []
        self.seq_replay = []
        self.rl_reward_history = []
        self.replay_reward_history = []
        # Maslow Trackers
        self.survival = False  # Tracks done, seeks to prevent done unless potential reached
        self.safety = False  # Tracks rings, seeks to prevent ring lose
        self.belonging = False  # Tracks collision, seeks to avoid bad friends
        self.esteem = False  # Tracks reward, seeks
        self.potential = False  # Tracks completion || reward > 9000
        #Queues
        self.memory = deque(maxlen=2000)
        self.is_stuck_pos = deque(maxlen=2000)
        self.is_stuck_pos.append(0)
        self.in_danger_pos = deque(maxlen=2000)
        self.in_danger_pos.append(0)
        self.track_evolution = deque(maxlen=2000)
        # Storage
        self.table = pd.read_sql_query("SELECT * from game_stats", conn)





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
        self._cur_x = 0
        self._max_x = 0
        self.total_reward = 0
        self.steps = 0
        return self.env.reset(**kwargs)

    def resume_rl(self, a=True):
        self.rl = a
        self.resume = a

    def maslow(self, done, info):
        if (self.total_reward >= completion) and done:
            self.potential = True
        else:
            self.potential = False
        if not done:
            self.survival = True
        if (int(self.reward_history[-2]) <= int(self.reward_history[-1])):
            self.esteem = True
            if self._cur_x == self._max_x and self.assist:  # If rewards are progressing have the computer resume.
                self.trainer = False
        else:  # If esteem is low (i.e, stuck, in time loop, new scenario ask for help)
            if self.assist:
                if not done:
                    self.trainer = True
            else:
                self.esteem = False
                if self._cur_x == self._max_x and self.assist:
                    self.trainer = False
                self.resume_rl()
                self.is_stuck_pos.append(self.total_reward)
        if info['rings'] > 0:
            self.safety = True
        else:
            self.safety = False

    def process_frames(self, obs):
        x_t1 = skimage.color.rgb2gray(obs)
        # x_t1 = skimage.transform.resize(x_t1, (84, 84))
        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        return x_t1

    # PROCESSING
    def get_rew_hist(self, a):
        acts = np.array(self.rl_reward_history)
        acts = np.sum(acts[-a:])
        acts = np.round(acts, 4)
        return acts

    def insert_stats(self,action,obs,rew,done,info):
        # tm = trained_model.predict(self.table[-1])
        self.action_history.append(action.copy())
        self.rl_reward_history.append(rew)  # True Reward
        obs = self.process_frames(obs)
        self.total_reward += rew
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)  # Net 0 reward
        self._max_x = max(self._max_x, self._cur_x)  #Max level reached
        self.reward_history.append(self.total_reward)
        if self.steps > 1:
            self.maslow(done, info)
        if done:
            self.game_over = True
            self.episode += 1
        self.steps += 1
        cur_action = str(self.action_history[-1])
        if self.steps > 5:
            prev_action = str(self.action_history[-2])
            action_cluster = str(self.action_history[-5:])
            db.execute("INSERT INTO sarsa VALUES (NULL,?,?, ?,?,?)",
                       (self.level_choice, action_cluster, self.get_rew_hist(5), self.get_rew_hist(1), self.esteem))
            conn.commit()
        else:
            prev_action = ''
        acts1 = self.get_rew_hist(1)
        acts3 = self.get_rew_hist(3)
        acts5 = self.get_rew_hist(5)
        acts7 = self.get_rew_hist(7)
        acts9 = self.get_rew_hist(9)
        acts11 = self.get_rew_hist(11)
        acts33 = self.get_rew_hist(33)
        db.execute("INSERT INTO game_stats VALUES (NULL,?,?, ?, ?,?,?, ?,?,?,?,?,?,?,?,?,?,?,?)",
                   (self.level_choice, self.episode, self.steps, cur_action, prev_action
                    , acts1, acts3, acts5, acts7,acts9, acts11, acts33
                    , int(self.safety), int(self.esteem), int(self.belonging), int(self.potential)
                    ,int(self.trainer), int(self.total_reward)))
        conn.commit()
        # return tm

    def control(self, a=None):  # Enable Disrete Actions pylint: disable=W0221
        if not a:
            a = self.action_space.sample()
        self.step(self._actions[a].copy())
        return self._actions[a].copy()

    def step(self, action,*args):
        test = np.array(action).ndim
        if test < 1:
            action = self._actions[action].copy()
        self.total_steps_ever += 1
        obs, rew, done, info = self.env.step(action)
        self.insert_stats(action,obs,rew,done,info)
        return obs, rew, done, info


if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)